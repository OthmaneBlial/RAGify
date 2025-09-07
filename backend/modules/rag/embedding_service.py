import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from uuid import UUID
import torch
import gc

from sqlalchemy.ext.asyncio import AsyncSession
from ..knowledge.models import Paragraph, Document
from .embedding import encode_text, embedding_model
from ...core.cache import get_cached_embedding, set_cached_embedding

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProgress:
    """Progress tracking for embedding operations."""

    total: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    current_batch: int = 0
    total_batches: int = 0

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        return (self.processed / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            "total": self.total,
            "processed": self.processed,
            "successful": self.successful,
            "failed": self.failed,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "completion_percentage": self.completion_percentage,
        }


class EmbeddingService:
    """Service for managing embedding generation and storage."""

    def __init__(
        self,
        batch_size: int = 32,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_concurrent_batches: int = 2,
        memory_pool_size: int = 1024,  # MB
    ):
        """
        Initialize the embedding service.

        Args:
            batch_size: Number of texts to process in each batch
            max_retries: Maximum number of retries for failed embeddings
            retry_delay: Delay between retries in seconds
            max_concurrent_batches: Maximum concurrent batch processing
            memory_pool_size: Memory pool size in MB
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_batches = max_concurrent_batches
        self.memory_pool_size = memory_pool_size

        # GPU detection
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_memory = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )  # MB
            logger.info(f"GPU detected with {self.gpu_memory:.1f}MB memory")
        else:
            logger.info("No GPU detected, using CPU")

        # Memory pool for large document sets
        self.memory_pool = []
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

    async def generate_embeddings_for_paragraphs(
        self,
        paragraphs: List[Paragraph],
        db: AsyncSession,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[Dict[str, Any]], EmbeddingProgress]:
        """
        Generate embeddings for a list of paragraphs with progress tracking.

        Args:
            paragraphs: List of paragraph objects
            db: Database session
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (results, progress)
        """
        if not paragraphs:
            return [], EmbeddingProgress()

        progress = EmbeddingProgress(total=len(paragraphs))
        results = []

        # Extract texts and create mapping
        texts = [p.content for p in paragraphs]
        paragraph_map = {i: p for i, p in enumerate(paragraphs)}

        # Process in batches
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        progress.total_batches = total_batches

        for batch_idx in range(total_batches):
            progress.current_batch = batch_idx + 1

            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_paragraphs = [paragraph_map[i] for i in range(start_idx, end_idx)]

            batch_results = await self._process_batch_with_retry(
                batch_texts, batch_paragraphs, db, progress
            )

            results.extend(batch_results)

            # Update progress
            progress.processed += len(batch_texts)

            if progress_callback:
                await progress_callback(progress)

        return results, progress

    async def _process_batch_with_retry(
        self,
        texts: List[str],
        paragraphs: List[Paragraph],
        db: AsyncSession,
        progress: EmbeddingProgress,
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of texts with retry logic, caching, and GPU optimization.

        Args:
            texts: Batch of texts to encode
            paragraphs: Corresponding paragraph objects
            db: Database session
            progress: Progress tracker

        Returns:
            List of processing results
        """
        from ..knowledge.crud import create_embedding

        results = []
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for existing embeddings
        for i, text in enumerate(texts):
            cached = await get_cached_embedding(text)
            if cached:
                cached_embeddings.append((i, cached))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Process uncached texts
        if uncached_texts:
            async with self._semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        # GPU memory management
                        if self.gpu_available:
                            torch.cuda.empty_cache()

                        # Encode the batch
                        embeddings = await encode_text(
                            uncached_texts, batch_size=self.batch_size
                        )

                        # Cache the new embeddings
                        for text, embedding in zip(uncached_texts, embeddings):
                            await set_cached_embedding(text, embedding)

                        break

                    except Exception as e:
                        if attempt < self.max_retries:
                            logger.warning(
                                f"Batch encoding attempt {attempt + 1} failed: {e}. Retrying..."
                            )
                            await asyncio.sleep(
                                self.retry_delay * (2**attempt)
                            )  # Exponential backoff
                            # Clear memory on retry
                            if self.gpu_available:
                                torch.cuda.empty_cache()
                            gc.collect()
                        else:
                            logger.error(
                                f"Batch encoding failed after {self.max_retries + 1} attempts: {e}"
                            )
                            # Mark uncached as failed
                            for idx in uncached_indices:
                                results.append(
                                    {
                                        "success": False,
                                        "paragraph_id": paragraphs[idx].id,
                                        "error": f"Encoding failed: {str(e)}",
                                    }
                                )
                                progress.failed += 1
                            return results

        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        for i, embedding in cached_embeddings:
            all_embeddings[i] = embedding

        if uncached_texts:
            for i, embedding in zip(uncached_indices, embeddings):
                all_embeddings[i] = embedding

        # Store embeddings in database
        for i, (text, embedding, paragraph) in enumerate(
            zip(texts, all_embeddings, paragraphs)
        ):
            if embedding is None:
                progress.failed += 1
                results.append(
                    {
                        "success": False,
                        "paragraph_id": paragraph.id,
                        "error": "No embedding",
                    }
                )
                continue
            try:
                # Create embedding record
                embedding_record = await create_embedding(db, embedding, paragraph.id)

                results.append(
                    {
                        "success": True,
                        "paragraph_id": paragraph.id,
                        "embedding_id": embedding_record.id,
                        "text_length": len(text),
                        "vector_dimension": len(embedding),
                        "cached": i in [idx for idx, _ in cached_embeddings],
                    }
                )

                progress.successful += 1

            except Exception as e:
                logger.error(
                    f"Failed to store embedding for paragraph {paragraph.id}: {e}"
                )
                results.append(
                    {
                        "success": False,
                        "paragraph_id": paragraph.id,
                        "error": f"Storage failed: {str(e)}",
                    }
                )
                progress.failed += 1

        # Memory cleanup
        if self.gpu_available:
            torch.cuda.empty_cache()
        gc.collect()

        return results

    async def generate_embeddings_for_document(
        self,
        document: Document,
        db: AsyncSession,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[List[Dict[str, Any]], EmbeddingProgress]:
        """
        Generate embeddings for all paragraphs in a document.

        Args:
            document: Document object
            db: Database session
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results, progress)
        """
        from ..knowledge.crud import list_paragraphs_by_document

        # Get all paragraphs for the document
        paragraphs = await list_paragraphs_by_document(db, document.id)

        if not paragraphs:
            logger.warning(f"No paragraphs found for document {document.id}")
            return [], EmbeddingProgress()

        return await self.generate_embeddings_for_paragraphs(
            paragraphs, db, progress_callback
        )

    async def process_documents_batch(
        self,
        documents: List[Document],
        db: AsyncSession,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process multiple documents concurrently with memory management.

        Args:
            documents: List of documents to process
            db: Database session
            progress_callback: Optional progress callback

        Returns:
            Batch processing results
        """
        if not documents:
            return {"total_documents": 0, "results": []}

        # Process documents concurrently but limit concurrency to manage memory
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent document processing

        async def process_single_document(doc: Document) -> Dict[str, Any]:
            async with semaphore:
                try:
                    results, progress = await self.generate_embeddings_for_document(
                        doc, db, progress_callback
                    )
                    return {
                        "document_id": doc.id,
                        "success": True,
                        "paragraphs_processed": progress.processed,
                        "embeddings_created": progress.successful,
                        "failures": progress.failed,
                        "results": results,
                    }
                except Exception as e:
                    logger.error(f"Failed to process document {doc.id}: {e}")
                    return {
                        "document_id": doc.id,
                        "success": False,
                        "error": str(e),
                        "paragraphs_processed": 0,
                        "embeddings_created": 0,
                        "failures": 0,
                        "results": [],
                    }

        # Process all documents concurrently
        tasks = [process_single_document(doc) for doc in documents]
        document_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        successful = 0
        total_paragraphs = 0
        total_embeddings = 0
        total_failures = 0

        processed_results = []
        for result in document_results:
            if isinstance(result, Exception):
                logger.error(f"Document processing task failed: {result}")
                continue

            processed_results.append(result)
            if result["success"]:
                successful += 1
            total_paragraphs += result["paragraphs_processed"]
            total_embeddings += result["embeddings_created"]
            total_failures += result["failures"]

        return {
            "total_documents": len(documents),
            "successful_documents": successful,
            "total_paragraphs_processed": total_paragraphs,
            "total_embeddings_created": total_embeddings,
            "total_failures": total_failures,
            "results": processed_results,
        }

    async def clear_embedding_cache(self):
        """Clear the embedding model's cache and our cache to free memory."""
        embedding_model.clear_cache()

        # Clear GPU memory if available
        if self.gpu_available:
            torch.cuda.empty_cache()

        # Clear our memory pool
        self.memory_pool.clear()

        # Force garbage collection
        gc.collect()

        logger.info("All embedding caches cleared by service")


# Global embedding service instance
embedding_service = EmbeddingService()


async def generate_document_embeddings(
    document_id: UUID, db: AsyncSession, progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate embeddings for a document.

    Args:
        document_id: ID of the document
        db: Database session
        progress_callback: Optional progress callback

    Returns:
        Processing results
    """
    from ..knowledge.crud import get_document

    document = await get_document(db, document_id)
    if not document:
        raise ValueError(f"Document {document_id} not found")

    results, progress = await embedding_service.generate_embeddings_for_document(
        document, db, progress_callback
    )

    return {
        "document_id": document_id,
        "progress": progress.to_dict(),
        "results": results,
    }
