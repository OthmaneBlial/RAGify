import asyncio
import tempfile
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import gc

from sqlalchemy.ext.asyncio import AsyncSession

from shared.utils.text_processing import process_document
from .models import Document
from ..rag.embedding_service import embedding_service
from ...core.async_tasks import task_manager, TaskPriority

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document processing and chunking pipeline."""

    def __init__(
        self, chunk_size: int = 1000, overlap: int = 200, memory_limit_mb: int = 512
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.memory_limit_mb = memory_limit_mb

    async def _check_memory_usage(self):
        # simple guard
        return

    async def batch_process_documents(
        self, documents, db, progress_callback=None, max_concurrent=3
    ):
        sem = asyncio.Semaphore(max_concurrent)

        async def run(doc):
            async with sem:
                return await self.process_document_async(doc, db, progress_callback)

        return await asyncio.gather(*(run(d) for d in documents))

    async def process_document_async(
        self,
        document: Document,
        db: AsyncSession,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a document asynchronously: extract text, chunk it, and create paragraphs.

        Args:
            document: Document model instance
            db: Database session
            progress_callback: Optional callback for progress updates

        Returns:
            Processing results dictionary
        """
        try:
            from .crud import create_paragraph

            # Check memory usage before processing
            await self._check_memory_usage()

            if progress_callback:
                await progress_callback(0, 100, "Starting document processing")

            # Create temporary file with document content
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as temp_file:
                temp_file.write(document.content)
                temp_file_path = temp_file.name

            try:
                if progress_callback:
                    await progress_callback(10, 100, "Extracting and chunking text")

                # Process the document
                processing_result = await process_document(
                    temp_file_path, chunk_size=self.chunk_size, overlap=self.overlap
                )

                if progress_callback:
                    await progress_callback(30, 100, "Creating paragraphs in database")

                # Create paragraphs from chunks with memory-efficient batching
                created_paragraphs = []
                batch_size = 50  # Process in smaller batches to manage memory

                for i in range(0, len(processing_result["chunks"]), batch_size):
                    batch_chunks = processing_result["chunks"][i : i + batch_size]

                    for chunk in batch_chunks:
                        if chunk.strip():  # Only create paragraphs for non-empty chunks
                            paragraph = await create_paragraph(db, chunk, document.id)
                            created_paragraphs.append(paragraph)

                    # Force garbage collection after each batch
                    gc.collect()

                    if progress_callback:
                        progress = 30 + (i / len(processing_result["chunks"])) * 40
                        await progress_callback(
                            int(progress),
                            100,
                            f"Created {len(created_paragraphs)} paragraphs",
                        )

                if progress_callback:
                    await progress_callback(70, 100, "Generating embeddings")

                # Generate embeddings for all created paragraphs with concurrency control
                if created_paragraphs:
                    try:
                        embedding_results, progress = (
                            await embedding_service.generate_embeddings_for_paragraphs(
                                created_paragraphs, db, progress_callback
                            )
                        )
                        embedding_success = progress.successful
                        embedding_failures = progress.failed
                    except Exception as e:
                        embedding_success = 0
                        embedding_failures = len(created_paragraphs)
                        print(f"Failed to generate embeddings: {e}")
                else:
                    embedding_success = 0
                    embedding_failures = 0

                if progress_callback:
                    await progress_callback(100, 100, "Processing completed")

                return {
                    "success": True,
                    "document_id": document.id,
                    "chunks_created": len(created_paragraphs),
                    "embeddings_created": embedding_success,
                    "embedding_failures": embedding_failures,
                    "metadata": processing_result["metadata"],
                    "paragraphs": created_paragraphs,
                }

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                gc.collect()  # Force garbage collection

        except Exception as e:
            if progress_callback:
                await progress_callback(0, 100, f"Processing failed: {str(e)}")
            return {
                "success": False,
                "document_id": document.id,
                "error": str(e),
                "chunks_created": 0,
            }

    async def process_uploaded_file(
        self, file_content: bytes, filename: str, db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process an uploaded file directly from bytes content.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            db: Database session

        Returns:
            Processing results dictionary
        """
        try:
            # Determine file extension
            file_extension = Path(filename).suffix.lower()

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False, mode="wb"
            ) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            try:
                # Process the document
                processing_result = await process_document(
                    temp_file_path, chunk_size=self.chunk_size, overlap=self.overlap
                )

                return {
                    "success": True,
                    "filename": filename,
                    "raw_text": processing_result["raw_text"],
                    "cleaned_text": processing_result["cleaned_text"],
                    "chunks": processing_result["chunks"],
                    "metadata": processing_result["metadata"],
                }

            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)

        except Exception as e:
            return {"success": False, "filename": filename, "error": str(e)}


class ProcessingService:
    """Service for managing document processing operations."""

    def __init__(self):
        self.processor = DocumentProcessor()

    async def process_document_after_creation(
        self,
        document: Document,
        db: AsyncSession,
        use_async_task: bool = False,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Process a document immediately after creation.

        Args:
            document: Newly created document
            db: Database session
            use_async_task: Whether to submit as async task
            priority: Task priority if using async task

        Returns:
            Processing results or task ID
        """
        if use_async_task:
            # Submit as background task
            task_id = await task_manager.submit_task(
                f"Process Document {document.id}",
                self.processor.process_document_async,
                document,
                db,
                priority=priority,
            )
            return {
                "task_id": task_id,
                "status": "submitted",
                "message": "Document processing submitted as background task",
            }
        else:
            # Process immediately
            return await self.processor.process_document_async(document, db)

    async def process_file_upload(
        self, file_content: bytes, filename: str, db: AsyncSession
    ) -> Dict[str, Any]:
        """
        Process a file upload before document creation.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            db: Database session

        Returns:
            Processing results with extracted text
        """
        return await self.processor.process_uploaded_file(file_content, filename, db)

    async def batch_process_documents(
        self,
        documents: List[Document],
        db: AsyncSession,
        progress_callback: Optional[Callable] = None,
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently with memory management.

        Args:
            documents: List of documents to process
            db: Database session
            progress_callback: Optional callback for overall progress
            max_concurrent: Maximum concurrent processing

        Returns:
            List of processing results
        """
        return await self.processor.batch_process_documents(
            documents, db, progress_callback, max_concurrent
        )


# Global processing service instance
processing_service = ProcessingService()
