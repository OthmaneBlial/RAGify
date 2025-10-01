import asyncio
import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .embedding import encode_text, embedding_model
from ..knowledge.crud import search_embeddings_by_similarity

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    content: str
    score: float
    metadata: Dict[str, Any]
    source: str


@dataclass
class RetrievalContext:
    """Context window for retrieved content."""

    content: str
    window_size: int = 1000
    overlap: int = 200

    def get_context_window(self, target_content: str, position: int = 0) -> str:
        """Extract context window around target content."""
        start = max(0, position - self.window_size // 2)
        end = min(len(self.content), position + self.window_size // 2)

        # Ensure we don't cut words in the middle
        if start > 0:
            # Find word boundary
            while start < len(self.content) and self.content[start] != " ":
                start += 1

        if end < len(self.content):
            # Find word boundary
            while end > 0 and self.content[end - 1] != " ":
                end -= 1

        return self.content[start:end].strip()


class RetrievalService:
    """Service for retrieving relevant content from knowledge bases."""

    def __init__(self, max_results: int = 10, similarity_threshold: float = 0.1):
        """
        Initialize the retrieval service.

        Args:
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
        """
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

    async def semantic_search(
        self,
        query: str,
        db: AsyncSession,
        knowledge_base_ids: Optional[List[UUID]] = None,
        application_id: Optional[UUID] = None,
        limit: Optional[int] = None,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using vector similarity.

        Args:
            query: Search query
            db: Database session
            knowledge_base_ids: Optional list of knowledge base IDs to search in
            limit: Maximum number of results

        Returns:
            List of retrieval results
        """
        if limit is None:
            limit = self.max_results

        # Generate embedding for query
        if query_vector is None:
            query_vector = await encode_text(query)

        results = []

        # Search in each knowledge base if specified, otherwise search all
        if knowledge_base_ids:
            for kb_id in knowledge_base_ids:
                kb_results = await search_embeddings_by_similarity(
                    db, query_vector, limit, kb_id, application_id, self.similarity_threshold
                )
                results.extend(kb_results)
        else:
            results = await search_embeddings_by_similarity(
                db, query_vector, limit, None, application_id, self.similarity_threshold
            )

        # Convert to RetrievalResult objects and sort by score
        retrieval_results = []
        for result in results:
            retrieval_result = RetrievalResult(
                content=result["paragraph_content"],
                score=result["similarity_score"],
                metadata={
                    "paragraph_id": str(result["paragraph_id"]),
                    "document_id": str(result["document_id"]),
                    "document_title": result["document_title"],
                    "knowledge_base_id": str(result["knowledge_base_id"]),
                    "paragraph_excerpt": result.get("paragraph_content"),
                },
                source=result["document_title"],
            )
            retrieval_results.append(retrieval_result)

        # Sort by score descending
        retrieval_results.sort(key=lambda x: x.score, reverse=True)

        return retrieval_results[:limit]

    async def keyword_search(
        self,
        query: str,
        db: AsyncSession,
        knowledge_base_ids: Optional[List[UUID]] = None,
        application_id: Optional[UUID] = None,
        limit: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Perform keyword-based search using text matching.

        Args:
            query: Search query
            db: Database session
            knowledge_base_ids: Optional list of knowledge base IDs to search in
            limit: Maximum number of results

        Returns:
            List of retrieval results
        """
        if limit is None:
            limit = self.max_results

        from sqlalchemy import text

        # Build query for keyword search
        sql_query = """
        SELECT
            p.id as paragraph_id,
            p.content as paragraph_content,
            p.document_id,
            d.title as document_title,
            d.knowledge_base_id,
            ts_rank_cd(to_tsvector('english', p.content), plainto_tsquery('english', :query)) as rank_score
        FROM paragraphs p
        JOIN documents d ON p.document_id = d.id
        WHERE to_tsvector('english', p.content) @@ plainto_tsquery('english', :query)
        """

        params = {"query": query}

        if knowledge_base_ids:
            placeholders = [f":kb_id_{i}" for i in range(len(knowledge_base_ids))]
            sql_query += f" AND d.knowledge_base_id IN ({', '.join(placeholders)})"
            for i, kb_id in enumerate(knowledge_base_ids):
                params[f"kb_id_{i}"] = kb_id

        if application_id:
            sql_query += " AND d.application_id = :app_id"
            params["app_id"] = application_id

        sql_query += " ORDER BY rank_score DESC LIMIT :limit"
        params["limit"] = limit

        result = await db.execute(text(sql_query), params)
        rows = result.fetchall()

        retrieval_results = []
        for row in rows:
            retrieval_result = RetrievalResult(
                content=row.paragraph_content,
                score=float(row.rank_score),
                metadata={
                    "paragraph_id": str(row.paragraph_id),
                    "document_id": str(row.document_id),
                    "document_title": row.document_title,
                    "knowledge_base_id": str(row.knowledge_base_id),
                    "paragraph_excerpt": row.paragraph_content,
                },
                source=row.document_title,
            )
            retrieval_results.append(retrieval_result)

        return retrieval_results

    async def hybrid_search(
        self,
        query: str,
        db: AsyncSession,
        knowledge_base_ids: Optional[List[UUID]] = None,
        application_id: Optional[UUID] = None,
        limit: Optional[int] = None,
        k: int = 60,
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            db: Database session
            knowledge_base_ids: Optional list of knowledge base IDs to search in
            limit: Maximum number of results
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results

        Returns:
            List of retrieval results
        """
        if limit is None:
            limit = self.max_results

        # Perform both searches concurrently
        semantic_task = self.semantic_search(
            query,
            db,
            knowledge_base_ids,
            application_id,
            limit * 2,
            query_vector=query_vector,
        )
        keyword_task = self.keyword_search(
            query, db, knowledge_base_ids, application_id, limit * 2
        )

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        # Combine and deduplicate results
        # Reciprocal Rank Fusion
        def to_rank_map(items: List[RetrievalResult]) -> Dict[str, int]:
            return {r.metadata["paragraph_id"]: idx + 1 for idx, r in enumerate(items)}

        sem_rank = to_rank_map(semantic_results)
        kw_rank = to_rank_map(keyword_results)

        combined: Dict[str, RetrievalResult] = {}
        for res in semantic_results + keyword_results:
            key = res.metadata["paragraph_id"]
            if key not in combined:
                combined[key] = res

        final_results: List[RetrievalResult] = []
        for key, res in combined.items():
            rs = sem_rank.get(key)
            rk = kw_rank.get(key)
            score = 0.0
            if rs:
                score += 1.0 / (k + rs)
            if rk:
                score += 1.0 / (k + rk)
            res.score = score
            final_results.append(res)

        if final_results:
            max_rrf = max(r.score for r in final_results) or 1.0
            for r in final_results:
                r.score = r.score / max_rrf

        # Sort by combined score and return top results
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:limit]

    async def retrieve_with_context(
        self,
        query: str,
        db: AsyncSession,
        knowledge_base_ids: Optional[List[UUID]] = None,
        application_id: Optional[UUID] = None,
        context_window: int = 1000,
        search_type: str = "hybrid",
        query_vector: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve content with context windows.

        Args:
            query: Search query
            db: Database session
            knowledge_base_ids: Optional list of knowledge base IDs to search in
            context_window: Size of context window around retrieved content
            search_type: Type of search ('semantic', 'keyword', 'hybrid')

        Returns:
            List of retrieval results with context
        """
        # Perform search based on type
        if search_type == "semantic":
            results = await self.semantic_search(
                query,
                db,
                knowledge_base_ids,
                application_id,
                query_vector=query_vector,
            )
        elif search_type == "keyword":
            results = await self.keyword_search(query, db, knowledge_base_ids, application_id)
        elif search_type == "hybrid":
            results = await self.hybrid_search(
                query,
                db,
                knowledge_base_ids,
                application_id,
                query_vector=query_vector,
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        # Add context to results
        for result in results:
            # Get full document content for context
            from ..knowledge.models import Document

            doc_result = await db.execute(
                select(Document).where(
                    Document.id == UUID(result.metadata["document_id"])
                )
            )
            document = doc_result.scalar_one_or_none()

            if document:
                context_manager = RetrievalContext(document.content, context_window)
                # Find position of the paragraph in the document
                position = document.content.find(result.content)
                if position >= 0:
                    result.content = context_manager.get_context_window(
                        target_content=document.content,
                        position=position + len(result.content) // 2,
                    )

        return results

    async def filter_and_rank_results(
        self,
        results: List[RetrievalResult],
        min_score: Optional[float] = None,
        max_results: Optional[int] = None,
        diversity_threshold: float = 0.8,
    ) -> List[RetrievalResult]:
        """
        Filter and rank retrieval results.

        Args:
            results: List of retrieval results
            min_score: Minimum score threshold
            max_results: Maximum number of results to return
            diversity_threshold: Threshold for diversity filtering

        Returns:
            Filtered and ranked results
        """
        if min_score is None:
            min_score = self.similarity_threshold
        if max_results is None:
            max_results = self.max_results

        # Filter by score
        filtered_results = [r for r in results if r.score >= min_score]

        # Apply diversity filtering (remove very similar results)
        diverse_results = []
        for result in filtered_results:
            is_diverse = True
            for existing in diverse_results:
                # Simple diversity check based on content similarity
                if (
                    await self._calculate_content_similarity(
                        result.content, existing.content
                    )
                    > diversity_threshold
                ):
                    is_diverse = False
                    break

            if is_diverse:
                diverse_results.append(result)

        # Sort by score and limit results
        diverse_results.sort(key=lambda x: x.score, reverse=True)
        return diverse_results[:max_results]

    async def _calculate_content_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text snippets."""
        try:
            vectors = await embedding_model.encode_batch([text1, text2])
            return await embedding_model.get_similarity(vectors[0], vectors[1])
        except Exception as e:
            logger.warning(f"Error calculating content similarity: {e}")
            return 0.0


# Global retrieval service instance
retrieval_service = RetrievalService()
