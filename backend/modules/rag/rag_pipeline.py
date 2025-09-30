import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from uuid import UUID
from dataclasses import dataclass
from sqlalchemy.ext.asyncio import AsyncSession

from .retrieval import retrieval_service, RetrievalResult
from .embedding import encode_text
from ..applications.models import Application
from ..models.model_manager import model_manager
from shared.models.Model import ProviderType, GenerationRequest, GenerationResponse
from backend.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """Query object for RAG processing."""

    text: str
    application_id: Optional[UUID] = None
    knowledge_base_ids: Optional[List[UUID]] = None
    search_type: str = "hybrid"
    max_context_length: int = 4000
    temperature: float = 0.7
    model_name: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    answer: str
    context: List[RetrievalResult]
    metadata: Dict[str, Any]
    confidence_score: float


class RAGPipeline:
    """RAG pipeline for processing queries with retrieved context."""

    def __init__(
        self, max_context_length: int = 4000, default_temperature: float = 0.7
    ):
        """
        Initialize the RAG pipeline.

        Args:
            max_context_length: Maximum length of context to include in prompt
            default_temperature: Default temperature for response generation
        """
        self.max_context_length = max_context_length
        self.default_temperature = default_temperature

    async def process_query(
        self,
        query: Union[str, RAGQuery],
        db: AsyncSession,
        application: Optional[Application] = None,
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.

        Args:
            query: Query text or RAGQuery object
            db: Database session
            application: Optional application configuration

        Returns:
            RAG response with answer and context
        """
        # Convert string query to RAGQuery object
        if isinstance(query, str):
            rag_query = RAGQuery(
                text=query,
                max_context_length=self.max_context_length,
                temperature=self.default_temperature,
            )
        else:
            rag_query = query

        # Step 1: Process and embed the query
        logger.info(f"Processing query: {rag_query.text[:100]}...")

        # Generate query embedding for potential use in retrieval
        query_embedding = await encode_text(rag_query.text)

        # Step 2: Retrieve relevant context
        context_results = await self._retrieve_context(rag_query, db)

        # Step 3: Construct prompt with retrieved context
        prompt = await self._construct_prompt(rag_query, context_results)

        # Step 4: Generate response
        response, provider_type, model_name = await self._generate_response(
            prompt, rag_query, application
        )

        # Calculate confidence score based on retrieval results
        confidence_score = self._calculate_confidence_score(context_results)

        model_metadata = {
            "provider": provider_type.value if provider_type else None,
            "model_name": response.model_name if response.model_name else model_name,
        }

        if model_metadata["provider"] and model_metadata["model_name"]:
            try:
                available_models = await model_manager.get_available_models(
                    ProviderType(model_metadata["provider"])
                )
                match = next(
                    (
                        m
                        for m in available_models
                        if m.get("name") == model_metadata["model_name"]
                    ),
                    None,
                )
                if match:
                    model_metadata["model_display_name"] = (
                        match.get("display_name") or match.get("name")
                    )
                    model_metadata["model_is_free"] = match.get("is_free")
            except Exception as lookup_error:
                logger.debug(
                    "Unable to enrich model metadata for %s/%s: %s",
                    model_metadata["provider"],
                    model_metadata["model_name"],
                    lookup_error,
                )

        return RAGResponse(
            answer=response.text,
            context=context_results,
            metadata={
                "query_text": rag_query.text,
                "context_count": len(context_results),
                "search_type": rag_query.search_type,
                "application_id": (
                    str(rag_query.application_id) if rag_query.application_id else None
                ),
                "query_embedding_dimension": len(query_embedding),
                **model_metadata,
            },
            confidence_score=confidence_score,
        )

    async def _retrieve_context(
        self, query: RAGQuery, db: AsyncSession
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant context for the query.

        Args:
            query: RAG query object
            db: Database session

        Returns:
            List of retrieval results
        """
        # Use retrieval service to get context
        context_results = await retrieval_service.retrieve_with_context(
            query=query.text,
            db=db,
            knowledge_base_ids=query.knowledge_base_ids,
            application_id=query.application_id,
            context_window=query.max_context_length // 4,  # Reserve space for prompt
            search_type=query.search_type,
        )

        # Filter and rank results
        filtered_results = await retrieval_service.filter_and_rank_results(
            context_results,
            min_score=0.1,  # Minimum relevance threshold
            max_results=5,  # Limit context items
        )

        logger.info(f"Retrieved {len(filtered_results)} context items")
        return filtered_results

    async def _construct_prompt(
        self, query: RAGQuery, context: List[RetrievalResult]
    ) -> str:
        """
        Construct a prompt with retrieved context.

        Args:
            query: RAG query object
            context: Retrieved context results

        Returns:
            Formatted prompt string
        """
        # Build context string
        context_parts = []
        total_length = 0

        for i, result in enumerate(context):
            context_item = f"[Context {i+1}] {result.content}"
            if total_length + len(context_item) > query.max_context_length:
                break
            context_parts.append(context_item)
            total_length += len(context_item)

        context_text = "\n\n".join(context_parts)

        # Construct the prompt
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question accurately and helpfully.

Context:
{context_text}

Question: {query.text}

Instructions:
- Answer based on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise but comprehensive
- Use evidence from the context to support your answer

Answer:"""

        return prompt

    async def _generate_response(
        self, prompt: str, query: RAGQuery, application: Optional[Application] = None
    ) -> Tuple[GenerationResponse, ProviderType, str]:
        """
        Generate response using configured model provider.

        Args:
            prompt: Formatted prompt with context
            query: RAG query object
            application: Optional application configuration

        Returns:
            Tuple containing the generation response, provider, and model used
        """
        provider_type = ProviderType(settings.default_provider)
        model_name = settings.default_model

        try:
            # Override with query parameters if provided
            if query.provider:
                try:
                    provider_type = ProviderType(query.provider.lower())
                except ValueError:
                    logger.warning(f"Invalid provider {query.provider}, using default")
            if query.model_name:
                model_name = query.model_name

            # Override with application settings if available
            if application and hasattr(application, "model_provider"):
                provider_type = ProviderType(application.model_provider)
            if application and hasattr(application, "model_name"):
                model_name = application.model_name

            # Create model configuration
            model_config = await model_manager.get_model_config(
                provider_type=provider_type,
                model_name=model_name,
                temperature=query.temperature,
                max_tokens=min(
                    4096, query.max_context_length // 2
                ),  # Reserve space for context
            )

            # Create generation request
            generation_request = GenerationRequest(
                prompt=prompt,
                model_configuration=model_config,
                max_tokens=model_config.max_tokens,
                temperature=query.temperature,
                metadata={
                    "query_text": query.text,
                    "application_id": (
                        str(query.application_id) if query.application_id else None
                    ),
                    "context_length": len(prompt),
                },
            )

            # Generate response
            response = await model_manager.generate_text(generation_request)

            logger.info(
                f"Generated response using {provider_type.value}/{model_name}, "
                f"tokens: {response.usage.get('total_tokens', 0)}"
            )

            return response, provider_type, model_name

        except Exception as e:
            logger.error(f"Error generating response with model provider: {e}")
            # Fallback to simple response
            fallback_response = GenerationResponse(
                text=(
                    "I apologize, but I encountered an error while processing your "
                    f"request: {str(e)}. Please try again."
                ),
                model_name=model_name,
                provider=provider_type,
                usage={},
                metadata={"error": str(e)},
                finish_reason="error",
            )
            return fallback_response, provider_type, model_name

    def _calculate_confidence_score(self, context: List[RetrievalResult]) -> float:
        """
        Calculate confidence score based on retrieval results.

        Args:
            context: Retrieved context results

        Returns:
            Confidence score between 0 and 1
        """
        if not context:
            return 0.0

        # Simple confidence calculation based on:
        # - Number of relevant results
        # - Average similarity score
        # - Score distribution

        avg_score = sum(result.score for result in context) / len(context)
        result_count_factor = min(
            len(context) / 5.0, 1.0
        )  # Max confidence at 5+ results

        # Weight the factors
        confidence = (avg_score * 0.7) + (result_count_factor * 0.3)

        return min(confidence, 1.0)


class StreamingRAGPipeline(RAGPipeline):
    """Streaming version of RAG pipeline for real-time responses."""

    async def process_query_stream(
        self,
        query: Union[str, RAGQuery],
        db: AsyncSession,
        application: Optional[Application] = None,
    ):
        """
        Process a query with streaming response.

        Args:
            query: Query text or RAGQuery object
            db: Database session
            application: Optional application configuration

        Yields:
            Response chunks as they are generated
        """
        # Convert string query to RAGQuery object
        if isinstance(query, str):
            rag_query = RAGQuery(
                text=query,
                max_context_length=self.max_context_length,
                temperature=self.default_temperature,
            )
        else:
            rag_query = query

        # Step 1: Process and embed the query
        logger.info(f"Processing streaming query: {rag_query.text[:100]}...")

        # Step 2: Retrieve relevant context
        context_results = await self._retrieve_context(rag_query, db)

        # Step 3: Construct prompt with retrieved context
        prompt = await self._construct_prompt(rag_query, context_results)

        # Step 4: Generate streaming response
        async for chunk in self._generate_streaming_response(
            prompt, rag_query, application
        ):
            yield chunk

    async def _generate_streaming_response(
        self, prompt: str, query: RAGQuery, application: Optional[Application] = None
    ):
        """
        Generate streaming response.

        Args:
            prompt: Formatted prompt with context
            query: RAG query object
            application: Optional application configuration

        Yields:
            Response chunks
        """
        try:
            # Determine provider and model from application or defaults
            provider_type = ProviderType.OPENAI  # Default to OpenAI
            model_name = "gpt-3.5-turbo"  # Default model

            if application and hasattr(application, "model_provider"):
                provider_type = ProviderType(application.model_provider)
            if application and hasattr(application, "model_name"):
                model_name = application.model_name

            # Create model configuration
            model_config = await model_manager.get_model_config(
                provider_type=provider_type,
                model_name=model_name,
                temperature=query.temperature,
                max_tokens=min(4096, query.max_context_length // 2),
            )

            # Create generation request
            generation_request = GenerationRequest(
                prompt=prompt,
                model_configuration=model_config,
                stream=True,
                max_tokens=model_config.max_tokens,
                temperature=query.temperature,
                metadata={
                    "query_text": query.text,
                    "application_id": (
                        str(query.application_id) if query.application_id else None
                    ),
                    "context_length": len(prompt),
                },
            )

            # Generate streaming response
            async for chunk in model_manager.generate_text_stream(generation_request):
                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            # Fallback to simple streaming response
            error_msg = (
                f"I apologize, but I encountered an error: {str(e)}. Please try again."
            )
            for word in error_msg.split():
                yield word + " "
                await asyncio.sleep(0.05)


# Global RAG pipeline instances
rag_pipeline = RAGPipeline()
streaming_rag_pipeline = StreamingRAGPipeline()
