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


LOW_CONFIDENCE_THRESHOLD = 0.35
MIN_CONTEXT_RESULTS_FOR_CONFIDENCE = 1


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
        context_results = await self._retrieve_context(
            rag_query, db, query_embedding
        )

        # Step 3: Construct prompt with retrieved context
        prompt = await self._construct_prompt(rag_query, context_results)

        # Evaluate confidence before calling the model
        confidence_score = self._calculate_confidence_score(context_results)

        if (
            confidence_score < LOW_CONFIDENCE_THRESHOLD
            or len(context_results) < MIN_CONTEXT_RESULTS_FOR_CONFIDENCE
        ):
            logger.info(
                "Low confidence (score=%.3f, contexts=%d); returning fallback response.",
                confidence_score,
                len(context_results),
            )
            fallback_text = (
                "I'm sorry, but I couldn't find enough information in the knowledge bases to answer that. "
                "Please upload relevant documents or clarify your question."
            )
            citations: List[Dict[str, Any]] = []
            return RAGResponse(
                answer=fallback_text,
                context=context_results,
                metadata={
                    "query_text": rag_query.text,
                    "context_count": len(context_results),
                    "search_type": rag_query.search_type,
                    "application_id": (
                        str(rag_query.application_id) if rag_query.application_id else None
                    ),
                    "query_embedding_dimension": len(query_embedding),
                    "confidence_score": confidence_score,
                    "fallback_reason": "low_context_confidence",
                    "is_fallback": True,
                    "citations": citations,
                },
                confidence_score=confidence_score,
            )

        # Step 4: Generate response
        response, provider_type, model_name = await self._generate_response(
            prompt, rag_query, application
        )

        has_error = bool(response.metadata and response.metadata.get("error"))
        citations = (
            []
            if has_error
            else self._build_citations(
                context_results, response.text, rag_query.text
            )
        )

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
                "confidence_score": confidence_score,
                "citations": citations,
                **model_metadata,
            },
            confidence_score=confidence_score,
        )

    def _build_citations(
        self,
        context_results: List[RetrievalResult],
        answer_text: Optional[str],
        query_text: Optional[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not context_results:
            return []

        highlight_terms = self._gather_highlight_terms(answer_text, query_text)

        scored: List[Tuple[float, float, RetrievalResult, str]] = []
        for result in context_results:
            snippet_source = (
                result.metadata.get("paragraph_excerpt")
                or result.metadata.get("paragraph_content")
                or result.content
                or ""
            )

            term_score = self._calculate_highlight_match_score(
                snippet_source, highlight_terms
            )
            scored.append((term_score, result.score, result, snippet_source))

        if not scored:
            return []

        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        positives = [t for t in scored if t[0] > 0]
        chosen = positives[:top_k] if positives else scored[: min(top_k, len(scored))]

        citations: List[Dict[str, Any]] = []
        for term_score, _, res, src in chosen:
            snippet = (
                self._format_snippet(src, highlight_terms)
                if term_score > 0
                else self._slice_snippet(src, [], 320)
            )
            citations.append(
                {
                    "document_id": res.metadata.get("document_id"),
                    "document_title": res.metadata.get("document_title"),
                    "paragraph_id": res.metadata.get("paragraph_id"),
                    "knowledge_base_id": res.metadata.get("knowledge_base_id"),
                    "score": res.score,
                    "snippet": snippet,
                }
            )

        return citations

    @staticmethod
    def _format_snippet(
        snippet_source: Optional[str], highlight_terms: List[Dict[str, Any]]
    ) -> str:
        import re

        if not snippet_source:
            return ""

        normalized = re.sub(r"\s+", " ", snippet_source).strip()
        term_strings = [term["text"] for term in highlight_terms if term["text"]]

        window = RAGPipeline._slice_snippet(normalized, term_strings)
        return RAGPipeline._apply_highlight(window, term_strings)

    @staticmethod
    def _gather_highlight_terms(
        answer_text: Optional[str], query_text: Optional[str]
    ) -> List[Dict[str, Any]]:
        import re

        stopwords = {
            "the",
            "and",
            "have",
            "with",
            "that",
            "this",
            "from",
            "will",
            "your",
            "been",
            "into",
            "more",
            "about",
            "than",
            "when",
            "what",
            "where",
            "which",
            "whose",
            "their",
            "there",
        }

        term_map: Dict[str, Dict[str, Any]] = {}

        def register(term: str, source: str, *, is_phrase: bool = False):
            if not term:
                return
            clean = term.strip()
            if not clean:
                return
            lower = clean.casefold()
            if lower in stopwords and not is_phrase:
                return
            is_numeric = any(ch.isdigit() for ch in clean)
            entry = term_map.get(lower)
            weight = RAGPipeline._term_weight(
                clean,
                from_query=(source == "query"),
                is_numeric=is_numeric,
                is_phrase=is_phrase,
            )
            if entry:
                entry["weight"] = max(entry["weight"], weight)
                entry["from_query"] = entry["from_query"] or (source == "query")
                entry["is_numeric"] = entry["is_numeric"] and is_numeric
                entry["is_phrase"] = entry["is_phrase"] or is_phrase
            else:
                term_map[lower] = {
                    "text": clean,
                    "weight": weight,
                    "from_query": source == "query",
                    "is_numeric": is_numeric,
                    "is_phrase": is_phrase,
                }

        def process_text(text: Optional[str], source: str):
            if not text:
                return

            # Numbers
            for match in re.findall(r"\d[\d,\.]*", text):
                register(match, source)

            # Words
            words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'\-]{1,}", text)
            filtered_words: List[str] = []
            for word in words:
                lower = word.casefold()
                if len(lower) < 4 and not any(ch.isdigit() for ch in word):
                    continue
                if lower in stopwords:
                    continue
                register(word, source)
                filtered_words.append(word)

            if source == "query" and filtered_words:
                for i in range(len(filtered_words) - 1):
                    first = filtered_words[i]
                    second = filtered_words[i + 1]
                    if not first or not second:
                        continue
                    phrase = f"{first} {second}"
                    register(phrase, source, is_phrase=True)

        process_text(answer_text, "answer")
        process_text(query_text, "query")

        return list(term_map.values())

    @staticmethod
    def _slice_snippet(text: str, terms: List[str], max_length: int = 320) -> str:
        if len(text) <= max_length:
            return text

        import re

        lower_text = text.casefold()
        accentless_text, index_map = RAGPipeline._build_accent_map(text)
        accentless_lower = accentless_text.casefold()

        for term in terms:
            if not term:
                continue
            term_lower = term.casefold()
            match = re.search(re.escape(term_lower), lower_text)
            start_index = None
            if match:
                start_index = match.start()
            else:
                accentless_term = RAGPipeline._remove_accents(term).casefold()
                if accentless_term:
                    match = re.search(
                        re.escape(accentless_term), accentless_lower
                    )
                    if match:
                        start_index = index_map[match.start()][0]

            if start_index is not None:
                half = max_length // 2
                start = max(0, start_index - half)
                end = min(len(text), start + max_length)
                if end - start < max_length:
                    start = max(0, end - max_length)
                snippet = text[start:end]
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                return f"{prefix}{snippet}{suffix}"

        return text[:max_length] + ("..." if len(text) > max_length else "")

    @staticmethod
    def _apply_highlight(text: str, terms: List[str]) -> str:
        from html import escape
        import re

        if not text:
            return ""

        matches: List[Tuple[int, int]] = []

        lower_text = text.casefold()
        for term in terms:
            if not term:
                continue
            term_lower = term.casefold()
            if not term_lower:
                continue
            for match in re.finditer(re.escape(term_lower), lower_text):
                matches.append((match.start(), match.start() + len(match.group(0))))

        accentless_text, index_map = RAGPipeline._build_accent_map(text)
        accentless_lower = accentless_text.casefold()
        for term in terms:
            if not term:
                continue
            accentless_term = RAGPipeline._remove_accents(term).casefold()
            if not accentless_term:
                continue
            for match in re.finditer(re.escape(accentless_term), accentless_lower):
                start_idx = index_map[match.start()][0]
                end_idx = index_map[match.end() - 1][1]
                matches.append((start_idx, end_idx))

        if not matches:
            return escape(text)

        matches.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in matches:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        result_parts: List[str] = []
        cursor = 0
        for start, end in merged:
            if start < cursor:
                continue
            result_parts.append(escape(text[cursor:start]))
            result_parts.append("<mark>")
            result_parts.append(escape(text[start:end]))
            result_parts.append("</mark>")
            cursor = end

        result_parts.append(escape(text[cursor:]))
        return "".join(result_parts)

    @staticmethod
    def _remove_accents(text: str) -> str:
        import unicodedata

        return "".join(
            ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
        )

    @staticmethod
    def _build_accent_map(text: str) -> Tuple[str, List[Tuple[int, int]]]:
        import unicodedata

        simple_chars: List[str] = []
        index_map: List[Tuple[int, int]] = []

        for idx, char in enumerate(text):
            normalized = unicodedata.normalize("NFKD", char)
            stripped = "".join(
                c for c in normalized if not unicodedata.combining(c)
            )
            if stripped:
                for _ in stripped:
                    simple_chars.append(_)
                    index_map.append((idx, idx + 1))
            else:
                simple_chars.append(char)
                index_map.append((idx, idx + 1))

        return "".join(simple_chars), index_map

    @staticmethod
    def _calculate_highlight_match_score(
        snippet_source: str, highlight_terms: List[Dict[str, Any]]
    ) -> float:
        if not snippet_source:
            return 0.0

        score = 0.0
        query_hits = 0
        total_hits = 0
        query_terms_available = any(
            not term["is_numeric"] and term["from_query"] for term in highlight_terms
        )

        for term in highlight_terms:
            text = term["text"]
            if not text:
                continue
            if RAGPipeline._contains_term(snippet_source, text):
                total_hits += 1
                score += term["weight"]
                if term["from_query"] and not term["is_numeric"]:
                    query_hits += 1

        if total_hits == 0:
            return 0.0

        if query_terms_available and query_hits == 0:
            score *= 0.2

        return score

    @staticmethod
    def _contains_term(snippet: str, term: str) -> bool:
        if not term:
            return False

        import re

        def _match(text: str, pattern: str) -> bool:
            try:
                return re.search(pattern, text) is not None
            except re.error:
                return pattern.casefold() in text

        def _pattern_for(t: str, numeric: bool, phrase: bool) -> str:
            escaped = re.escape(t)
            if numeric:
                return rf"(?<!\d){escaped}(?!\d)"
            if phrase or " " in t:
                return rf"\b{escaped}\b"
            return rf"\b{escaped}\b"

        is_numeric = any(ch.isdigit() for ch in term)
        is_phrase = " " in term

        lower_snippet = snippet.casefold()
        pattern = _pattern_for(term.casefold(), is_numeric, is_phrase)
        if _match(lower_snippet, pattern):
            return True

        accentless_snippet = RAGPipeline._remove_accents(snippet).casefold()
        accentless_term = RAGPipeline._remove_accents(term).casefold()
        if accentless_term:
            pattern_acc = _pattern_for(accentless_term, is_numeric, is_phrase)
            if _match(accentless_snippet, pattern_acc):
                return True

        return False

    @staticmethod
    def _term_weight(
        term: str, *, from_query: bool, is_numeric: bool, is_phrase: bool
    ) -> float:
        base = float(max(len(term.strip()), 1))
        if is_numeric:
            base *= 2.5
        if is_phrase:
            base += 2.0
        if from_query:
            base *= 2.0
        if term and term[0].isupper():
            base *= 1.3
        return base

    async def _retrieve_context(
        self,
        query: RAGQuery,
        db: AsyncSession,
        query_vector: Optional[List[float]] = None,
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
            query_vector=query_vector,
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

        scores = sorted((r.score for r in context), reverse=True)
        avg_score = sum(scores) / len(scores)
        top = scores[0]
        margin = (top - scores[1]) if len(scores) > 1 else top
        result_count_factor = min(len(context) / 5.0, 1.0)

        max_score = top if top > 0 else 1.0
        normalized_avg = avg_score / max_score
        normalized_margin = margin / max_score

        confidence = (
            (0.55 * normalized_avg)
            + (0.25 * result_count_factor)
            + (0.20 * normalized_margin)
        )

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
        confidence_score = self._calculate_confidence_score(context_results)
        if (
            confidence_score < LOW_CONFIDENCE_THRESHOLD
            or len(context_results) < MIN_CONTEXT_RESULTS_FOR_CONFIDENCE
        ):
            fallback_text = (
                "I'm sorry, but I couldn't find enough information in the knowledge bases to answer that. "
                "Please upload relevant documents or clarify your question."
            )
            logger.info(
                "Streaming fallback due to low confidence (score=%.3f, contexts=%d).",
                confidence_score,
                len(context_results),
            )
            yield fallback_text
            return

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
