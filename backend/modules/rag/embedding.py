import asyncio
import hashlib
import numpy as np
from typing import List, Dict, Optional, Union
import logging

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    )

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """SentenceTransformer-based embedding model with caching and async support."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the SentenceTransformer model to use
            cache_size: Maximum number of cached embeddings
        """
        self.model_name = model_name
        self.cache_size = cache_size
        self._model: Optional[SentenceTransformer] = None
        self._cache: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def _load_model(self) -> SentenceTransformer:
        """Load the model asynchronously if not already loaded."""
        if self._model is None:
            async with self._lock:
                if self._model is None:  # Double-check pattern
                    logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                    # Model loading is CPU-bound, run in thread pool
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(
                        None, SentenceTransformer, self.model_name
                    )
                    logger.info("Model loaded successfully")
        return self._model

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize_vector(self, vector) -> List[float]:
        """Normalize vector to unit length."""
        # Handle both numpy arrays and lists
        if isinstance(vector, list):
            vector = np.array(vector)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Ensure we return a list
        if hasattr(vector, "tolist"):
            return vector.tolist()
        else:
            return list(vector)

    async def encode_single(self, text: str, normalize: bool = True) -> List[float]:
        """
        Encode a single text into an embedding vector.

        Args:
            text: Text to encode
            normalize: Whether to normalize the vector

        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            return [0.0] * 384  # Return zero vector for empty text

        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        model = await self._load_model()

        # Encode in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, model.encode, text)

        if normalize:
            vector = self._normalize_vector(vector)

        # Ensure we return a list
        if hasattr(vector, "tolist"):
            embedding = vector.tolist()
        else:
            embedding = list(vector)

        # Cache the result
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = embedding

        return embedding

    async def encode_batch(
        self, texts: List[str], normalize: bool = True, batch_size: int = 32
    ) -> List[List[float]]:
        """
        Encode multiple texts into embedding vectors with batch processing.

        Args:
            texts: List of texts to encode
            normalize: Whether to normalize vectors
            batch_size: Size of batches for processing

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = await self._load_model()
        results = []

        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = []

            for text in batch_texts:
                if not text.strip():
                    batch_results.append([0.0] * 384)
                    continue

                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    batch_results.append(self._cache[cache_key])
                else:
                    batch_results.append(None)  # Placeholder for uncached

            # Encode uncached texts
            uncached_indices = [
                j for j, result in enumerate(batch_results) if result is None
            ]
            if uncached_indices:
                uncached_texts = [batch_texts[j] for j in uncached_indices]

                # Encode batch in thread pool
                loop = asyncio.get_event_loop()
                vectors = await loop.run_in_executor(None, model.encode, uncached_texts)

                # Process and cache results
                for j, (idx, vector) in enumerate(zip(uncached_indices, vectors)):
                    if normalize:
                        vector = self._normalize_vector(vector)

                    # Ensure we return a list
                    if hasattr(vector, "tolist"):
                        embedding = vector.tolist()
                    else:
                        embedding = list(vector)
                    batch_results[idx] = embedding

                    # Cache if space available
                    if len(self._cache) < self.cache_size:
                        cache_key = self._get_cache_key(uncached_texts[j])
                        self._cache[cache_key] = embedding

            results.extend(batch_results)

        return results

    async def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First embedding vector
            vector2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        # Check if vectors have the same shape
        if v1.shape != v2.shape:
            return 0.0

        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors


# Global embedding model instance
embedding_model = EmbeddingModel()


async def encode_text(
    text: Union[str, List[str]], normalize: bool = True, batch_size: int = 32
) -> Union[List[float], List[List[float]]]:
    """
    Convenience function to encode text(s) using the global embedding model.

    Args:
        text: Text or list of texts to encode
        normalize: Whether to normalize vectors
        batch_size: Batch size for multiple texts

    Returns:
        Embedding vector(s)
    """
    if isinstance(text, str):
        return await embedding_model.encode_single(text, normalize)
    else:
        return await embedding_model.encode_batch(text, normalize, batch_size)


async def get_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    vectors = await embedding_model.encode_batch([text1, text2])
    return await embedding_model.get_similarity(vectors[0], vectors[1])
