import asyncio
import logging
import time
from typing import Dict, Optional, List, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager

from .providers import ModelProvider, get_provider_class
from shared.models.Model import (
    ProviderType,
    ModelConfig,
    GenerationRequest,
    GenerationResponse,
    ProviderSettings,
    TestConnectionRequest,
    TestConnectionResponse,
)
from backend.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Rate limiting information."""

    requests_made: int = 0
    window_start: float = 0.0
    last_request_time: float = 0.0


class ModelManager:
    """Central manager for model providers with rate limiting and caching."""

    def __init__(self):
        self.providers: Dict[ProviderType, ModelProvider] = {}
        self.rate_limits: Dict[ProviderType, RateLimitInfo] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize all configured providers."""
        if self._initialized:
            return

        # Initialize OpenAI provider
        if settings.openai_api_key:
            openai_settings = ProviderSettings(
                api_key=settings.openai_api_key,
                rate_limit_requests=getattr(settings, "openai_rate_limit_requests", 60),
                rate_limit_window_seconds=getattr(
                    settings, "openai_rate_limit_window", 60
                ),
            )
            self.providers[ProviderType.OPENAI] = get_provider_class(
                ProviderType.OPENAI
            )(openai_settings)
            self.rate_limits[ProviderType.OPENAI] = RateLimitInfo()

        # Initialize Anthropic provider
        if hasattr(settings, "anthropic_api_key") and settings.anthropic_api_key:
            anthropic_settings = ProviderSettings(
                api_key=settings.anthropic_api_key,
                rate_limit_requests=getattr(
                    settings, "anthropic_rate_limit_requests", 60
                ),
                rate_limit_window_seconds=getattr(
                    settings, "anthropic_rate_limit_window", 60
                ),
            )
            self.providers[ProviderType.ANTHROPIC] = get_provider_class(
                ProviderType.ANTHROPIC
            )(anthropic_settings)
            self.rate_limits[ProviderType.ANTHROPIC] = RateLimitInfo()

        # Initialize Google provider
        google_api_key = getattr(settings, "google_api_key", None) or getattr(
            settings, "gemini_api_key", None
        )
        if google_api_key:
            google_settings = ProviderSettings(
                api_key=google_api_key,
                rate_limit_requests=60,  # Google has generous rate limits
                rate_limit_window_seconds=60,
            )
            self.providers[ProviderType.GOOGLE] = get_provider_class(
                ProviderType.GOOGLE
            )(google_settings)
            self.rate_limits[ProviderType.GOOGLE] = RateLimitInfo()

        # Initialize OpenRouter provider
        if hasattr(settings, "openrouter_api_key") and settings.openrouter_api_key:
            openrouter_settings = ProviderSettings(
                api_key=settings.openrouter_api_key,
                rate_limit_requests=60,
                rate_limit_window_seconds=60,
            )
            self.providers[ProviderType.OPENROUTER] = get_provider_class(
                ProviderType.OPENROUTER
            )(openrouter_settings)
            self.rate_limits[ProviderType.OPENROUTER] = RateLimitInfo()

        self._initialized = True
        logger.info(f"Initialized {len(self.providers)} model providers")

    async def get_provider(
        self, provider_type: ProviderType
    ) -> Optional[ModelProvider]:
        """Get a provider instance, ensuring it's initialized."""
        if not self._initialized:
            await self.initialize()

        provider = self.providers.get(provider_type)
        if provider:
            await provider.ensure_session()
        return provider

    async def _check_rate_limit(self, provider_type: ProviderType) -> bool:
        """Check if request is within rate limits."""
        if provider_type not in self.rate_limits:
            return True

        rate_info = self.rate_limits[provider_type]
        current_time = time.time()

        # Reset window if needed
        if (
            current_time - rate_info.window_start
            >= self.providers[provider_type].settings.rate_limit_window_seconds
        ):
            rate_info.requests_made = 0
            rate_info.window_start = current_time

        # Check if under limit
        if (
            rate_info.requests_made
            >= self.providers[provider_type].settings.rate_limit_requests
        ):
            return False

        # Update counters
        rate_info.requests_made += 1
        rate_info.last_request_time = current_time
        return True

    async def _wait_for_rate_limit(self, provider_type: ProviderType):
        """Wait until rate limit allows request."""
        if provider_type not in self.rate_limits:
            return

        rate_info = self.rate_limits[provider_type]
        current_time = time.time()

        # Reset window if needed
        if (
            current_time - rate_info.window_start
            >= self.providers[provider_type].settings.rate_limit_window_seconds
        ):
            rate_info.requests_made = 0
            rate_info.window_start = current_time
            return

        # If at limit, wait until window resets
        if (
            rate_info.requests_made
            >= self.providers[provider_type].settings.rate_limit_requests
        ):
            wait_time = self.providers[
                provider_type
            ].settings.rate_limit_window_seconds - (
                current_time - rate_info.window_start
            )
            if wait_time > 0:
                logger.info(
                    f"Rate limit reached for {provider_type}, waiting {wait_time:.2f}s"
                )
                await asyncio.sleep(wait_time)
                # Reset after waiting
                rate_info.requests_made = 0
                rate_info.window_start = time.time()

    @asynccontextmanager
    async def provider_context(self, provider_type: ProviderType):
        """Context manager for provider usage with rate limiting."""
        await self._wait_for_rate_limit(provider_type)
        provider = await self.get_provider(provider_type)
        if not provider:
            raise ValueError(f"Provider {provider_type} not available")

        try:
            yield provider
        finally:
            pass  # Provider session management handled by provider itself

    async def generate_text(
        self, request: GenerationRequest, retry_on_rate_limit: bool = True
    ) -> GenerationResponse:
        """Generate text with automatic provider selection and rate limiting."""
        provider_type = request.model_configuration.provider
        provider = await self.get_provider(provider_type)

        if not provider:
            raise ValueError(f"Provider {provider_type} not configured")

        max_retries = 3 if retry_on_rate_limit else 1
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with self.provider_context(provider_type):
                    response = await provider.generate(request)

                    # Update rate limit info
                    if provider_type in self.rate_limits:
                        self.rate_limits[provider_type].requests_made += 1

                    return response

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check for rate limit errors
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1 and retry_on_rate_limit:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.warning(f"Rate limit hit, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue

                # For other errors, don't retry
                break

        raise last_exception or Exception("Generation failed after retries")

    async def generate_text_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""
        provider_type = request.model_configuration.provider
        provider = await self.get_provider(provider_type)

        if not provider:
            raise ValueError(f"Provider {provider_type} not configured")

        async with self.provider_context(provider_type):
            async for chunk in provider.generate_stream(request):
                yield chunk

    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        """Test connection to a provider."""
        provider = await self.get_provider(request.provider)
        if not provider:
            return TestConnectionResponse(
                success=False,
                message=f"Provider {request.provider} not configured",
                error="Provider not available",
            )

        # Override API key if provided in request
        if request.api_key:
            provider.settings.api_key = request.api_key

        async with self.provider_context(request.provider):
            return await provider.test_connection(request)

    async def get_available_models(
        self, provider_type: Optional[ProviderType] = None
    ) -> List[Dict]:
        """Get list of available models."""
        if not self._initialized:
            await self.initialize()

        models: List[Dict] = []

        if provider_type:
            providers_to_check = (
                [provider_type] if provider_type in self.providers else []
            )
        else:
            providers_to_check = list(self.providers.keys())

        for p_type in providers_to_check:
            provider = await self.get_provider(p_type)
            if not provider:
                continue

            try:
                provider_models = await provider.get_available_models()
            except Exception as exc:
                logger.error(
                    "Failed to retrieve models for provider %s: %s", p_type.value, exc
                )
                continue

            for model_info in provider_models:
                models.append(
                    {
                        "name": model_info.name,
                        "provider": model_info.provider.value,
                        "context_window": model_info.context_window,
                        "supports_streaming": model_info.supports_streaming,
                        "description": model_info.description,
                        "display_name": model_info.display_name,
                        "pricing_prompt": model_info.pricing_prompt,
                        "pricing_completion": model_info.pricing_completion,
                        "tags": model_info.tags,
                        "is_free": model_info.is_free,
                    }
                )

        models.sort(key=lambda m: (m.get("provider"), not m.get("is_free"), (m.get("display_name") or m.get("name") or "").lower()))

        return models

    def estimate_cost(
        self, tokens_used: int, model_name: str, provider_type: ProviderType
    ) -> float:
        """Estimate cost for token usage."""
        provider = self.providers.get(provider_type)
        if provider:
            return provider.estimate_cost(tokens_used, model_name)
        return 0.0

    def count_tokens(self, text: str, model_name: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: ~4 characters per token for English text
        # This is a rough estimate; in production, use actual tokenizer
        return len(text) // 4

    async def get_model_config(
        self, provider_type: ProviderType, model_name: str, **kwargs
    ) -> ModelConfig:
        """Create a model configuration."""
        return ModelConfig(
            provider=provider_type,
            model_name=model_name,
            api_key=getattr(settings, f"{provider_type.value}_api_key", None),
            **kwargs,
        )

    async def close(self):
        """Close all provider sessions."""
        for provider in self.providers.values():
            if provider.session:
                await provider.__aexit__(None, None, None)
        self.providers.clear()
        self._initialized = False


# Global model manager instance
model_manager = ModelManager()
