import asyncio
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, List
import aiohttp
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from shared.models.Model import (
    ProviderType,
    ModelConfig,
    GenerationRequest,
    GenerationResponse,
    ModelInfo,
    ProviderSettings,
    TestConnectionRequest,
    TestConnectionResponse,
)

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    def __init__(self, settings: ProviderSettings):
        self.settings = settings
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.settings.timeout_seconds)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using the model."""

    @abstractmethod
    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        """Generate text with streaming."""

    @abstractmethod
    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        """Test connection to the provider."""

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""

    @abstractmethod
    def estimate_cost(self, tokens_used: int, model_name: str) -> float:
        """Estimate cost for token usage."""


class OpenAIProvider(ModelProvider):
    """OpenAI model provider implementation."""

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self.base_url = settings.base_url or "https://api.openai.com/v1"
        self.api_key = settings.api_key

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="gpt-3.5-turbo",
                provider=ProviderType.OPENAI,
                context_window=16385,
                supports_streaming=True,
                description="Fast and cost-effective model",
            ),
            ModelInfo(
                name="gpt-4",
                provider=ProviderType.OPENAI,
                context_window=8192,
                supports_streaming=True,
                description="Most capable model",
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=ProviderType.OPENAI,
                context_window=128000,
                supports_streaming=True,
                description="Latest GPT-4 with larger context",
            ),
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model_configuration.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "stream": False,
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.model_configuration.frequency_penalty is not None:
            payload["frequency_penalty"] = request.model_configuration.frequency_penalty
        if request.model_configuration.presence_penalty is not None:
            payload["presence_penalty"] = request.model_configuration.presence_penalty
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")

            data = await response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return GenerationResponse(
                text=choice["message"]["content"],
                model_name=request.model_configuration.model_name,
                provider=ProviderType.OPENAI,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason"),
                metadata=request.metadata,
            )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model_configuration.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "stream": True,
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenAI API error: {response.status} - {error_text}")

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        try:
            # Simple test by listing models
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    models = await response.json()
                    available_models = [m["id"] for m in models.get("data", [])]
                    if request.model_name in available_models:
                        return TestConnectionResponse(
                            success=True,
                            message="Connection successful",
                            model_info=next(
                                (
                                    m
                                    for m in self.get_available_models()
                                    if m.name == request.model_name
                                ),
                                None,
                            ),
                        )
                    else:
                        return TestConnectionResponse(
                            success=False,
                            message=f"Model {request.model_name} not available",
                            error="Model not found",
                        )
                else:
                    return TestConnectionResponse(
                        success=False,
                        message=f"API error: {response.status}",
                        error=await response.text(),
                    )
        except Exception as e:
            return TestConnectionResponse(
                success=False, message=f"Connection failed: {str(e)}", error=str(e)
            )

    def estimate_cost(self, tokens_used: int, model_name: str) -> float:
        # OpenAI pricing (approximate, per 1K tokens)
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }

        rates = pricing.get(model_name, {"input": 0.002, "output": 0.002})
        # Assume 50/50 split for simplicity
        input_tokens = tokens_used // 2
        output_tokens = tokens_used - input_tokens

        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


class AnthropicProvider(ModelProvider):
    """Anthropic model provider implementation."""

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self.base_url = settings.base_url or "https://api.anthropic.com"
        self.api_key = settings.api_key

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider=ProviderType.ANTHROPIC,
                context_window=200000,
                supports_streaming=True,
                description="Fast and efficient model",
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                provider=ProviderType.ANTHROPIC,
                context_window=200000,
                supports_streaming=True,
                description="Balanced performance and capability",
            ),
            ModelInfo(
                name="claude-3-opus-20240229",
                provider=ProviderType.ANTHROPIC,
                context_window=200000,
                supports_streaming=True,
                description="Most capable model",
            ),
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model_configuration.model_name,
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Anthropic API error: {response.status} - {error_text}"
                )

            data = await response.json()
            content = data["content"][0]["text"]
            usage = data.get("usage", {})

            return GenerationResponse(
                text=content,
                model_name=request.model_configuration.model_name,
                provider=ProviderType.ANTHROPIC,
                usage={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                },
                finish_reason=data.get("stop_reason"),
                metadata=request.metadata,
            )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model_configuration.model_name,
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Anthropic API error: {response.status} - {error_text}"
                )

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("type") == "content_block_delta":
                            yield chunk["delta"]["text"]
                    except json.JSONDecodeError:
                        continue

    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        try:
            # Test with a simple message
            test_request = GenerationRequest(
                prompt="Hello",
                model_config=ModelConfig(
                    provider=ProviderType.ANTHROPIC,
                    model_name=request.model_name,
                    api_key=self.api_key,
                ),
                max_tokens=10,
            )

            _ = await self.generate(test_request)
            return TestConnectionResponse(
                success=True,
                message="Connection successful",
                model_info=next(
                    (
                        m
                        for m in self.get_available_models()
                        if m.name == request.model_name
                    ),
                    None,
                ),
            )
        except Exception as e:
            return TestConnectionResponse(
                success=False, message=f"Connection failed: {str(e)}", error=str(e)
            )

    def estimate_cost(self, tokens_used: int, model_name: str) -> float:
        # Anthropic pricing (per 1K tokens)
        pricing = {
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
            "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
            "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
        }

        rates = pricing.get(model_name, {"input": 1.0, "output": 5.0})
        # Assume 50/50 split for simplicity
        input_tokens = tokens_used // 2
        output_tokens = tokens_used - input_tokens

        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


class GoogleProvider(ModelProvider):
    """Google Gemini model provider implementation."""

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self.base_url = settings.base_url or "https://generativelanguage.googleapis.com"
        self.api_key = settings.api_key

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="models/gemini-2.5-flash",
                provider=ProviderType.GOOGLE,
                context_window=1048576,  # 1M tokens
                supports_streaming=True,
                description="Fast and efficient Gemini model",
            ),
            ModelInfo(
                name="models/gemini-1.5-pro",
                provider=ProviderType.GOOGLE,
                context_window=2097152,  # 2M tokens
                supports_streaming=True,
                description="High-performance Gemini model",
            ),
            ModelInfo(
                name="models/gemini-1.5-flash",
                provider=ProviderType.GOOGLE,
                context_window=1048576,  # 1M tokens
                supports_streaming=True,
                description="Fast Gemini model",
            ),
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/v1beta/models/{request.model_configuration.model_name}:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": request.temperature
                or request.model_configuration.temperature,
                "maxOutputTokens": request.max_tokens
                or request.model_configuration.max_tokens,
            },
        }

        if request.model_configuration.top_p is not None:
            payload["generationConfig"]["topP"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["generationConfig"]["stopSequences"] = request.stop_sequences

        headers = {"Content-Type": "application/json"}

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Google Gemini API error: {response.status} - {error_text}"
                )

            data = await response.json()
            candidate = data["candidates"][0]
            content = candidate["content"]["parts"][0]["text"]

            # Estimate usage (Gemini doesn't provide exact token counts in response)
            estimated_tokens = len(request.prompt.split()) + len(content.split())

            return GenerationResponse(
                text=content,
                model_name=request.model_configuration.model_name,
                provider=ProviderType.GOOGLE,
                usage={"estimated_tokens": estimated_tokens},
                finish_reason=candidate.get("finishReason"),
                metadata=request.metadata,
            )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/v1beta/models/{request.model_configuration.model_name}:streamGenerateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": request.temperature
                or request.model_configuration.temperature,
                "maxOutputTokens": request.max_tokens
                or request.model_configuration.max_tokens,
            },
        }

        if request.model_configuration.top_p is not None:
            payload["generationConfig"]["topP"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["generationConfig"]["stopSequences"] = request.stop_sequences

        headers = {"Content-Type": "application/json"}

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"Google Gemini API error: {response.status} - {error_text}"
                )

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "candidates" in data:
                            candidate = data["candidates"][0]
                            if (
                                "content" in candidate
                                and "parts" in candidate["content"]
                            ):
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        yield part["text"]
                    except json.JSONDecodeError:
                        continue

    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        try:
            # Test with a simple message
            test_request = GenerationRequest(
                prompt="Hello",
                model_config=ModelConfig(
                    provider=ProviderType.GOOGLE,
                    model_name=request.model_name,
                    api_key=self.api_key,
                ),
                max_tokens=10,
            )

            _ = await self.generate(test_request)
            return TestConnectionResponse(
                success=True,
                message="Connection successful",
                model_info=next(
                    (
                        m
                        for m in self.get_available_models()
                        if m.name == request.model_name
                    ),
                    None,
                ),
            )
        except Exception as e:
            return TestConnectionResponse(
                success=False, message=f"Connection failed: {str(e)}", error=str(e)
            )

    def estimate_cost(self, tokens_used: int, model_name: str) -> float:
        # Google Gemini pricing (per 1K tokens, approximate)
        pricing = {
            "models/gemini-2.5-flash": {"input": 0.15, "output": 0.60},
            "models/gemini-1.5-pro": {"input": 1.25, "output": 5.0},
            "models/gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        }

        rates = pricing.get(model_name, {"input": 0.15, "output": 0.60})
        # Assume 50/50 split for simplicity
        input_tokens = tokens_used // 2
        output_tokens = tokens_used - input_tokens

        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


class OpenRouterProvider(ModelProvider):
    """OpenRouter model provider implementation."""

    def __init__(self, settings: ProviderSettings):
        super().__init__(settings)
        self.base_url = settings.base_url or "https://openrouter.ai/api/v1"
        self.api_key = settings.api_key

    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                name="openai/gpt-5-nano",
                provider=ProviderType.OPENROUTER,
                context_window=128000,
                supports_streaming=True,
                description="Fast and efficient OpenAI GPT-5 nano model",
            ),
            ModelInfo(
                name="openai/gpt-oss-20b",
                provider=ProviderType.OPENROUTER,
                context_window=4096,
                supports_streaming=True,
                description="Open-source 20B parameter model",
            ),
            ModelInfo(
                name="z-ai/glm-4-32b",
                provider=ProviderType.OPENROUTER,
                context_window=131072,
                supports_streaming=True,
                description="Z-AI GLM-4 32B model",
            ),
            ModelInfo(
                name="google/gemini-2.5-flash-lite",
                provider=ProviderType.OPENROUTER,
                context_window=1048576,
                supports_streaming=True,
                description="Google Gemini 2.5 Flash Lite",
            ),
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    )
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.com",  # Replace with your app's URL
            "X-Title": "RAGify",  # Replace with your app's name
        }

        payload = {
            "model": request.model_configuration.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "stream": False,
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.model_configuration.frequency_penalty is not None:
            payload["frequency_penalty"] = request.model_configuration.frequency_penalty
        if request.model_configuration.presence_penalty is not None:
            payload["presence_penalty"] = request.model_configuration.presence_penalty
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"OpenRouter API error: {response.status} - {error_text}"
                )

            data = await response.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})

            return GenerationResponse(
                text=choice["message"]["content"],
                model_name=request.model_configuration.model_name,
                provider=ProviderType.OPENROUTER,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                finish_reason=choice.get("finish_reason"),
                metadata=request.metadata,
            )

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncGenerator[str, None]:
        if not self.session:
            raise RuntimeError("Provider not initialized. Use async context manager.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app.com",
            "X-Title": "RAGify",
        }

        payload = {
            "model": request.model_configuration.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens or request.model_configuration.max_tokens,
            "temperature": request.temperature
            or request.model_configuration.temperature,
            "stream": True,
        }

        if request.model_configuration.top_p is not None:
            payload["top_p"] = request.model_configuration.top_p
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        async with self.session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(
                    f"OpenRouter API error: {response.status} - {error_text}"
                )

            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    async def test_connection(
        self, request: TestConnectionRequest
    ) -> TestConnectionResponse:
        try:
            # Simple test by listing models
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    models_data = await response.json()
                    available_models = [m["id"] for m in models_data.get("data", [])]
                    if request.model_name in available_models:
                        return TestConnectionResponse(
                            success=True,
                            message="Connection successful",
                            model_info=next(
                                (
                                    m
                                    for m in self.get_available_models()
                                    if m.name == request.model_name
                                ),
                                None,
                            ),
                        )
                    else:
                        return TestConnectionResponse(
                            success=False,
                            message=f"Model {request.model_name} not available",
                            error="Model not found",
                        )
                else:
                    return TestConnectionResponse(
                        success=False,
                        message=f"API error: {response.status}",
                        error=await response.text(),
                    )
        except Exception as e:
            return TestConnectionResponse(
                success=False, message=f"Connection failed: {str(e)}", error=str(e)
            )

    def estimate_cost(self, tokens_used: int, model_name: str) -> float:
        # OpenRouter pricing (approximate, per 1K tokens)
        pricing = {
            "openai/gpt-5-nano": {"input": 0.0001, "output": 0.0002},
            "openai/gpt-oss-20b": {"input": 0.001, "output": 0.002},
            "z-ai/glm-4-32b": {"input": 0.002, "output": 0.004},
            "google/gemini-2.5-flash-lite": {"input": 0.00015, "output": 0.0006},
        }

        rates = pricing.get(model_name, {"input": 0.001, "output": 0.002})
        # Assume 50/50 split for simplicity
        input_tokens = tokens_used // 2
        output_tokens = tokens_used - input_tokens

        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000


def get_provider_class(provider_type: ProviderType) -> type:
    """Get provider class for the given type."""
    providers = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.GOOGLE: GoogleProvider,
        ProviderType.OPENROUTER: OpenRouterProvider,
    }
    return providers[provider_type]
