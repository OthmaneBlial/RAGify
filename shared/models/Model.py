from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class ModelConfig(BaseModel):
    """Configuration for a model provider."""

    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)


class GenerationRequest(BaseModel):
    """Request for text generation."""

    prompt: str
    model_configuration: ModelConfig
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class GenerationResponse(BaseModel):
    """Response from text generation."""

    text: str
    model_name: str
    provider: ProviderType
    usage: Dict[str, int] = Field(default_factory=dict)  # tokens used
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """Information about available models."""

    name: str
    provider: ProviderType
    context_window: int = 0
    supports_streaming: bool = True
    description: Optional[str] = None
    display_name: Optional[str] = None
    pricing_prompt: Optional[str] = None
    pricing_completion: Optional[str] = None
    tags: Optional[List[str]] = None
    is_free: bool = False


class ProviderSettings(BaseModel):
    """Settings for a model provider."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_requests: int = Field(default=60, ge=1)
    rate_limit_window_seconds: int = Field(default=60, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)


class TestConnectionRequest(BaseModel):
    """Request to test model connection."""

    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None


class TestConnectionResponse(BaseModel):
    """Response from connection test."""

    success: bool
    message: str
    model_info: Optional[ModelInfo] = None
    error: Optional[str] = None
