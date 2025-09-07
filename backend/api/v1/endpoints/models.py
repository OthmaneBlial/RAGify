import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_db
from backend.modules.models.model_manager import model_manager
from shared.models.Model import (
    ProviderType,
    GenerationRequest,
    GenerationResponse,
    TestConnectionRequest,
    TestConnectionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[Dict[str, Any]])
async def list_available_models(provider: Optional[ProviderType] = None):
    """
    Get list of available models.

    Args:
        provider: Optional provider filter

    Returns:
        List of available models with metadata
    """
    try:
        models = model_manager.get_available_models(provider)
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/test", response_model=TestConnectionResponse)
async def test_model_connection(
    request: TestConnectionRequest, db: AsyncSession = Depends(get_db)
):
    """
    Test connection to a model provider.

    Args:
        request: Test connection request
        db: Database session

    Returns:
        Test connection response
    """
    try:
        response = await model_manager.test_connection(request)
        return response
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        raise HTTPException(status_code=500, detail=f"Connection test failed: {str(e)}")


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate text using specified model.

    Args:
        request: Generation request with model config and prompt
        db: Database session

    Returns:
        Generation response with text and metadata
    """
    try:
        # Validate model configuration
        if not request.model_configuration.provider:
            raise HTTPException(status_code=400, detail="Provider must be specified")

        if not request.model_configuration.model_name:
            raise HTTPException(status_code=400, detail="Model name must be specified")

        if not request.prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        # Generate text
        response = await model_manager.generate_text(request)

        # Log usage for monitoring
        logger.info(
            f"Generated text using {request.model_configuration.provider}/{request.model_configuration.model_name}, "
            f"tokens: {response.usage.get('total_tokens', 0)}"
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.get("/cost-estimate")
async def estimate_cost(tokens_used: int, model_name: str, provider: ProviderType):
    """
    Estimate cost for token usage.

    Args:
        tokens_used: Number of tokens used
        model_name: Name of the model
        provider: Provider type

    Returns:
        Cost estimate in USD
    """
    try:
        cost = model_manager.estimate_cost(tokens_used, model_name, provider)
        return {"estimated_cost_usd": cost}
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        raise HTTPException(status_code=500, detail="Cost estimation failed")


@router.get("/providers")
async def list_providers():
    """
    Get list of configured providers.

    Returns:
        List of available providers
    """
    try:
        providers = []
        for provider_type in ProviderType:
            if await model_manager.get_provider(provider_type):
                providers.append(
                    {
                        "type": provider_type.value,
                        "name": provider_type.name,
                        "available": True,
                    }
                )
            else:
                providers.append(
                    {
                        "type": provider_type.value,
                        "name": provider_type.name,
                        "available": False,
                    }
                )

        return providers
    except Exception as e:
        logger.error(f"Error listing providers: {e}")
        raise HTTPException(status_code=500, detail="Failed to list providers")
