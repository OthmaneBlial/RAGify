import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.core.database import get_db
from backend.modules.models.model_manager import model_manager
from shared.models.Model import (
    ProviderType,
    GenerationRequest,
    GenerationResponse,
    TestConnectionRequest,
    TestConnectionResponse,
)


class SetCurrentModelRequest(BaseModel):
    model_name: str
    provider: str = "openrouter"

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


@router.get("/current")
async def get_current_model(db: AsyncSession = Depends(get_db)):
    """
    Get the current selected model from the default application.

    Returns:
        Current model configuration
    """
    try:
        from backend.modules.applications.crud import list_applications, get_application_with_config
        from uuid import UUID

        applications = await list_applications(db)

        # Find default application
        default_app = next(
            (
                app
                for app in applications
                if app["name"] == "Default Chat Application"
            ),
            None,
        )

        if not default_app:
            return {"model_name": None, "provider": None}

        # Get application config
        app_config = await get_application_with_config(db, UUID(default_app["id"]))
        if not app_config or not app_config.get("config", {}).get("model_config"):
            return {"model_name": None, "provider": None}

        model_config = app_config["config"]["model_config"]
        return {
            "model_name": model_config.get("model", model_config.get("model_name")),
            "provider": model_config.get("provider")
        }
    except Exception as e:
        logger.error(f"Error getting current model: {e}")
        raise HTTPException(status_code=500, detail="Failed to get current model")


@router.post("/current")
async def set_current_model(
    request: SetCurrentModelRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Set the current model for the default application.

    Args:
        model_name: Name of the model
        provider: Provider name

    Returns:
        Success message
    """
    try:
        from backend.modules.applications.crud import list_applications, update_application
        from uuid import UUID

        applications = await list_applications(db)

        # Find or create default application
        default_app = next(
            (
                app
                for app in applications
                if app["name"] == "Default Chat Application"
            ),
            None,
        )

        if not default_app:
            # Create default application if it doesn't exist
            from backend.modules.applications.crud import create_application
            default_app = await create_application(
                db=db,
                name="Default Chat Application",
                description="Default application for chat functionality",
                config={"provider": request.provider, "model": request.model_name},
                knowledge_base_ids=[],
            )
            app_id = default_app.id
        else:
            app_id = UUID(default_app["id"])
            # Update existing application
            await update_application(
                db=db,
                application_id=app_id,
                config={"provider": request.provider, "model": request.model_name}
            )

        return {"message": "Current model updated successfully"}
    except Exception as e:
        logger.error(f"Error setting current model: {e}")
        raise HTTPException(status_code=500, detail="Failed to set current model")


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
