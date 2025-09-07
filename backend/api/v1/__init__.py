from fastapi import APIRouter
from .endpoints.knowledge import router as knowledge_router
from .endpoints.applications import router as applications_router
from .endpoints.chat import router as chat_router
from .endpoints.models import router as models_router

print("Loading API v1 router")
api_router = APIRouter()
print("Including knowledge router")
api_router.include_router(
    knowledge_router, prefix="/api/v1/knowledge", tags=["knowledge"]
)
print("Including applications router")
api_router.include_router(
    applications_router, prefix="/api/v1/applications", tags=["applications"]
)
print("Including chat router")
api_router.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
print("Including models router")
api_router.include_router(models_router, prefix="/api/v1/models", tags=["models"])
print("API v1 router loaded")
