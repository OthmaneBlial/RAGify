from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.modules.applications.crud import (
    create_application,
    get_application_with_config,
    update_application,
    delete_application,
    list_applications,
    get_application_knowledge_bases,
    create_chat_message,
    get_application_chat_history,
)
from backend.modules.applications.service import ApplicationService
from backend.modules.rag.rag_pipeline import (
    rag_pipeline,
    streaming_rag_pipeline,
    RAGQuery,
)

router = APIRouter()


# Pydantic models for request/response
class ApplicationCreateRequest(BaseModel):
    name: str = Field(..., description="Application name")
    description: Optional[str] = Field(None, description="Application description")
    config: Dict[str, Any] = Field(
        default_factory=dict, description="Model configuration"
    )
    knowledge_base_ids: List[UUID] = Field(
        default_factory=list, description="Associated knowledge base IDs"
    )


class ApplicationUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Application name")
    description: Optional[str] = Field(None, description="Application description")
    config: Optional[Dict[str, Any]] = Field(None, description="Model configuration")
    knowledge_base_ids: Optional[List[UUID]] = Field(
        None, description="Associated knowledge base IDs"
    )


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    search_type: str = Field(
        "hybrid", description="Search type: semantic, keyword, or hybrid"
    )
    max_context_length: int = Field(4000, description="Maximum context length")
    temperature: float = Field(0.7, description="Response temperature")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    response: str
    context_count: int
    confidence_score: float
    metadata: Dict[str, Any]


@router.post("/", response_model=Dict[str, Any])
async def create_new_application(
    request: ApplicationCreateRequest, db: AsyncSession = Depends(get_db)
):
    """
    Create a new application.

    - **name**: Application name
    - **description**: Optional application description
    - **model_config**: Optional model configuration
    - **knowledge_base_ids**: Optional list of knowledge base IDs
    """
    try:
        application = await create_application(
            db=db,
            name=request.name,
            description=request.description,
            config=request.config,
            knowledge_base_ids=request.knowledge_base_ids,
        )

        return await get_application_with_config(db, application.id)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create application: {str(e)}"
        )


@router.get("/", response_model=List[Dict[str, Any]])
async def get_applications(db: AsyncSession = Depends(get_db)):
    """
    List all applications with their configurations.
    """
    try:
        return await list_applications(db)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list applications: {str(e)}"
        )


@router.get("/{application_id}", response_model=Dict[str, Any])
async def get_application(application_id: UUID, db: AsyncSession = Depends(get_db)):
    """
    Get a specific application by ID with its configuration.

    - **application_id**: Application UUID
    """
    try:
        application = await get_application_with_config(db, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        return application

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get application: {str(e)}"
        )


@router.put("/{application_id}", response_model=Dict[str, Any])
async def update_existing_application(
    application_id: UUID,
    request: ApplicationUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Update an existing application.

    - **application_id**: Application UUID
    - **name**: New application name
    - **description**: New application description
    - **model_config**: New model configuration
    - **knowledge_base_ids**: New knowledge base associations
    """
    try:
        application = await update_application(
            db=db,
            application_id=application_id,
            name=request.name,
            description=request.description,
            config=request.config,
            knowledge_base_ids=request.knowledge_base_ids,
        )

        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        return await get_application_with_config(db, application_id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update application: {str(e)}"
        )


@router.delete("/{application_id}")
async def delete_existing_application(
    application_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Delete an application.

    - **application_id**: Application UUID
    """
    try:
        # Check if this is the default application
        application = await get_application_with_config(db, application_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        # Prevent deletion of default application
        if application["name"] == "Default Chat Application":
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the default application. This application is required for the system to function.",
            )

        success = await delete_application(db, application_id)
        if not success:
            raise HTTPException(status_code=404, detail="Application not found")

        return {"message": "Application deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete application: {str(e)}"
        )


@router.post("/{application_id}/chat", response_model=ChatResponse)
async def chat_with_application(
    application_id: UUID,
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Chat with an application using RAG.

    - **application_id**: Application UUID
    - **message**: User message
    - **search_type**: Search type (semantic, keyword, hybrid)
    - **max_context_length**: Maximum context length
    - **temperature**: Response temperature
    - **stream**: Enable streaming response
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        # Get knowledge base IDs for the application
        knowledge_base_ids = await get_application_knowledge_bases(db, application_id)

        # Create RAG query
        rag_query = RAGQuery(
            text=request.message,
            application_id=application_id,
            knowledge_base_ids=knowledge_base_ids if knowledge_base_ids else None,
            search_type=request.search_type,
            max_context_length=request.max_context_length,
            temperature=request.temperature,
        )

        # Process query through RAG pipeline
        if request.stream:
            # For streaming, we'd return a streaming response
            # For now, process normally and return complete response
            rag_response = await streaming_rag_pipeline.process_query(rag_query, db)
        else:
            rag_response = await rag_pipeline.process_query(rag_query, db)

        # Store chat message in background
        background_tasks.add_task(
            create_chat_message,
            db=db,
            application_id=application_id,
            user_message=request.message,
            bot_message=rag_response.answer,
        )

        return ChatResponse(
            response=rag_response.answer,
            context_count=len(rag_response.context),
            confidence_score=rag_response.confidence_score,
            metadata=rag_response.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")


@router.get("/{application_id}/chat/history")
async def get_chat_history(
    application_id: UUID, limit: int = 50, db: AsyncSession = Depends(get_db)
):
    """
    Get chat history for an application.

    - **application_id**: Application UUID
    - **limit**: Maximum number of messages to return
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        history = await get_application_chat_history(db, application_id, limit)
        return {"messages": history}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get chat history: {str(e)}"
        )


@router.get("/{application_id}/knowledge-bases")
async def get_application_kb_associations(
    application_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Get knowledge bases associated with an application.

    - **application_id**: Application UUID
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        kb_ids = await get_application_knowledge_bases(db, application_id)
        return {"knowledge_base_ids": [str(kb_id) for kb_id in kb_ids]}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge bases: {str(e)}"
        )


@router.post("/{application_id}/knowledge-bases")
async def associate_knowledge_bases(
    application_id: UUID,
    knowledge_base_ids: List[UUID] = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    """
    Associate knowledge bases with an application.

    - **application_id**: Application UUID
    - **knowledge_base_ids**: List of knowledge base IDs to associate
    """
    try:
        service = ApplicationService(db)
        result = await service.associate_knowledge_bases(
            application_id, knowledge_base_ids
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to associate knowledge bases: {str(e)}"
        )
