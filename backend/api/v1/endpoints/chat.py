from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, Optional, AsyncGenerator
from uuid import UUID
from pydantic import BaseModel, Field
import logging

from backend.core.database import get_db
from backend.modules.applications.crud import (
    get_application_with_config,
    get_application_knowledge_bases,
    create_chat_message,
    get_application_chat_history,
    clear_chat_history,
)
from backend.modules.rag.rag_pipeline import streaming_rag_pipeline, RAGQuery

print("Loading chat router")
router = APIRouter()
print("Chat router loaded")


# Pydantic models for request/response
class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="User message")
    application_id: Optional[UUID] = Field(
        None, description="Application ID (optional, will use default if not provided)"
    )
    search_type: str = Field(
        "hybrid", description="Search type: semantic, keyword, or hybrid"
    )
    max_context_length: int = Field(4000, description="Maximum context length")
    temperature: float = Field(0.7, description="Response temperature")
    stream: bool = Field(True, description="Enable streaming response")
    model_name: Optional[str] = Field(
        None,
        description="Model name to use (optional, will use default if not provided)",
    )
    provider: Optional[str] = Field(
        None, description="Provider name (optional, will use default if not provided)"
    )


class ChatMessageResponse(BaseModel):
    message_id: str
    response: str
    context_count: int
    confidence_score: float
    metadata: Dict[str, Any]


class ConversationHistoryRequest(BaseModel):
    application_id: UUID = Field(..., description="Application ID")
    limit: int = Field(50, description="Maximum number of messages to return")
    before_message_id: Optional[UUID] = Field(
        None, description="Get messages before this ID"
    )


@router.post("/", response_model=ChatMessageResponse)
async def send_chat_message(
    request: ChatMessageRequest, db: AsyncSession = Depends(get_db)
):
    """
    Send a chat message and get a response.

    - **message**: User message
    - **application_id**: Application UUID (optional, will use default if not provided)
    - **search_type**: Search type (semantic, keyword, hybrid)
    - **max_context_length**: Maximum context length
    - **temperature**: Response temperature
    - **stream**: Enable streaming response (not used in this endpoint)
    """
    logging.info(f"Chat endpoint called with request: {request}")
    try:
        # Use default application if application_id is None
        application_id = request.application_id
        if application_id is None:
            # Get the first application or create a default one
            from backend.modules.applications.crud import (
                list_applications,
                create_application,
            )

            applications = await list_applications(db)

            # Check if default application already exists
            default_app = next(
                (
                    app
                    for app in applications
                    if app["name"] == "Default Chat Application"
                ),
                None,
            )

            if not default_app:
                logging.info("Creating default application...")
                default_app = await create_application(
                    db=db,
                    name="Default Chat Application",
                    description="Default application for chat functionality",
                    config={"provider": "openrouter", "model": "openai/gpt-5-nano"},
                    knowledge_base_ids=[],
                )
                application_id = default_app.id
                logging.info(f"Default application created with ID: {application_id}")
            else:
                application_id = UUID(default_app["id"])
                logging.info(
                    f"Using existing default application with ID: {application_id}"
                )

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
            model_name=request.model_name,
            provider=request.provider,
        )

        # Process query through RAG pipeline
        from backend.modules.rag.rag_pipeline import rag_pipeline

        rag_response = await rag_pipeline.process_query(rag_query, db)

        # Store chat message
        chat_message = await create_chat_message(
            db=db,
            application_id=application_id,
            user_message=request.message,
            bot_message=rag_response.answer,
        )

        return ChatMessageResponse(
            message_id=chat_message["id"],
            response=rag_response.answer,
            context_count=len(rag_response.context),
            confidence_score=rag_response.confidence_score,
            metadata=rag_response.metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process message: {str(e)}"
        )


@router.post("/message/stream")
async def send_chat_message_streaming(
    request: ChatMessageRequest, db: AsyncSession = Depends(get_db)
):
    """
    Send a chat message and get a streaming response.

    - **message**: User message
    - **application_id**: Application UUID
    - **search_type**: Search type (semantic, keyword, hybrid)
    - **max_context_length**: Maximum context length
    - **temperature**: Response temperature
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, request.application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        # Get knowledge base IDs for the application
        knowledge_base_ids = await get_application_knowledge_bases(
            db, request.application_id
        )

        # Create RAG query
        rag_query = RAGQuery(
            text=request.message,
            application_id=request.application_id,
            knowledge_base_ids=knowledge_base_ids if knowledge_base_ids else None,
            search_type=request.search_type,
            max_context_length=request.max_context_length,
            temperature=request.temperature,
            model_name=request.model_name,
            provider=request.provider,
        )

        # Store the user message first
        await create_chat_message(
            db=db,
            application_id=request.application_id,
            user_message=request.message,
            bot_message=None,  # Will be updated after streaming
        )

        # Return streaming response
        return StreamingResponse(
            stream_chat_response(rag_query, request.application_id, db),
            media_type="text/plain",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to start streaming: {str(e)}"
        )


async def stream_chat_response(
    rag_query: RAGQuery, application_id: UUID, db: AsyncSession
) -> AsyncGenerator[str, None]:
    """
    Stream chat response chunks.

    Args:
        rag_query: RAG query object
        application_id: Application ID
        db: Database session

    Yields:
        Response chunks
    """
    full_response = ""

    try:
        # Process streaming query
        async for chunk in streaming_rag_pipeline.process_query_stream(rag_query, db):
            full_response += chunk
            yield chunk

        # Store the complete bot response
        # Note: In a real implementation, you'd want to update the chat message
        # with the complete response after streaming is done

    except Exception as e:
        error_msg = f"\n\nError: {str(e)}"
        full_response += error_msg
        yield error_msg


@router.get("/history/{application_id}")
async def get_conversation_history(
    application_id: UUID, limit: int = 50, db: AsyncSession = Depends(get_db)
):
    """
    Get conversation history for an application.

    - **application_id**: Application UUID
    - **limit**: Maximum number of messages to return
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        history = await get_application_chat_history(db, application_id, limit)

        return {
            "application_id": str(application_id),
            "messages": history,
            "total_count": len(history),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.delete("/history/{application_id}")
async def clear_conversation_history(
    application_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Clear conversation history for an application.

    - **application_id**: Application UUID
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        deleted_count = await clear_chat_history(db, application_id)

        return {
            "message": "Conversation history cleared successfully",
            "deleted_count": deleted_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear history: {str(e)}"
        )


@router.get("/applications/{application_id}/stats")
async def get_chat_statistics(application_id: UUID, db: AsyncSession = Depends(get_db)):
    """
    Get chat statistics for an application.

    - **application_id**: Application UUID
    """
    try:
        # Verify application exists
        application_data = await get_application_with_config(db, application_id)
        if not application_data:
            raise HTTPException(status_code=404, detail="Application not found")

        history = await get_application_chat_history(db, application_id, limit=1000)

        # Calculate basic statistics
        total_messages = len(history)
        user_messages = len([m for m in history if m["user_message"]])
        bot_messages = len([m for m in history if m["bot_message"]])

        return {
            "application_id": str(application_id),
            "total_messages": total_messages,
            "user_messages": user_messages,
            "bot_messages": bot_messages,
            "conversations": user_messages,  # Assuming each user message starts a conversation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get statistics: {str(e)}"
        )


# WebSocket endpoint for real-time chat (optional)
@router.websocket("/ws/{application_id}")
async def websocket_chat(websocket, application_id: UUID):
    """
    WebSocket endpoint for real-time chat.

    - **application_id**: Application UUID
    """
    await websocket.accept()

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Process message (simplified)
            message = data.get("message", "")
            search_type = data.get("search_type", "hybrid")

            # Send response
            await websocket.send_json(
                {
                    "type": "response",
                    "message": f"Echo: {message}",
                    "search_type": search_type,
                }
            )

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()
