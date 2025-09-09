from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Body, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field
import logging

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
from backend.modules.knowledge.crud import (
    create_document,
    list_documents_by_application,
    get_document,
)
from backend.modules.knowledge.processing import processing_service
from backend.modules.rag.rag_pipeline import (
    rag_pipeline,
    streaming_rag_pipeline,
    RAGQuery,
)
from shared.models.Document import Document

router = APIRouter()
logger = logging.getLogger(__name__)


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


# Application-specific document endpoints
@router.post("/{app_id}/documents/", response_model=Document)
async def upload_document_to_application(
    app_id: UUID, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    """
    Upload a document to a specific application.
    """
    try:
        # Verify application exists
        application = await get_application_with_config(db, app_id)
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")

        # Check if we have a default knowledge base for this application
        from backend.modules.knowledge.crud import list_knowledge_bases
        knowledge_bases = await list_knowledge_bases(db)
        if not knowledge_bases:
            # Create a default knowledge base
            from backend.modules.knowledge.crud import create_knowledge_base
            kb = await create_knowledge_base(
                db, "Default Knowledge Base", "Default knowledge base for uploads"
            )
            kb_id = kb.id
        else:
            kb_id = knowledge_bases[0].id

        # Check if knowledge base exists
        from backend.modules.knowledge.crud import get_knowledge_base
        kb = await get_knowledge_base(db, kb_id)
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        # Read file content
        file_content = await file.read()

        # Process the uploaded file to extract text
        processing_result = await processing_service.process_file_upload(
            file_content, file.filename, db
        )

        if not processing_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process file: {processing_result.get('error', 'Unknown error')}",
            )

        # Use extracted text as document content
        extracted_text = processing_result["cleaned_text"]

        # Create document with application_id
        doc = await create_document(db, file.filename, extracted_text, kb_id, app_id)
        logger.info(f"Document uploaded: {doc.title} for application {app_id}")

        # Return the Document model
        return Document.from_orm(doc)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/{app_id}/documents/")
async def list_application_documents(app_id: UUID, db: AsyncSession = Depends(get_db)):
    """
    List all documents for a specific application.
    """
    # Verify application exists
    application = await get_application_with_config(db, app_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    docs = await list_documents_by_application(db, app_id)
    logger.info(f"Retrieved {len(docs)} documents for application {app_id}")
    documents = []
    for doc in docs:
        from backend.modules.knowledge.crud import get_knowledge_base
        kb = await get_knowledge_base(db, doc.knowledge_base_id)
        kb_name = kb.name if kb else "Unknown KB"
        documents.append(
            {
                "id": str(doc.id),
                "filename": doc.title,
                "size": len(doc.content) if doc.content else 0,
                "upload_date": doc.created_at.isoformat() if doc.created_at else None,
                "knowledge_base_id": str(doc.knowledge_base_id),
                "knowledge_base_name": kb_name,
                "application_name": application["name"],
            }
        )
    return documents


@router.delete("/{app_id}/documents/{doc_id}")
async def delete_application_document(
    app_id: UUID, doc_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Delete a document from a specific application.
    """
    # Verify application exists
    application = await get_application_with_config(db, app_id)
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    # Check if document exists and belongs to the application
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.application_id != app_id:
        raise HTTPException(
            status_code=403, detail="Document does not belong to this application"
        )

    try:
        # Delete in correct order to handle foreign key constraints:
        # 1. Delete embeddings (referenced by paragraphs)
        # 2. Delete paragraphs (referenced by document)
        # 3. Delete document

        # Get all paragraphs for this document
        from backend.modules.knowledge.crud import list_paragraphs_by_document
        paragraphs = await list_paragraphs_by_document(db, doc_id)

        # Delete embeddings for each paragraph
        for paragraph in paragraphs:
            from backend.modules.knowledge.crud import get_embedding_by_paragraph
            embedding = await get_embedding_by_paragraph(db, paragraph.id)
            if embedding:
                await db.delete(embedding)

        # Delete all paragraphs for this document
        for paragraph in paragraphs:
            await db.delete(paragraph)

        # Finally delete the document
        await db.delete(doc)

        # Commit all changes
        await db.commit()

        return {"message": f"Document '{doc.title}' deleted successfully"}

    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
