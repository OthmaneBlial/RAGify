from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from uuid import UUID
import logging

from backend.core.database import get_db
from backend.modules.knowledge.crud import (
    create_knowledge_base,
    get_knowledge_base,
    update_knowledge_base,
    delete_knowledge_base,
    list_knowledge_bases,
    create_document,
    list_documents_by_knowledge_base,
    list_documents_by_application,
    get_document,
    delete_document,
    search_paragraphs_by_text,
    get_embeddings_stats,
)
from backend.modules.applications.crud import get_application_with_config
from backend.modules.knowledge.processing import processing_service
from shared.models.KnowledgeBase import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBase,
)
from shared.models.Document import Document

router = APIRouter()
logger = logging.getLogger(__name__)


# Knowledge overview endpoint
@router.get("/")
async def get_knowledge_overview(
    application_id: Optional[UUID] = None, db: AsyncSession = Depends(get_db)
):
    """
    Get knowledge overview - documents list for frontend.
    Optionally filter by application_id.
    """
    try:
        if application_id:
            # Verify application exists
            application = await get_application_with_config(db, application_id)
            if not application:
                raise HTTPException(status_code=404, detail="Application not found")

            # Get documents for this application
            docs = await list_documents_by_application(db, application_id)
            documents = []
            for doc in docs:
                kb = await get_knowledge_base(db, doc.knowledge_base_id)
                kb_name = kb.name if kb else "Unknown KB"
                documents.append(
                    {
                        "id": str(doc.id),
                        "filename": doc.title,
                        "size": (
                            len(doc.content) if doc.content else 0
                        ),  # Approximate size
                        "upload_date": (
                            doc.created_at.isoformat() if doc.created_at else None
                        ),
                        "knowledge_base_id": str(doc.knowledge_base_id),
                        "knowledge_base_name": kb_name,
                        "application_name": application["name"],
                    }
                )
        else:
            # Get all knowledge bases
            knowledge_bases = await list_knowledge_bases(db)

            # Collect all documents from all knowledge bases
            documents = []
            for kb in knowledge_bases:
                # Get documents for this knowledge base
                docs = await list_documents_by_knowledge_base(db, kb.id)
                for doc in docs:
                    # Find which application this knowledge base belongs to
                    app_name = "Unknown Application"
                    try:
                        # This is a simplified approach - in a real implementation,
                        # you'd have a proper relationship between applications and knowledge bases
                        from backend.modules.applications.crud import list_applications
                        applications = await list_applications(db)
                        for app in applications:
                            kb_ids = app.get("config", {}).get("knowledge_base_ids", [])
                            if str(kb.id) in kb_ids:
                                app_name = app["name"]
                                break
                    except Exception:
                        pass

                    documents.append(
                        {
                            "id": str(doc.id),
                            "filename": doc.title,
                            "size": (
                                len(doc.content) if doc.content else 0
                            ),  # Approximate size
                            "upload_date": (
                                doc.created_at.isoformat() if doc.created_at else None
                            ),
                            "knowledge_base_id": str(kb.id),
                            "knowledge_base_name": kb.name,
                            "application_name": app_name,
                        }
                    )

        return documents
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge overview: {str(e)}"
        )


# Upload document endpoint
@router.post("/upload/")
async def upload_document_to_knowledge_base(
    file: UploadFile = File(...),
    application_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document to the default knowledge base.
    Optionally associate with an application.
    """
    try:
        if application_id:
            # Verify application exists
            application = await get_application_with_config(db, application_id)
            if not application:
                raise HTTPException(status_code=404, detail="Application not found")

        # Check if we have a default knowledge base
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

        # Create document with optional application_id
        doc = await create_document(db, file.filename, extracted_text, kb_id, application_id)

        return {
            "success": True,
            "document_id": str(doc.id),
            "filename": doc.title,
            "knowledge_base_id": str(kb_id),
            "application_id": str(application_id) if application_id else None,
            "message": f"Document '{file.filename}' uploaded successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload document: {str(e)}"
        )


# Knowledge Base endpoints
@router.post("/knowledge-bases/", response_model=KnowledgeBase)
async def create_kb(kb: KnowledgeBaseCreate, db: AsyncSession = Depends(get_db)):
    return await create_knowledge_base(db, kb.name, kb.description)


@router.get("/knowledge-bases/", response_model=List[KnowledgeBase])
async def list_kbs(db: AsyncSession = Depends(get_db)):
    return await list_knowledge_bases(db)


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBase)
async def get_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)):
    kb = await get_knowledge_base(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.put("/knowledge-bases/{kb_id}", response_model=KnowledgeBase)
async def update_kb(
    kb_id: UUID, kb_update: KnowledgeBaseUpdate, db: AsyncSession = Depends(get_db)
):
    kb = await update_knowledge_base(db, kb_id, kb_update.name, kb_update.description)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.delete("/knowledge-bases/{kb_id}")
async def delete_kb(kb_id: UUID, db: AsyncSession = Depends(get_db)):
    success = await delete_knowledge_base(db, kb_id)
    if not success:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"message": "Knowledge base deleted"}


# Document endpoints
@router.post("/knowledge-bases/{kb_id}/documents/", response_model=Document)
async def upload_document(
    kb_id: UUID, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)
):
    # Check if knowledge base exists
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

    # Create document with processed content
    doc = await create_document(db, file.filename, extracted_text, kb_id)
    return doc


@router.get("/knowledge-bases/{kb_id}/documents/", response_model=List[Document])
async def list_documents(kb_id: UUID, db: AsyncSession = Depends(get_db)):
    # Check if knowledge base exists
    kb = await get_knowledge_base(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    return await list_documents_by_knowledge_base(db, kb_id)


@router.get("/documents/{doc_id}/status")
async def get_document_processing_status(
    doc_id: UUID, db: AsyncSession = Depends(get_db)
):
    """Get the processing status of a document."""
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "document_id": doc.id,
        "processing_status": doc.processing_status,
        "title": doc.title,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
    }


@router.delete("/documents/{doc_id}")
async def delete_document_endpoint(doc_id: UUID, db: AsyncSession = Depends(get_db)):
    """Delete a document and all its associated data (paragraphs and embeddings)."""
    # Check if document exists
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

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


# Search endpoints
@router.post("/search/")
async def search_knowledge_bases(
    query: str,
    knowledge_base_id: Optional[UUID] = None,
    application_id: Optional[UUID] = None,
    limit: int = 10,
    threshold: Optional[float] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Perform semantic search across knowledge bases.

    Args:
        query: Search query text
        knowledge_base_id: Optional filter by specific knowledge base
        application_id: Optional filter by specific application
        limit: Maximum number of results
        threshold: Optional similarity threshold (0-1)
        db: Database session

    Returns:
        Search results with similarity scores
    """
    if not query.strip():
        return {"results": [], "query": query, "total_results": 0}

    try:
        if application_id:
            # Verify application exists
            application = await get_application_with_config(db, application_id)
            if not application:
                raise HTTPException(status_code=404, detail="Application not found")

        # Perform semantic search
        results = await search_paragraphs_by_text(db, query, limit, knowledge_base_id, application_id)

        # Filter by threshold if specified
        if threshold is not None:
            results = [r for r in results if r["similarity_score"] >= threshold]

        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
            "application_id": str(application_id) if application_id else None,
            "threshold": threshold,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/knowledge-bases/{kb_id}/embeddings/stats")
async def get_knowledge_base_embedding_stats(
    kb_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Get embedding statistics for a knowledge base.

    Args:
        kb_id: Knowledge base ID
        db: Database session

    Returns:
        Embedding statistics
    """
    # Check if knowledge base exists
    kb = await get_knowledge_base(db, kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    try:
        stats = await get_embeddings_stats(db, kb_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/embeddings/generate/{doc_id}")
async def generate_document_embeddings(
    doc_id: UUID, db: AsyncSession = Depends(get_db)
):
    """
    Manually trigger embedding generation for a document.

    Args:
        doc_id: Document ID
        db: Database session

    Returns:
        Embedding generation results
    """
    # Check if document exists
    doc = await get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        from backend.modules.rag.embedding_service import generate_document_embeddings

        result = await generate_document_embeddings(doc_id, db)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {str(e)}"
        )
