from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List, Optional, Dict, Any
from uuid import UUID

from .models import KnowledgeBase, Document, Paragraph, Embedding
from .processing import processing_service


# KnowledgeBase CRUD
async def create_knowledge_base(
    db: AsyncSession, name: str, description: Optional[str] = None
) -> KnowledgeBase:
    kb = KnowledgeBase(name=name, description=description)
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return kb


async def get_knowledge_base(db: AsyncSession, kb_id: UUID) -> Optional[KnowledgeBase]:
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    return result.scalar_one_or_none()


async def update_knowledge_base(
    db: AsyncSession,
    kb_id: UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Optional[KnowledgeBase]:
    kb = await get_knowledge_base(db, kb_id)
    if kb:
        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        await db.commit()
        await db.refresh(kb)
    return kb


async def delete_knowledge_base(db: AsyncSession, kb_id: UUID) -> bool:
    result = await db.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb_id))
    await db.commit()
    return result.rowcount > 0


async def list_knowledge_bases(db: AsyncSession) -> List[KnowledgeBase]:
    result = await db.execute(select(KnowledgeBase))
    return result.scalars().all()


# Document CRUD
async def create_document(
    db: AsyncSession, title: str, content: str, knowledge_base_id: UUID
) -> Document:
    doc = Document(
        title=title,
        content=content,
        knowledge_base_id=knowledge_base_id,
        processing_status="processing",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Trigger document processing after creation
    processing_result = await processing_service.process_document_after_creation(
        doc, db
    )

    # Update processing status based on result
    if processing_result["success"]:
        doc.processing_status = "completed"
    else:
        doc.processing_status = "failed"

    await db.commit()
    await db.refresh(doc)

    return doc


async def get_document(db: AsyncSession, doc_id: UUID) -> Optional[Document]:
    result = await db.execute(select(Document).where(Document.id == doc_id))
    return result.scalar_one_or_none()


async def update_document(
    db: AsyncSession,
    doc_id: UUID,
    title: Optional[str] = None,
    content: Optional[str] = None,
) -> Optional[Document]:
    doc = await get_document(db, doc_id)
    if doc:
        if title is not None:
            doc.title = title
        if content is not None:
            doc.content = content
        await db.commit()
        await db.refresh(doc)
    return doc


async def delete_document(db: AsyncSession, doc_id: UUID) -> bool:
    result = await db.execute(delete(Document).where(Document.id == doc_id))
    await db.commit()
    return result.rowcount > 0


async def list_documents_by_knowledge_base(
    db: AsyncSession, knowledge_base_id: UUID
) -> List[Document]:
    result = await db.execute(
        select(Document).where(Document.knowledge_base_id == knowledge_base_id)
    )
    return result.scalars().all()


# Paragraph CRUD
async def create_paragraph(
    db: AsyncSession, content: str, document_id: UUID
) -> Paragraph:
    para = Paragraph(content=content, document_id=document_id)
    db.add(para)
    await db.commit()
    await db.refresh(para)
    return para


async def get_paragraph(db: AsyncSession, para_id: UUID) -> Optional[Paragraph]:
    result = await db.execute(select(Paragraph).where(Paragraph.id == para_id))
    return result.scalar_one_or_none()


async def list_paragraphs_by_document(
    db: AsyncSession, document_id: UUID
) -> List[Paragraph]:
    result = await db.execute(
        select(Paragraph).where(Paragraph.document_id == document_id)
    )
    return result.scalars().all()


# Embedding CRUD
async def create_embedding(
    db: AsyncSession, vector: List[float], paragraph_id: UUID
) -> Embedding:
    emb = Embedding(vector=vector, paragraph_id=paragraph_id)
    db.add(emb)
    await db.commit()
    await db.refresh(emb)
    return emb


async def search_embeddings_by_similarity(
    db: AsyncSession,
    query_vector: List[float],
    limit: int = 10,
    knowledge_base_id: Optional[UUID] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar embeddings using pgvector cosine similarity.

    Args:
        db: Database session
        query_vector: Query embedding vector
        limit: Maximum number of results
        knowledge_base_id: Optional filter by knowledge base
        threshold: Optional similarity threshold (0-1)

    Returns:
        List of similarity results with scores and paragraph data
    """
    from sqlalchemy import text

    # Build the query with pgvector cosine similarity
    query = """
    SELECT
        e.id as embedding_id,
        e.vector as embedding_vector,
        e.paragraph_id,
        p.content as paragraph_content,
        p.document_id,
        d.title as document_title,
        d.knowledge_base_id,
        1 - (e.vector <=> :query_vector) as similarity_score
    FROM embeddings e
    JOIN paragraphs p ON e.paragraph_id = p.id
    JOIN documents d ON p.document_id = d.id
    """

    # Convert vector to PostgreSQL vector string format
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"
    params = {"query_vector": vector_str}

    # Add knowledge base filter if specified
    if knowledge_base_id:
        query += " WHERE d.knowledge_base_id = :kb_id"
        params["kb_id"] = knowledge_base_id

    # Add similarity threshold if specified
    if threshold is not None:
        where_clause = " WHERE " if not knowledge_base_id else " AND "
        query += f"{where_clause} (1 - (e.vector <=> :query_vector)) >= :threshold"
        params["threshold"] = threshold

    # Order by similarity (highest first) and limit results
    query += " ORDER BY e.vector <=> :query_vector LIMIT :limit"
    params["limit"] = limit

    result = await db.execute(text(query), params)

    rows = result.fetchall()
    return [
        {
            "embedding_id": row.embedding_id,
            "paragraph_id": row.paragraph_id,
            "document_id": row.document_id,
            "knowledge_base_id": row.knowledge_base_id,
            "paragraph_content": row.paragraph_content,
            "document_title": row.document_title,
            "similarity_score": float(row.similarity_score),
            "embedding_vector": row.embedding_vector,
        }
        for row in rows
    ]


async def search_paragraphs_by_text(
    db: AsyncSession,
    query_text: str,
    limit: int = 10,
    knowledge_base_id: Optional[UUID] = None,
) -> List[Dict[str, Any]]:
    """
    Search paragraphs by semantic similarity to query text.

    Args:
        db: Database session
        query_text: Query text to search for
        limit: Maximum number of results
        knowledge_base_id: Optional filter by knowledge base

    Returns:
        List of search results with similarity scores
    """
    from ..rag.embedding import encode_text

    # Generate embedding for query text
    query_vector = await encode_text(query_text)

    # Search for similar embeddings
    return await search_embeddings_by_similarity(
        db, query_vector, limit, knowledge_base_id
    )


async def get_embedding_by_paragraph(
    db: AsyncSession, paragraph_id: UUID
) -> Optional[Embedding]:
    """
    Get embedding for a specific paragraph.

    Args:
        db: Database session
        paragraph_id: Paragraph ID

    Returns:
        Embedding object or None
    """
    result = await db.execute(
        select(Embedding).where(Embedding.paragraph_id == paragraph_id)
    )
    return result.scalar_one_or_none()


async def delete_embeddings_by_document(db: AsyncSession, document_id: UUID) -> int:
    """
    Delete all embeddings for a document.

    Args:
        db: Database session
        document_id: Document ID

    Returns:
        Number of embeddings deleted
    """
    # First get paragraph IDs for the document
    from sqlalchemy import select

    result = await db.execute(
        select(Paragraph.id).where(Paragraph.document_id == document_id)
    )
    paragraph_ids = [row[0] for row in result.fetchall()]

    if not paragraph_ids:
        return 0

    # Delete embeddings for these paragraphs
    result = await db.execute(
        select(Embedding).where(Embedding.paragraph_id.in_(paragraph_ids))
    )
    embeddings_to_delete = result.scalars().all()

    for embedding in embeddings_to_delete:
        await db.delete(embedding)

    await db.commit()
    return len(embeddings_to_delete)


async def get_embeddings_stats(
    db: AsyncSession, knowledge_base_id: Optional[UUID] = None
) -> Dict[str, Any]:
    """
    Get statistics about embeddings in the database.

    Args:
        db: Database session
        knowledge_base_id: Optional filter by knowledge base

    Returns:
        Statistics dictionary
    """
    from sqlalchemy import func

    # Count total embeddings
    query = select(func.count(Embedding.id))

    if knowledge_base_id:
        query = (
            query.join(Paragraph)
            .join(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )

    result = await db.execute(query)
    total_embeddings = result.scalar()

    # Count paragraphs with embeddings
    query = select(func.count(func.distinct(Embedding.paragraph_id)))

    if knowledge_base_id:
        query = (
            query.join(Paragraph)
            .join(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )

    result = await db.execute(query)
    paragraphs_with_embeddings = result.scalar()

    return {
        "total_embeddings": total_embeddings,
        "paragraphs_with_embeddings": paragraphs_with_embeddings,
        "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
    }
