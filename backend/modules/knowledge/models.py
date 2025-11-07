from sqlalchemy import String, Text, DateTime, ForeignKey, Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime, timezone
from ...core.database import Vector, is_sqlite
from ...core.config import settings

# Handle UUID for SQLite compatibility
if is_sqlite:
    # For SQLite, use String type for UUIDs since SQLite doesn't have native UUID
    UUIDType = String
else:
    UUIDType = UUID(as_uuid=True)

Base = declarative_base()


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    documents = relationship("Document", back_populates="knowledge_base")


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    knowledge_base_id = Column(UUIDType, ForeignKey("knowledge_bases.id"))
    application_id = Column(UUIDType, ForeignKey("applications.id"))
    processing_status = Column(String, nullable=True, default="pending")
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    application = relationship("Application", back_populates="documents")
    paragraphs = relationship("Paragraph", back_populates="document")


class Paragraph(Base):
    __tablename__ = "paragraphs"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    content = Column(Text, nullable=False)
    document_id = Column(UUIDType, ForeignKey("documents.id"))
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    document = relationship("Document", back_populates="paragraphs")
    embedding = relationship("Embedding", back_populates="paragraph", uselist=False)


class Embedding(Base):
    __tablename__ = "embeddings"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    vector = Column(Vector(settings.vector_dimension) if Vector else Text)
    paragraph_id = Column(UUIDType, ForeignKey("paragraphs.id"))
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    paragraph = relationship("Paragraph", back_populates="embedding")
