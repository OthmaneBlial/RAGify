from sqlalchemy import String, Text, DateTime, ForeignKey, Column
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from uuid import uuid4
from datetime import datetime, timezone

from backend.modules.knowledge.models import Base, UUIDType


class Application(Base):
    __tablename__ = "applications"

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

    versions = relationship("ApplicationVersion", back_populates="application")
    documents = relationship("Document", back_populates="application")


class ApplicationVersion(Base):
    __tablename__ = "application_versions"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    application_id = Column(UUIDType, ForeignKey("applications.id"))
    version = Column(String, nullable=False)
    config = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    application = relationship("Application", back_populates="versions")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    application_id = Column(UUIDType, ForeignKey("applications.id"))
    user_message = Column(Text, nullable=False)
    bot_message = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


class IPUsage(Base):
    __tablename__ = "ip_usage"

    id = Column(UUIDType, primary_key=True, default=lambda: str(uuid4()))
    ip_address = Column(String, nullable=False, index=True)
    request_count = Column(String, nullable=False, default="0")
    last_request_at = Column(DateTime, nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
