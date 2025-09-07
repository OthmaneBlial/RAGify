from pydantic import BaseModel, UUID4, ConfigDict
from typing import Optional
from datetime import datetime


class DocumentBase(BaseModel):
    title: str
    content: str
    knowledge_base_id: UUID4
    processing_status: Optional[str] = "pending"


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(DocumentBase):
    pass


class Document(DocumentBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
