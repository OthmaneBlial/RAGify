from pydantic import BaseModel, UUID4, ConfigDict
from typing import Optional
from datetime import datetime


class ChatMessageBase(BaseModel):
    user_message: str
    bot_message: Optional[str] = None
    application_id: UUID4


class ChatMessageCreate(ChatMessageBase):
    pass


class ChatMessageUpdate(ChatMessageBase):
    pass


class ChatMessage(ChatMessageBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
