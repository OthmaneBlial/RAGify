from pydantic import BaseModel, UUID4, Field, field_validator, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


class ApplicationBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class ApplicationCreate(ApplicationBase):
    app_model_config: Optional[Dict[str, Any]] = None
    knowledge_base_ids: Optional[List[UUID4]] = None


class ApplicationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    app_model_config: Optional[Dict[str, Any]] = None
    knowledge_base_ids: Optional[List[UUID4]] = None


class ApplicationConfig(BaseModel):
    app_model_config: Optional[Dict[str, Any]] = None
    knowledge_base_ids: Optional[List[str]] = None


class ApplicationResponse(ApplicationBase):
    id: UUID4
    created_at: datetime
    updated_at: datetime
    config: ApplicationConfig
    latest_version: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ApplicationListResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str
    config: ApplicationConfig
    latest_version: Optional[str] = None


class ApplicationDetailResponse(ApplicationResponse):
    knowledge_bases: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None


# Validation
class ApplicationCreateRequest(ApplicationCreate):
    @field_validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Application name cannot be empty")
        return v.strip()


class ApplicationUpdateRequest(ApplicationUpdate):
    @field_validator("name")
    def validate_name(cls, v):
        if v is not None and not v.strip():
            raise ValueError("Application name cannot be empty")
        return v.strip() if v else v
