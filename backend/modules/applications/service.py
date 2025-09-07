from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any
from uuid import UUID

from .crud import (
    create_application,
    get_application_with_config,
    update_application,
    get_application_knowledge_bases,
    get_application_chat_history,
)
from ..knowledge.crud import get_knowledge_base  # Assuming knowledge base CRUD exists
from shared.models.Application import ApplicationCreate, ApplicationUpdate


class ApplicationService:
    """Service layer for application business logic"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_application_with_validation(
        self, app_data: ApplicationCreate
    ) -> Dict[str, Any]:
        """
        Create application with validation and knowledge base verification

        Args:
            app_data: Application creation data

        Returns:
            Created application data

        Raises:
            ValueError: If validation fails
        """
        # Validate knowledge base IDs if provided
        if app_data.knowledge_base_ids:
            await self._validate_knowledge_bases(app_data.knowledge_base_ids)

        # Validate configuration
        if app_data.app_model_config:
            self._validate_model_config(app_data.app_model_config)

        # Create application
        application = await create_application(
            db=self.db,
            name=app_data.name,
            description=app_data.description,
            config=app_data.app_model_config,
            knowledge_base_ids=app_data.knowledge_base_ids,
        )

        return await get_application_with_config(self.db, application.id)

    async def update_application_with_validation(
        self, application_id: UUID, update_data: ApplicationUpdate
    ) -> Dict[str, Any]:
        """
        Update application with validation

        Args:
            application_id: Application ID
            update_data: Update data

        Returns:
            Updated application data

        Raises:
            ValueError: If validation fails
        """
        # Validate knowledge base IDs if provided
        if update_data.knowledge_base_ids:
            await self._validate_knowledge_bases(update_data.knowledge_base_ids)

        # Validate configuration
        if update_data.app_model_config:
            self._validate_model_config(update_data.app_model_config)

        # Update application
        application = await update_application(
            db=self.db,
            application_id=application_id,
            name=update_data.name,
            description=update_data.description,
            config=update_data.app_model_config,
            knowledge_base_ids=update_data.knowledge_base_ids,
        )

        if not application:
            raise ValueError("Application not found")

        return await get_application_with_config(self.db, application_id)

    async def associate_knowledge_bases(
        self, application_id: UUID, knowledge_base_ids: List[UUID]
    ) -> Dict[str, Any]:
        """
        Associate knowledge bases with application

        Args:
            application_id: Application ID
            knowledge_base_ids: Knowledge base IDs to associate

        Returns:
            Updated application data
        """
        # Validate knowledge bases exist
        await self._validate_knowledge_bases(knowledge_base_ids)

        # Update associations
        application = await update_application(
            db=self.db,
            application_id=application_id,
            knowledge_base_ids=knowledge_base_ids,
        )

        if not application:
            raise ValueError("Application not found")

        return await get_application_with_config(self.db, application_id)

    async def get_application_statistics(self, application_id: UUID) -> Dict[str, Any]:
        """
        Get application statistics

        Args:
            application_id: Application ID

        Returns:
            Application statistics
        """
        # Get chat history count
        chat_history = await get_application_chat_history(
            self.db, application_id, limit=1000
        )
        total_chats = len(chat_history)

        # Get knowledge base count
        kb_ids = await get_application_knowledge_bases(self.db, application_id)
        kb_count = len(kb_ids)

        # Calculate average response time (placeholder)
        avg_response_time = 0.0  # Would need to track this in chat messages

        return {
            "total_chats": total_chats,
            "knowledge_bases_count": kb_count,
            "average_response_time": avg_response_time,
            "last_activity": chat_history[0]["created_at"] if chat_history else None,
        }

    async def _validate_knowledge_bases(self, knowledge_base_ids: List[UUID]):
        """
        Validate that knowledge bases exist

        Args:
            knowledge_base_ids: Knowledge base IDs to validate

        Raises:
            ValueError: If any knowledge base doesn't exist
        """
        for kb_id in knowledge_base_ids:
            kb = await get_knowledge_base(self.db, kb_id)
            if not kb:
                raise ValueError(f"Knowledge base {kb_id} not found")

    def _validate_model_config(self, config: Dict[str, Any]):
        """
        Validate model configuration

        Args:
            config: Model configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Basic validation - can be extended
        if not isinstance(config, dict):
            raise ValueError("Model config must be a dictionary")

        # Validate required fields if any
        # This can be extended based on specific model requirements


# Convenience functions
async def create_application_service(db: AsyncSession) -> ApplicationService:
    """Create application service instance"""
    return ApplicationService(db)
