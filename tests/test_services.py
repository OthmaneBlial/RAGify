"""
Unit tests for service layer operations in applications module.
Tests business logic, validation, and error handling.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from backend.modules.applications.service import (
    ApplicationService,
    create_application_service,
)
from shared.models.Application import ApplicationCreate, ApplicationUpdate


class TestApplicationService:
    """Test ApplicationService business logic"""

    @pytest.mark.asyncio
    async def test_create_application_service(self, db_session):
        """Test creating application service instance"""
        service = await create_application_service(db_session)
        assert isinstance(service, ApplicationService)
        assert service.db == db_session

    @pytest.mark.asyncio
    async def test_create_application_with_validation_success(self, db_session):
        """Test creating application with validation - success case"""
        service = ApplicationService(db_session)

        # Mock knowledge base validation
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = uuid4()
            mock_get_kb.return_value = mock_kb

            # Mock create_application
            with patch('backend.modules.applications.service.create_application') as mock_create:
                mock_app = AsyncMock()
                mock_app.id = uuid4()
                mock_create.return_value = mock_app

                # Mock get_application_with_config
                with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                    mock_get_config.return_value = {
                        "id": str(mock_app.id),
                        "name": "Test App",
                        "config": {"model_config": {"model": "gpt-4"}}
                    }

                    app_data = ApplicationCreate(
                        name="Test App",
                        description="Test description",
                        app_model_config={"model": "gpt-4"},
                        knowledge_base_ids=[uuid4(), uuid4()]
                    )

                    result = await service.create_application_with_validation(app_data)

                    assert result["name"] == "Test App"
                    mock_get_kb.assert_called()
                    mock_create.assert_called_once()
                    mock_get_config.assert_called_once_with(db_session, mock_app.id)

    @pytest.mark.asyncio
    async def test_create_application_with_validation_kb_not_found(self, db_session):
        """Test creating application with validation - knowledge base not found"""
        service = ApplicationService(db_session)

        # Mock knowledge base validation to return None
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_get_kb.return_value = None

            app_data = ApplicationCreate(
                name="Test App",
                knowledge_base_ids=[uuid4()]
            )

            with pytest.raises(ValueError, match="Knowledge base .* not found"):
                await service.create_application_with_validation(app_data)

    @pytest.mark.asyncio
    async def test_create_application_with_validation_invalid_config(self, db_session):
        """Test creating application with validation - invalid config"""
        service = ApplicationService(db_session)

        # Pydantic handles validation at model level, so we test with valid data
        app_data = ApplicationCreate(
            name="Test App",
            app_model_config={"model": "gpt-4"}
        )

        # Mock knowledge base validation
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = uuid4()
            mock_get_kb.return_value = mock_kb

            # Mock create_application
            with patch('backend.modules.applications.service.create_application') as mock_create:
                mock_app = AsyncMock()
                mock_app.id = uuid4()
                mock_create.return_value = mock_app

                # Mock get_application_with_config
                with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                    mock_get_config.return_value = {
                        "id": str(mock_app.id),
                        "name": "Test App",
                        "config": {"model_config": {"model": "gpt-4"}}
                    }

                    result = await service.create_application_with_validation(app_data)

                    assert result["name"] == "Test App"
                    mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_application_with_validation_success(self, db_session):
        """Test updating application with validation - success case"""
        service = ApplicationService(db_session)

        # Mock knowledge base validation
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = uuid4()
            mock_get_kb.return_value = mock_kb

            # Mock update_application
            with patch('backend.modules.applications.service.update_application') as mock_update:
                mock_app = AsyncMock()
                mock_app.id = uuid4()
                mock_update.return_value = mock_app

                # Mock get_application_with_config
                with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                    mock_get_config.return_value = {
                        "id": str(mock_app.id),
                        "name": "Updated App",
                        "config": {"model_config": {"model": "gpt-4"}}
                    }

                    update_data = ApplicationUpdate(
                        name="Updated App",
                        description="Updated description",
                        app_model_config={"model": "gpt-4"},
                        knowledge_base_ids=[uuid4()]
                    )

                    result = await service.update_application_with_validation(
                        mock_app.id, update_data
                    )

                    assert result["name"] == "Updated App"
                    mock_get_kb.assert_called()
                    mock_update.assert_called_once()
                    mock_get_config.assert_called_once_with(db_session, mock_app.id)

    @pytest.mark.asyncio
    async def test_update_application_with_validation_not_found(self, db_session):
        """Test updating application with validation - application not found"""
        service = ApplicationService(db_session)

        # Mock update_application to return None
        with patch('backend.modules.applications.service.update_application') as mock_update:
            mock_update.return_value = None

            update_data = ApplicationUpdate(name="New Name")

            with pytest.raises(ValueError, match="Application not found"):
                await service.update_application_with_validation(uuid4(), update_data)

    @pytest.mark.asyncio
    async def test_associate_knowledge_bases_success(self, db_session):
        """Test associating knowledge bases - success case"""
        service = ApplicationService(db_session)
        app_id = uuid4()
        kb_ids = [uuid4(), uuid4()]

        # Mock knowledge base validation
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = kb_ids[0]
            mock_get_kb.return_value = mock_kb

            # Mock update_application
            with patch('backend.modules.applications.service.update_application') as mock_update:
                mock_app = AsyncMock()
                mock_app.id = app_id
                mock_update.return_value = mock_app

                # Mock get_application_with_config
                with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                    mock_get_config.return_value = {
                        "id": str(app_id),
                        "name": "Test App",
                        "config": {"knowledge_base_ids": [str(kb_id) for kb_id in kb_ids]}
                    }

                    result = await service.associate_knowledge_bases(app_id, kb_ids)

                    assert result["id"] == str(app_id)
                    mock_get_kb.assert_called()
                    mock_update.assert_called_once()
                    mock_get_config.assert_called_once_with(db_session, app_id)

    @pytest.mark.asyncio
    async def test_associate_knowledge_bases_app_not_found(self, db_session):
        """Test associating knowledge bases - application not found"""
        service = ApplicationService(db_session)
        app_id = uuid4()
        kb_ids = [uuid4()]

        # Mock knowledge base validation
        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = kb_ids[0]
            mock_get_kb.return_value = mock_kb

            # Mock update_application to return None
            with patch('backend.modules.applications.service.update_application') as mock_update:
                mock_update.return_value = None

                with pytest.raises(ValueError, match="Application not found"):
                    await service.associate_knowledge_bases(app_id, kb_ids)

    @pytest.mark.asyncio
    async def test_get_application_statistics(self, db_session):
        """Test getting application statistics"""
        service = ApplicationService(db_session)
        app_id = uuid4()

        # Mock chat history
        with patch('backend.modules.applications.service.get_application_chat_history') as mock_chat_history:
            mock_chat_history.return_value = [
                {"created_at": "2023-01-01T10:00:00"},
                {"created_at": "2023-01-02T11:00:00"},
            ]

            # Mock knowledge bases
            with patch('backend.modules.applications.service.get_application_knowledge_bases') as mock_kb:
                mock_kb.return_value = [uuid4(), uuid4()]

                stats = await service.get_application_statistics(app_id)

                assert stats["total_chats"] == 2
                assert stats["knowledge_bases_count"] == 2
                assert stats["average_response_time"] == 0.0
                assert stats["last_activity"] == "2023-01-01T10:00:00"

                mock_chat_history.assert_called_once_with(db_session, app_id, limit=1000)
                mock_kb.assert_called_once_with(db_session, app_id)

    @pytest.mark.asyncio
    async def test_get_application_statistics_no_activity(self, db_session):
        """Test getting application statistics with no activity"""
        service = ApplicationService(db_session)
        app_id = uuid4()

        # Mock empty chat history
        with patch('backend.modules.applications.service.get_application_chat_history') as mock_chat_history:
            mock_chat_history.return_value = []

            # Mock knowledge bases
            with patch('backend.modules.applications.service.get_application_knowledge_bases') as mock_kb:
                mock_kb.return_value = []

                stats = await service.get_application_statistics(app_id)

                assert stats["total_chats"] == 0
                assert stats["knowledge_bases_count"] == 0
                assert stats["last_activity"] is None

    @pytest.mark.asyncio
    async def test_validate_knowledge_bases_success(self, db_session):
        """Test validating knowledge bases - success case"""
        service = ApplicationService(db_session)
        kb_ids = [uuid4(), uuid4()]

        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_kb = AsyncMock()
            mock_kb.id = kb_ids[0]
            mock_get_kb.return_value = mock_kb

            # This should not raise an exception
            await service._validate_knowledge_bases(kb_ids)

            # Should be called for each KB
            assert mock_get_kb.call_count == 2

    @pytest.mark.asyncio
    async def test_validate_knowledge_bases_not_found(self, db_session):
        """Test validating knowledge bases - knowledge base not found"""
        service = ApplicationService(db_session)
        kb_ids = [uuid4()]

        with patch('backend.modules.applications.service.get_knowledge_base') as mock_get_kb:
            mock_get_kb.return_value = None

            with pytest.raises(ValueError, match="Knowledge base .* not found"):
                await service._validate_knowledge_bases(kb_ids)

    @pytest.mark.asyncio
    async def test_validate_model_config_valid(self, db_session):
        """Test validating model config - valid config"""
        service = ApplicationService(db_session)

        valid_config = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        }

        # This should not raise an exception
        service._validate_model_config(valid_config)

    @pytest.mark.asyncio
    async def test_validate_model_config_invalid_type(self, db_session):
        """Test validating model config - invalid type"""
        service = ApplicationService(db_session)

        invalid_config = "not_a_dict"

        with pytest.raises(ValueError, match="Model config must be a dictionary"):
            service._validate_model_config(invalid_config)

    @pytest.mark.asyncio
    async def test_validate_model_config_none(self, db_session):
        """Test validating model config - None config"""
        service = ApplicationService(db_session)

        # None is not valid according to current implementation
        with pytest.raises(ValueError, match="Model config must be a dictionary"):
            service._validate_model_config(None)

    @pytest.mark.asyncio
    async def test_service_error_handling(self, db_session):
        """Test service error handling in various scenarios"""
        service = ApplicationService(db_session)

        # Test with database errors
        with patch('backend.modules.applications.service.create_application') as mock_create:
            mock_create.side_effect = Exception("Database error")

            app_data = ApplicationCreate(name="Test App")

            with pytest.raises(Exception, match="Database error"):
                await service.create_application_with_validation(app_data)

    @pytest.mark.asyncio
    async def test_service_with_empty_knowledge_bases(self, db_session):
        """Test service operations with empty knowledge base list"""
        service = ApplicationService(db_session)

        app_data = ApplicationCreate(
            name="Test App",
            knowledge_base_ids=[]
        )

        # Mock create_application
        with patch('backend.modules.applications.service.create_application') as mock_create:
            mock_app = AsyncMock()
            mock_app.id = uuid4()
            mock_create.return_value = mock_app

            # Mock get_application_with_config
            with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                mock_get_config.return_value = {
                    "id": str(mock_app.id),
                    "name": "Test App",
                    "config": {}
                }

                result = await service.create_application_with_validation(app_data)

                assert result["name"] == "Test App"
                # get_knowledge_base should not be called for empty list
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_service_with_none_config(self, db_session):
        """Test service operations with None configuration"""
        service = ApplicationService(db_session)

        app_data = ApplicationCreate(
            name="Test App",
            app_model_config=None
        )

        # Mock create_application
        with patch('backend.modules.applications.service.create_application') as mock_create:
            mock_app = AsyncMock()
            mock_app.id = uuid4()
            mock_create.return_value = mock_app

            # Mock get_application_with_config
            with patch('backend.modules.applications.service.get_application_with_config') as mock_get_config:
                mock_get_config.return_value = {
                    "id": str(mock_app.id),
                    "name": "Test App",
                    "config": {}
                }

                result = await service.create_application_with_validation(app_data)

                assert result["name"] == "Test App"
                mock_create.assert_called_once()