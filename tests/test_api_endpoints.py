"""
API endpoint tests for RAGify.
Tests FastAPI endpoint functionality, request/response validation, error handling, and authentication.
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestKnowledgeEndpoints:
    """Test cases for knowledge-related API endpoints."""

    def test_create_knowledge_base_endpoint(self, test_client):
        """Test creating knowledge base via API."""
        kb_data = {"name": "Test Knowledge Base", "description": "Test description"}

        response = test_client.post("/api/v1/knowledge/knowledge-bases/", json=kb_data)
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert data["name"] == kb_data["name"]
        assert data["description"] == kb_data["description"]


class TestApplicationEndpoints:
    """Test cases for application-related API endpoints."""


class TestChatEndpoints:
    """Test cases for chat-related API endpoints."""


class TestRequestValidation:
    """Test request validation for API endpoints."""

    def test_invalid_application_creation(self, test_client):
        """Test validation for invalid application creation."""
        invalid_data = {
            "name": "",  # Empty name should fail
            "config": "invalid",  # Should be dict
        }

        response = test_client.post("/api/v1/applications/", json=invalid_data)
        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        incomplete_data = {"description": "Missing name field"}

        response = test_client.post(
            "/api/v1/knowledge/knowledge-bases/", json=incomplete_data
        )
        assert response.status_code == 422  # Validation error

    def test_invalid_uuid_format(self, test_client):
        """Test handling of invalid UUID format."""
        response = test_client.get("/api/v1/knowledge/knowledge-bases/invalid-uuid")
        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test error handling in API endpoints."""

    def test_method_not_allowed(self, test_client):
        """Test handling of incorrect HTTP methods."""
        response = test_client.put("/api/v1/knowledge/knowledge-bases/", json={})
        assert response.status_code == 405

    def test_unsupported_content_type(self, test_client):
        """Test handling of unsupported content types."""
        response = test_client.post(
            "/api/v1/knowledge/knowledge-bases/",
            content="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code == 422

    def test_invalid_json(self, test_client):
        """Test handling of invalid JSON."""
        response = test_client.post(
            "/api/v1/knowledge/knowledge-bases/",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


class TestResponseFormat:
    """Test response format and structure."""

    def test_health_endpoint(self, test_client):
        """Test health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "database" in data
        assert "cache" in data


class TestIntegration:
    """Integration tests for API endpoints."""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
