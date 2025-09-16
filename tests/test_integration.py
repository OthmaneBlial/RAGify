"""
Integration tests for RAGify.
Tests complete end-to-end workflows from document upload to chat responses.
"""

import pytest
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""


class TestSystemIntegration:
    """Test system-level integrations."""

    def test_health_endpoint_integration(self, test_client):
        """Test health endpoint provides system information."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        required_fields = ["status", "timestamp", "database", "cache", "task_queue"]
        for field in required_fields:
            assert field in data

        assert data["status"] == "ok"
        assert isinstance(data["timestamp"], (int, float))

    def test_root_endpoint_integration(self, test_client):
        """Test root endpoint."""
        response = test_client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "RAGify" in data["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])