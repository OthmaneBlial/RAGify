#!/usr/bin/env python3
"""
RAGify Database Reset Script
Deletes all applications except default and recreates fresh ones
"""

import requests
import sys

# Configuration
API_BASE = "http://localhost:8000"


def check_server_running():
    """Check if the server is running."""
    try:
        response = requests.get(f"{API_BASE}/api/v1/applications/", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def reset_database():
    """Delete all applications and create fresh default."""
    print("🗑️  RAGify Database Reset")
    print("=" * 30)

    # Check if server is running
    if not check_server_running():
        print("❌ Server is not running!")
        print("Please start the server first:")
        print("  python -m backend.main")
        print("Then run this script again.")
        sys.exit(1)

    print("✅ Server is running")

    try:
        # Get all applications
        response = requests.get(f"{API_BASE}/api/v1/applications/")
        if response.status_code != 200:
            print(f"❌ Failed to get applications: {response.status_code}")
            return

        applications = response.json()
        print(f"Found {len(applications)} applications")

        # Delete all applications except default
        deleted_count = 0
        for app in applications:
            if app["name"] != "Default Chat Application":
                delete_response = requests.delete(
                    f"{API_BASE}/api/v1/applications/{app['id']}"
                )
                if delete_response.status_code == 200:
                    print(f"🗑️  Deleted: {app['name']}")
                    deleted_count += 1
                else:
                    error_detail = delete_response.json().get("detail", "Unknown error")
                    print(f"❌ Failed to delete: {app['name']} - {error_detail}")
            else:
                print(f"⏭️  Skipping default application: {app['name']}")

        print(f"✅ Successfully deleted {deleted_count} applications")

        # Create fresh default application if it doesn't exist
        default_exists = any(
            app["name"] == "Default Chat Application" for app in applications
        )
        if not default_exists:
            print("Creating fresh default application...")
            create_response = requests.post(
                f"{API_BASE}/api/v1/applications/",
                json={
                    "name": "Default Chat Application",
                    "description": "Fresh default application for chat functionality",
                    "config": {"provider": "openrouter", "model": "openai/gpt-5-nano"},
                    "knowledge_base_ids": [],
                },
            )
            if create_response.status_code == 200:
                print("✅ Fresh default application created")
            else:
                print(
                    f"❌ Failed to create default application: {create_response.status_code}"
                )

        print("🎉 Database reset complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    reset_database()
