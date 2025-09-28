"""
Script to load example data into the RAGify system.

This script reads example application configurations from examples/data/applications/
and creates the corresponding knowledge bases, applications, and documents in the database.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.database import get_db_session
from backend.modules.knowledge.crud import create_knowledge_base, create_document
from backend.modules.applications.crud import create_application


async def load_examples():
    """
    Load example data into the system.
    """
    print("Loading RAGify examples...")

    examples_dir = Path(__file__).parent.parent / "data"
    applications_dir = examples_dir / "applications"
    documents_dir = examples_dir / "documents"

    if not applications_dir.exists():
        print(f"Error: Applications directory not found: {applications_dir}")
        return

    if not documents_dir.exists():
        print(f"Error: Documents directory not found: {documents_dir}")
        return

    async with get_db_session() as db:
        # Get list of application config files
        config_files = list(applications_dir.glob("*.json"))
        if not config_files:
            print(f"No application config files found in {applications_dir}")
            return

        print(f"Found {len(config_files)} application configurations")

        for config_file in config_files:
            try:
                print(f"Processing {config_file.name}...")

                # Load configuration
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # Extract configuration data
                kb_name = config.get('knowledge_base')
                app_details = config.get('application_details', {})
                app_name = app_details.get('name')
                app_description = app_details.get('description')
                documents = config.get('documents', [])

                if not kb_name:
                    print(f"Warning: No knowledge_base specified in {config_file.name}")
                    continue

                if not app_name:
                    print(f"Warning: No application name specified in {config_file.name}")
                    continue

                # Create knowledge base
                print(f"  Creating knowledge base: {kb_name}")
                kb = await create_knowledge_base(db, kb_name, f"Knowledge base for {app_name}")

                # Create application
                print(f"  Creating application: {app_name}")
                app = await create_application(db, app_name, app_description, knowledge_base_ids=[kb.id])

                # Load documents
                for doc_config in documents:
                    doc_title = doc_config.get('title')
                    doc_filename = doc_config.get('filename')

                    if not doc_title or not doc_filename:
                        print(f"Warning: Invalid document config in {config_file.name}: {doc_config}")
                        continue

                    doc_path = documents_dir / doc_filename
                    if not doc_path.exists():
                        print(f"Warning: Document file not found: {doc_path}")
                        continue

                    # Read document content
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except Exception as e:
                        print(f"Error reading document {doc_path}: {e}")
                        continue

                    # Create document
                    print(f"  Creating document: {doc_title}")
                    doc = await create_document(db, doc_title, content, kb.id, app.id)

                    if doc.processing_status == "failed":
                        print(f"Warning: Document processing failed for {doc_title}")
                    else:
                        print(f"  Document created successfully: {doc_title}")

                print(f"Successfully loaded {config_file.name}")

            except Exception as e:
                print(f"Error processing {config_file.name}: {e}")
                continue

    print("Example loading completed!")


if __name__ == "__main__":
    asyncio.run(load_examples())