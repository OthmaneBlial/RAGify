"""
Script to reset example data in the RAGify system.

This script removes all example applications and their associated knowledge bases
that were created by the load_examples.py script.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.database import get_db_session
from backend.modules.applications.crud import list_applications, delete_application
from backend.modules.knowledge.crud import (
    list_knowledge_bases, delete_knowledge_base, delete_document
)


# List of example application names that should be removed
EXAMPLE_APP_NAMES = {
    "API Documentation Assistant",
    "Codebase Q&A Assistant",
    "Customer Support Assistant",
    "HR Assistant",
    "Legal Analysis Assistant",
    "Sales Enablement Assistant",
    "Research Paper Assistant",
    "Interactive Study Guide"
}

# List of example knowledge base names that should be removed
EXAMPLE_KB_NAMES = {
    "api_documentation_knowledge_base",
    "codebase_qa_knowledge_base",
    "customer_support_knowledge_base",
    "hr_assistant_knowledge_base",
    "legal_analysis_knowledge_base",
    "sales_enablement_knowledge_base",
    "research_paper_assistant_knowledge_base",
    "study_guide_knowledge_base"
}


async def reset_examples():
    """
    Reset example data by removing example applications and knowledge bases.
    """
    print("Resetting RAGify examples...")

    async with get_db_session() as db:
        # First, remove all documents from example knowledge bases
        print("Removing example documents...")
        from backend.modules.knowledge.crud import (
            list_documents_by_knowledge_base,
            list_paragraphs_by_document,
            delete_embeddings_by_document
        )

        kbs = await list_knowledge_bases(db)
        removed_docs = 0

        for kb in kbs:
            if kb.name in EXAMPLE_KB_NAMES:
                print(f"  Deleting documents for knowledge base: {kb.name}")
                docs = await list_documents_by_knowledge_base(db, kb.id)
                for doc in docs:
                    # Delete embeddings and paragraphs first
                    await delete_embeddings_by_document(db, doc.id)
                    paragraphs = await list_paragraphs_by_document(db, doc.id)
                    for para in paragraphs:
                        await db.delete(para)
                    await db.commit()

                    # Now delete the document
                    success = await delete_document(db, doc.id)
                    if success:
                        removed_docs += 1

        print(f"Removed {removed_docs} example documents")

        # Remove example applications
        print("Removing example applications...")
        apps = await list_applications(db)  # Re-fetch after deleting documents
        removed_apps = 0

        for app in apps:
            if app['name'] in EXAMPLE_APP_NAMES:
                print(f"  Deleting application: {app['name']}")
                success = await delete_application(db, app['id'])
                if success:
                    removed_apps += 1
                else:
                    print(f"  Failed to delete application: {app['name']}")

        print(f"Removed {removed_apps} example applications")

        # Remove example knowledge bases
        print("Removing example knowledge bases...")
        kbs = await list_knowledge_bases(db)
        removed_kbs = 0

        for kb in kbs:
            if kb.name in EXAMPLE_KB_NAMES:
                print(f"  Deleting knowledge base: {kb.name}")
                success = await delete_knowledge_base(db, kb.id)
                if success:
                    removed_kbs += 1
                else:
                    print(f"  Failed to delete knowledge base: {kb.name}")

        print(f"Removed {removed_kbs} example knowledge bases")

    print("Example reset completed!")


if __name__ == "__main__":
    asyncio.run(reset_examples())
