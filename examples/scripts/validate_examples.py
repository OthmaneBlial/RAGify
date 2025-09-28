"""
Script to validate example data in the RAGify system.

This script checks that all example configuration files are valid JSON,
that referenced document files exist, and that the overall structure is correct.
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any


def validate_examples():
    """
    Validate example data files and configurations.
    """
    print("Validating RAGify examples...")

    examples_dir = Path(__file__).parent.parent / "data"
    applications_dir = examples_dir / "applications"
    documents_dir = examples_dir / "documents"

    errors = []
    warnings = []

    # Check directory structure
    if not applications_dir.exists():
        errors.append(f"Applications directory not found: {applications_dir}")
        return False

    if not documents_dir.exists():
        errors.append(f"Documents directory not found: {documents_dir}")
        return False

    # Get list of application config files
    config_files = list(applications_dir.glob("*.json"))
    if not config_files:
        warnings.append(f"No application config files found in {applications_dir}")
    else:
        print(f"Found {len(config_files)} application configuration files")

    # Validate each configuration file
    for config_file in config_files:
        print(f"Validating {config_file.name}...")

        try:
            # Load and parse JSON
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Validate required fields
            if 'knowledge_base' not in config:
                errors.append(f"{config_file.name}: Missing 'knowledge_base' field")
                continue

            kb_name = config['knowledge_base']
            if not isinstance(kb_name, str) or not kb_name.strip():
                errors.append(f"{config_file.name}: 'knowledge_base' must be a non-empty string")

            if 'application_details' not in config:
                errors.append(f"{config_file.name}: Missing 'application_details' field")
                continue

            app_details = config['application_details']
            if not isinstance(app_details, dict):
                errors.append(f"{config_file.name}: 'application_details' must be an object")
                continue

            if 'name' not in app_details:
                errors.append(f"{config_file.name}: Missing 'application_details.name' field")
                continue

            app_name = app_details['name']
            if not isinstance(app_name, str) or not app_name.strip():
                errors.append(f"{config_file.name}: 'application_details.name' must be a non-empty string")

            if 'description' not in app_details:
                warnings.append(f"{config_file.name}: Missing 'application_details.description' field")

            if 'sample_questions' not in app_details:
                warnings.append(f"{config_file.name}: Missing 'application_details.sample_questions' field")

            # Validate documents array
            if 'documents' not in config:
                errors.append(f"{config_file.name}: Missing 'documents' field")
                continue

            documents = config['documents']
            if not isinstance(documents, list):
                errors.append(f"{config_file.name}: 'documents' must be an array")
                continue

            if not documents:
                warnings.append(f"{config_file.name}: No documents specified")

            # Validate each document
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    errors.append(f"{config_file.name}: Document {i} must be an object")
                    continue

                if 'title' not in doc:
                    errors.append(f"{config_file.name}: Document {i} missing 'title' field")
                    continue

                if 'filename' not in doc:
                    errors.append(f"{config_file.name}: Document {i} missing 'filename' field")
                    continue

                doc_title = doc['title']
                doc_filename = doc['filename']

                if not isinstance(doc_title, str) or not doc_title.strip():
                    errors.append(f"{config_file.name}: Document {i} 'title' must be a non-empty string")

                if not isinstance(doc_filename, str) or not doc_filename.strip():
                    errors.append(f"{config_file.name}: Document {i} 'filename' must be a non-empty string")
                    continue

                # Check if document file exists
                doc_path = documents_dir / doc_filename
                if not doc_path.exists():
                    errors.append(f"{config_file.name}: Referenced document file not found: {doc_path}")
                    continue

                # Try to read the document file
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            warnings.append(f"{config_file.name}: Document file is empty: {doc_path}")
                except Exception as e:
                    errors.append(f"{config_file.name}: Error reading document file {doc_path}: {e}")

            print(f"  ✓ {config_file.name} validation completed")

        except json.JSONDecodeError as e:
            errors.append(f"{config_file.name}: Invalid JSON - {e}")
        except Exception as e:
            errors.append(f"{config_file.name}: Unexpected error - {e}")

    # Report results
    if errors:
        print(f"\n❌ Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    if warnings:
        print(f"\n⚠️  Validation completed with {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors:
        print(f"\n✅ All validations passed! Found {len(config_files)} valid configurations.")

    return len(errors) == 0


if __name__ == "__main__":
    success = validate_examples()
    sys.exit(0 if success else 1)