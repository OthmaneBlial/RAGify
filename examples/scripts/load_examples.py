"""
Script to load example data into the RAGify system.

This script reads example application configurations from examples/data/applications/
and creates the corresponding knowledge bases, applications, and documents in the database.
You can now select a subset of examples to load when running the script.
"""

import argparse
import sys
import os
import json
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Set, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.database import get_db_session
from backend.modules.knowledge.crud import create_knowledge_base, create_document
from backend.modules.applications.crud import create_application


@dataclass
class ExampleDefinition:
    path: Path
    config: Dict[str, Any]
    display_name: str
    description: Optional[str]
    aliases: Set[str] = field(default_factory=set)


def normalize_token(value: str) -> str:
    """Return a lowercase alphanumeric token for matching user selections."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def load_example_definitions(applications_dir: Path) -> List[ExampleDefinition]:
    """Load example configurations and metadata for selection."""
    definitions: List[ExampleDefinition] = []

    for path in sorted(applications_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as file_handle:
                config = json.load(file_handle)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Warning: Skipping {path.name}: {exc}")
            continue

        app_details = config.get("application_details", {})
        display_name = app_details.get("name") or path.stem.replace("_", " ").title()
        description = app_details.get("description")

        aliases: Set[str] = {
            normalize_token(path.stem),
            normalize_token(path.name),
        }
        if display_name:
            display_alias = normalize_token(display_name)
            if display_alias:
                aliases.add(display_alias)
            for part in display_name.split():
                part_alias = normalize_token(part)
                if part_alias:
                    aliases.add(part_alias)

        definitions.append(
            ExampleDefinition(
                path=path,
                config=config,
                display_name=display_name,
                description=description,
                aliases=aliases,
            )
        )

    return definitions


def resolve_selection(
    definitions: List[ExampleDefinition], tokens: Sequence[str]
) -> Tuple[List[ExampleDefinition], List[str]]:
    """Resolve user-provided selection tokens to concrete example definitions."""
    alias_lookup: Dict[str, List[ExampleDefinition]] = {}
    for definition in definitions:
        for alias in definition.aliases:
            alias_lookup.setdefault(alias, []).append(definition)

    selected: List[ExampleDefinition] = []
    unmatched: List[str] = []

    for raw_token in tokens:
        token = raw_token.strip()
        if not token:
            continue

        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(definitions):
                definition = definitions[index - 1]
                if definition not in selected:
                    selected.append(definition)
                continue

        normalized = normalize_token(token)
        matches = alias_lookup.get(normalized)
        if matches:
            for definition in matches:
                if definition not in selected:
                    selected.append(definition)
                    break
            continue

        unmatched.append(raw_token)

    return selected, unmatched


def prompt_for_selection(definitions: List[ExampleDefinition]) -> List[ExampleDefinition]:
    """Interactively ask the user which examples to load."""
    print("Available examples:")
    for idx, definition in enumerate(definitions, start=1):
        label = definition.display_name or definition.path.stem
        print(f"  [{idx}] {label} ({definition.path.name})")
        if definition.description:
            print(f"      {definition.description}")

    prompt_text = (
        "Select examples to load (comma-separated numbers or names, leave blank for all): "
    )

    try:
        selection = input(prompt_text).strip()
    except EOFError:
        return definitions

    if not selection or selection.lower() in {"all", "a", "*"}:
        return definitions

    tokens = [token.strip() for token in selection.split(",") if token.strip()]
    selected, unmatched = resolve_selection(definitions, tokens)

    if unmatched:
        print(
            "Warning: Ignored unrecognized selections: "
            + ", ".join(unmatched)
        )

    if not selected:
        return []

    return selected


async def load_examples(
    selected_examples: Optional[Sequence[str]] = None, interactive: bool = False
) -> None:
    """
    Load example data into the system.

    Args:
        selected_examples: Optional selection tokens supplied via CLI.
        interactive: When True, prompt for selections if none were provided.
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

    definitions = load_example_definitions(applications_dir)
    if not definitions:
        print(f"No application config files found in {applications_dir}")
        return

    print(f"Found {len(definitions)} application configurations")

    selected_definitions = definitions

    if selected_examples:
        tokens: List[str] = []
        for item in selected_examples:
            if "," in item:
                tokens.extend(part.strip() for part in item.split(",") if part.strip())
            else:
                tokens.append(item)

        selected_definitions, unmatched = resolve_selection(definitions, tokens)
        if unmatched:
            print(
                "Warning: Ignored unrecognized selections: "
                + ", ".join(unmatched)
            )
        if not selected_definitions:
            print("No matching examples found for the provided selections.")
            return
    elif interactive and len(definitions) > 1:
        selected_definitions = prompt_for_selection(definitions)
        if not selected_definitions:
            print("No examples selected. Nothing to load.")
            return

    print(f"Preparing to load {len(selected_definitions)} example(s).")

    async with get_db_session() as db:
        for definition in selected_definitions:
            config_file = definition.path
            try:
                print(f"Processing {config_file.name}...")

                # Load configuration
                config = definition.config

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
    parser = argparse.ArgumentParser(description="Load example data into RAGify.")
    parser.add_argument(
        "-e",
        "--examples",
        nargs="+",
        help=(
            "Specific examples to load. Accepts indexes or names such as "
            "'customer_support' or 'Customer Support Assistant'."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load all examples without prompting for a selection.",
    )

    args = parser.parse_args()
    if args.all and args.examples:
        print("Warning: --all is ignored because specific examples were provided.")

    interactive_prompt = (
        not args.all and not args.examples and sys.stdin.isatty()
    )

    asyncio.run(
        load_examples(
            selected_examples=args.examples,
            interactive=interactive_prompt,
        )
    )
