import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from sqlalchemy.orm import selectinload
from typing import List, Optional, Dict, Any
from uuid import UUID

from backend.core.database import normalize_uuid
from .models import Application, ApplicationVersion, ChatMessage


# Application CRUD Operations
async def create_application(
    db: AsyncSession,
    name: str,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    knowledge_base_ids: Optional[List[UUID]] = None,
) -> Application:
    """
    Create a new application.

    Args:
        db: Database session
        name: Application name
        description: Optional application description
        model_config: Optional model configuration
        knowledge_base_ids: Optional list of associated knowledge base IDs

    Returns:
        Created application
    """
    # Create application
    application = Application(name=name, description=description)
    db.add(application)
    await db.commit()
    await db.refresh(application)

    # Create initial version with configuration
    config_data = {
        "model_config": config or {},
        "knowledge_base_ids": [str(kb_id) for kb_id in (knowledge_base_ids or [])],
    }

    version = ApplicationVersion(
        application_id=application.id, version="1.0", config=json.dumps(config_data)
    )
    db.add(version)
    await db.commit()
    await db.refresh(application)

    return application


async def get_application(
    db: AsyncSession, application_id: UUID
) -> Optional[Application]:
    """
    Get an application by ID with latest version.

    Args:
        db: Database session
        application_id: Application ID

    Returns:
        Application with latest version or None
    """
    db_application_id = normalize_uuid(application_id)
    result = await db.execute(
        select(Application)
        .where(Application.id == db_application_id)
        .options(selectinload(Application.versions))
    )
    application = result.scalar_one_or_none()

    if application and application.versions:
        # Sort versions by creation date and get latest
        application.versions.sort(key=lambda v: v.created_at, reverse=True)

    return application


async def get_application_with_config(
    db: AsyncSession, application_id: UUID
) -> Optional[Dict[str, Any]]:
    """
    Get an application with its configuration.

    Args:
        db: Database session
        application_id: Application ID

    Returns:
        Application data with configuration
    """
    application = await get_application(db, application_id)
    if not application:
        return None

    # Get latest version
    latest_version = None
    if application.versions:
        latest_version = max(application.versions, key=lambda v: v.created_at)

    config = {}
    if latest_version and latest_version.config:
        try:
            import json

            config = json.loads(latest_version.config)
        except (json.JSONDecodeError, TypeError):
            config = {}

    return {
        "id": str(application.id),
        "name": application.name,
        "description": application.description,
        "created_at": application.created_at.isoformat(),
        "updated_at": application.updated_at.isoformat(),
        "config": config,
        "latest_version": latest_version.version if latest_version else None,
    }


async def update_application(
    db: AsyncSession,
    application_id: UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    knowledge_base_ids: Optional[List[UUID]] = None,
) -> Optional[Application]:
    """
    Update an application and create a new version if config changes.

    Args:
        db: Database session
        application_id: Application ID
        name: New application name
        description: New application description
        model_config: New model configuration
        knowledge_base_ids: New knowledge base associations

    Returns:
        Updated application or None
    """
    db_application_id = normalize_uuid(application_id)
    application = await get_application(db, application_id)
    if not application:
        return None

    # Update basic fields
    update_data = {}
    if name is not None:
        update_data["name"] = name
    if description is not None:
        update_data["description"] = description

    if update_data:
        await db.execute(
            update(Application)
            .where(Application.id == db_application_id)
            .values(**update_data)
        )
        await db.commit()
        await db.refresh(application)

    # Create new version if configuration changed
    if config is not None or knowledge_base_ids is not None:
        # Get current config
        current_config = {}
        if application.versions:
            latest_version = max(application.versions, key=lambda v: v.created_at)
            if latest_version.config:
                try:
                    import json

                    current_config = json.loads(latest_version.config)
                except (json.JSONDecodeError, TypeError):
                    current_config = {}

        # Update config
        new_config = current_config.copy()
        if config is not None:
            new_config["model_config"] = config
        if knowledge_base_ids is not None:
            new_config["knowledge_base_ids"] = [
                str(kb_id) for kb_id in knowledge_base_ids
            ]

        # Create new version
        next_version = "1.0"
        if application.versions:
            # Increment version number
            try:
                latest_version_num = float(max(v.version for v in application.versions))
                next_version = f"{latest_version_num + 0.1:.1f}"
            except (ValueError, AttributeError):
                next_version = f"v{len(application.versions) + 1}"

        version = ApplicationVersion(
            application_id=db_application_id,
            version=next_version,
            config=json.dumps(new_config),
        )
        db.add(version)
        await db.commit()
        await db.refresh(application)

    return application


async def delete_application(db: AsyncSession, application_id: UUID) -> bool:
    """
    Delete an application and all its related data (versions, chat messages).

    Args:
        db: Database session
        application_id: Application ID

    Returns:
        True if deleted, False otherwise
    """
    db_application_id = normalize_uuid(application_id)

    # Delete chat messages first
    await db.execute(
        delete(ChatMessage).where(ChatMessage.application_id == db_application_id)
    )

    # Delete versions
    await db.execute(
        delete(ApplicationVersion).where(
            ApplicationVersion.application_id == db_application_id
        )
    )

    # Delete application
    result = await db.execute(
        delete(Application).where(Application.id == db_application_id)
    )

    await db.commit()
    return result.rowcount > 0


async def list_applications(db: AsyncSession) -> List[Dict[str, Any]]:
    """
    List all applications with their latest configuration.

    Args:
        db: Database session

    Returns:
        List of applications with config
    """
    result = await db.execute(
        select(Application).options(selectinload(Application.versions))
    )
    applications = result.scalars().all()

    app_list = []
    for app in applications:
        # Get latest version
        latest_version = None
        if app.versions:
            latest_version = max(app.versions, key=lambda v: v.created_at)

        config = {}
        if latest_version and latest_version.config:
            try:
                import json

                config = json.loads(latest_version.config)
            except (json.JSONDecodeError, TypeError):
                config = {}

        app_list.append(
            {
                "id": str(app.id),
                "name": app.name,
                "description": app.description,
                "created_at": app.created_at.isoformat(),
                "updated_at": app.updated_at.isoformat(),
                "config": config,
                "latest_version": latest_version.version if latest_version else None,
            }
        )

    return app_list


async def get_application_knowledge_bases(
    db: AsyncSession, application_id: UUID
) -> List[UUID]:
    """
    Get knowledge base IDs associated with an application.

    Args:
        db: Database session
        application_id: Application ID

    Returns:
        List of knowledge base IDs
    """
    app_data = await get_application_with_config(db, application_id)
    if not app_data or "config" not in app_data:
        return []

    kb_ids = app_data["config"].get("knowledge_base_ids", [])
    return [UUID(kb_id) for kb_id in kb_ids if kb_id]


async def update_application_model_config(
    db: AsyncSession, application_id: UUID, model_config: Dict[str, Any]
) -> Optional[Application]:
    """
    Update only the model configuration of an application.

    Args:
        db: Database session
        application_id: Application ID
        model_config: New model configuration

    Returns:
        Updated application or None
    """
    return await update_application(db, application_id, config=model_config)


async def update_application_knowledge_bases(
    db: AsyncSession, application_id: UUID, knowledge_base_ids: List[UUID]
) -> Optional[Application]:
    """
    Update knowledge base associations for an application.

    Args:
        db: Database session
        application_id: Application ID
        knowledge_base_ids: List of knowledge base IDs

    Returns:
        Updated application or None
    """
    return await update_application(
        db, application_id, knowledge_base_ids=knowledge_base_ids
    )


# Chat message operations for applications
async def create_chat_message(
    db: AsyncSession,
    application_id: UUID,
    user_message: str,
    bot_message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a chat message for an application.

    Args:
        db: Database session
        application_id: Application ID
        user_message: User's message
        bot_message: Bot's response (optional)

    Returns:
        Created message data
    """
    db_application_id = normalize_uuid(application_id)

    message = ChatMessage(
        application_id=db_application_id,
        user_message=user_message,
        bot_message=bot_message,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)

    return {
        "id": str(message.id),
        "application_id": str(message.application_id),
        "user_message": message.user_message,
        "bot_message": message.bot_message,
        "created_at": message.created_at.isoformat(),
    }


async def get_application_chat_history(
    db: AsyncSession, application_id: UUID, limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get chat history for an application.

    Args:
        db: Database session
        application_id: Application ID
        limit: Maximum number of messages to return

    Returns:
        List of chat messages
    """
    db_application_id = normalize_uuid(application_id)
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.application_id == db_application_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(limit)
    )
    messages = result.scalars().all()

    return [
        {
            "id": str(msg.id),
            "application_id": str(msg.application_id),
            "user_message": msg.user_message,
            "bot_message": msg.bot_message,
            "created_at": msg.created_at.isoformat(),
        }
        for msg in messages
    ]


async def clear_chat_history(db: AsyncSession, application_id: UUID) -> int:
    """Delete all chat messages for an application and return removed count."""
    db_application_id = normalize_uuid(application_id)
    result = await db.execute(
        delete(ChatMessage).where(ChatMessage.application_id == db_application_id)
    )
    await db.commit()
    return result.rowcount or 0
