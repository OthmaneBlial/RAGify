"""add application_id to documents table

Revision ID: edb084f40a3e
Revises: 
Create Date: 2025-09-08 23:02:13.397630

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'edb084f40a3e'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add application_id column to documents table
    op.add_column('documents', sa.Column('application_id', UUID(as_uuid=True), nullable=True))
    # Add foreign key constraint
    op.create_foreign_key(
        'fk_documents_application_id',
        'documents',
        'applications',
        ['application_id'],
        ['id']
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop foreign key constraint
    op.drop_constraint('fk_documents_application_id', 'documents', type_='foreignkey')
    # Drop application_id column
    op.drop_column('documents', 'application_id')
