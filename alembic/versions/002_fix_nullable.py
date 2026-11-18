"""Fix nullable mismatches on jobs.progress, jobs.created_at, jobs.updated_at.

These columns are Mapped[float] / Mapped[datetime] (non-nullable) in the ORM
model but were created without explicit ``nullable=False`` in migration 001,
making them nullable at the database level.  jobs.status already had
``nullable=False`` in 001; we include it here for completeness so that
server_default is guaranteed set.

Revision ID: 002
Revises: 001
Create Date: 2026-03-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- jobs.status (should already be NOT NULL, ensure server_default) ---
    op.alter_column(
        "jobs",
        "status",
        existing_type=sa.String(20),
        nullable=False,
        server_default="pending",
    )

    # --- jobs.progress ---
    op.execute("UPDATE jobs SET progress = 0.0 WHERE progress IS NULL")
    op.alter_column(
        "jobs",
        "progress",
        existing_type=sa.Float(),
        nullable=False,
        server_default="0.0",
    )

    # --- jobs.created_at ---
    op.execute("UPDATE jobs SET created_at = now() WHERE created_at IS NULL")
    op.alter_column(
        "jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )

    # --- jobs.updated_at ---
    op.execute("UPDATE jobs SET updated_at = now() WHERE updated_at IS NULL")
    op.alter_column(
        "jobs",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("now()"),
    )


def downgrade() -> None:
    op.alter_column(
        "jobs",
        "updated_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
        server_default=sa.text("now()"),
    )
    op.alter_column(
        "jobs",
        "created_at",
        existing_type=sa.DateTime(timezone=True),
        nullable=True,
        server_default=sa.text("now()"),
    )
    op.alter_column(
        "jobs",
        "progress",
        existing_type=sa.Float(),
        nullable=True,
        server_default="0.0",
    )
    # status was already NOT NULL in 001, so downgrade is a no-op for it.
