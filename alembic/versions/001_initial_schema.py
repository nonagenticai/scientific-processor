"""Initial schema — jobs and verification_history tables.

Revision ID: 001
Create Date: 2026-03-19
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("tenant_id", sa.String(100), nullable=True),
        sa.Column("params", postgresql.JSONB(), nullable=True),
        sa.Column("result_url", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("progress", sa.Float(), server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_jobs_tenant_status", "jobs", ["tenant_id", "status"])
    op.create_index("idx_jobs_type_created", "jobs", ["job_type", "created_at"])

    op.create_table(
        "verification_history",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("entity_id", sa.String(200), nullable=False),
        sa.Column("tenant_id", sa.String(100), nullable=False),
        sa.Column("integrity_score", sa.Float(), nullable=False),
        sa.Column("components", postgresql.JSONB(), nullable=False),
        sa.Column("trigger", sa.String(50), nullable=False),
        sa.Column("cove_verdict", sa.String(20), nullable=True),
        sa.Column("formal_result", postgresql.JSONB(), nullable=True),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_vh_entity", "verification_history", ["entity_id", "tenant_id", "computed_at"])


def downgrade() -> None:
    op.drop_index("idx_vh_entity", table_name="verification_history")
    op.drop_table("verification_history")
    op.drop_index("idx_jobs_type_created", table_name="jobs")
    op.drop_index("idx_jobs_tenant_status", table_name="jobs")
    op.drop_table("jobs")
