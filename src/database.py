"""SQLAlchemy async engine, session factory, and ORM models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, Float, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid7()))
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # A real Postgres ``uuid`` column, but ``as_uuid=False`` so the Python-side value
    # stays a ``str``. LOAD-BEARING, NOT STYLE: with ``as_uuid=True`` SQLAlchemy hands
    # back ``uuid.UUID`` objects, and ``uuid.UUID(t) == t`` is False for every str ``t``
    # — no type checker flags it. That would silently break, at minimum:
    #   * ``persistent_job_manager.get_job``'s cross-tenant guard
    #     (``row.tenant_id != tenant_id``) → 404s the owner on their own job;
    #   * ``_row_to_status`` feeding ``JobStatus.tenant_id: str | None``.
    # NULLABLE on purpose: NULL rows are shared/global job metadata, and the RLS
    # "global-or-own" 4-policy shape (003/005) depends on this nullability.
    tenant_id: Mapped[str | None] = mapped_column(PgUUID(as_uuid=False))
    params: Mapped[Any] = mapped_column(JSONB, nullable=True)
    result_url: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_jobs_tenant_status", "tenant_id", "status"),
        Index("idx_jobs_type_created", "job_type", "created_at"),
    )


class VerificationHistoryRow(Base):
    __tablename__ = "verification_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid7()))
    entity_id: Mapped[str] = mapped_column(String(200), nullable=False)
    # See JobRow.tenant_id: ``as_uuid=False`` keeps the Python value a ``str``.
    # NOT NULL on purpose → the strict single-policy RLS shape.
    tenant_id: Mapped[str] = mapped_column(PgUUID(as_uuid=False), nullable=False)
    integrity_score: Mapped[float] = mapped_column(Float, nullable=False)
    components: Mapped[dict] = mapped_column(JSONB, nullable=False)
    trigger: Mapped[str] = mapped_column(String(50), nullable=False)
    cove_verdict: Mapped[str | None] = mapped_column(String(20), nullable=True)
    formal_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    __table_args__ = (Index("idx_vh_entity", "entity_id", "tenant_id", "computed_at"),)


def create_async_engine_from_url(url: str, **kwargs):
    """Create an async SQLAlchemy engine."""
    return create_async_engine(url, **kwargs)


def create_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the given engine."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
