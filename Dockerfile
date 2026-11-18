# -----------------------------
# Stage 0: Base (shared)
# -----------------------------
ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH"

WORKDIR /app

# Multi-arch compatible uv binary
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /usr/local/bin/


# -----------------------------
# Stage 1: Builder
# -----------------------------
FROM base AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Prime cache with metadata + lock. uv.lock is REQUIRED (no glob): a missing
# lock must fail the build, and the image never re-resolves. An in-image
# `uv lock` / unlocked install is how pyjwt 2.15.1 shipped while the lock said
# 2.14.0 and broke auth estate-wide (gitops#4248, license-authz#63).
COPY README.md pyproject.toml uv.lock ./

# Install dependencies only, exactly as locked (cached layer - not invalidated by code changes)
RUN uv sync --frozen --no-dev --no-install-project

# App source after deps for better cache reuse
COPY src/ /app/src/

# Install the local project (fast - deps already cached)
RUN uv sync --frozen --no-dev

# Build-time guard: fail the build if any installed version != uv.lock
# (pyjwt asserted explicitly). Same script as license-authz#63.
COPY scripts/assert_lock_versions.py /tmp/assert_lock_versions.py
RUN uv export --frozen --no-dev --no-emit-project --no-hashes -o /tmp/lock-pins.txt && \
    /app/.venv/bin/python /tmp/assert_lock_versions.py uv.lock /tmp/lock-pins.txt && \
    uv sync --frozen --no-dev --check


# -----------------------------
# Stage 2: Production
# -----------------------------
FROM base AS production

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src

# Create the runtime user and chown /app (non-recursive) before the copies
# below land the venv, rather than a recursive chown after: a `chown -R`
# over a populated .venv walks every inode a second time and forces a full
# non-cacheable layer re-upload -- measured 702s of the build in this repo.
# Each COPY below carries --chown so the final ownership is identical.
RUN useradd -m -u 1000 appuser && chown appuser:appuser /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock
COPY --from=builder --chown=appuser:appuser /app/README.md /app/README.md
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser scripts/ /app/scripts/
COPY --chown=appuser:appuser alembic.ini /app/alembic.ini
COPY --chown=appuser:appuser alembic /app/alembic

# Non-root execution
USER appuser

EXPOSE 8808
ENV PORT=8808

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8808", "--workers", "1"]
