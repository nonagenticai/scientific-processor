"""Shared pytest configuration.

Forces ENVIRONMENT=test before any `from config import Settings` import so
the production fail-fast validator in `Settings.__post_init__` does not
trip when tests instantiate `Settings()` without a fully-populated env.

Per gitops-production/docs/config-externalization-policy.md, infrastructure
URLs (graph-adapter, pytorch-backend, inference-router, ...) are required
in production and only in production.
"""

import os

os.environ.setdefault("ENVIRONMENT", "test")
