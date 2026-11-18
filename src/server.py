"""
Convenience entrypoints for running the scientific processor service via uvicorn.
"""

from __future__ import annotations

import os

import uvicorn

from app import create_app
from config import Settings, get_settings
from log_config import configure_logging, get_logger

logger = get_logger(__name__)

# Create app instance for uvicorn to use
app = create_app()


def run_uvicorn(settings: Settings | None = None) -> None:
    """Run the FastAPI app using uvicorn."""
    settings = settings or get_settings()
    # Use /app/logs in container, local path otherwise
    log_dir = "/app/logs" if os.path.exists("/app/logs") else os.path.expanduser("~/Movies/.logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_logging(log_file=os.path.join(log_dir, "scientific_processor.log"))
    logger.info(
        "Starting scientific processor server",
        extra={"host": settings.bind_host, "port": settings.bind_port},
    )

    uvicorn.run(
        "app:create_app",
        factory=True,
        host=settings.bind_host,
        port=settings.bind_port,
        reload=settings.reload,
        loop="asyncio",
    )


def main() -> None:
    """CLI entrypoint for `python -m scientific_processor`."""
    run_uvicorn()


if __name__ == "__main__":
    main()
