"""
Structured logging with OpenTelemetry trace correlation.
"""

import os
from typing import Any, cast

import structlog
from kag_common.logging import configure_logging as _kag_common_configure_logging

# Check if OTEL is available
OTEL_AVAILABLE = False
trace: Any = None
try:
    from opentelemetry import trace

    OTEL_AVAILABLE = True
except ImportError:
    pass


def add_trace_context(logger, method_name, event_dict):
    """Inject OpenTelemetry trace context into structured logs."""
    if OTEL_AVAILABLE and trace:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                event_dict["trace_id"] = format(ctx.trace_id, "032x")
                event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def add_service_context(logger, method_name, event_dict):
    """Add service metadata to logs."""
    event_dict["service"] = os.getenv("OTEL_SERVICE_NAME", "scientific-processor")
    event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
    return event_dict


_configured = False


def configure_logging(level: int | None = None, log_file: str | None = None) -> None:
    """Configure structured JSON logging with trace context.

    Delegates to ``kag_common.logging.configure_logging`` rather than
    hand-rolling the setup.

    WHY THIS IS A DELEGATION AND NOT A LOCAL IMPLEMENTATION
    -------------------------------------------------------
    The previous body was a copy of the shared library's setup — including its
    defect. It put the renderer INSIDE the structlog chain and then called
    ``logging.basicConfig(format="%(message)s")`` with no ``ProcessorFormatter``.
    That renders fine for a structlog logger and does NOTHING for a plain
    ``logging.getLogger(__name__)`` record, which printed its bare message with
    no level, no timestamp and no service — invisible both to a JSON
    ``"level":"error"`` query and to an ``ERROR`` grep.

    ``kag_common`` 0.23.0 fixed that at the root: the renderer moves onto the
    HANDLER via ``ProcessorFormatter``, and ``basicConfig`` (which silently
    no-ops when handlers already exist) is replaced by an explicit
    ``root.handlers[:] = [handler]``. Copying the setup locally is precisely
    what kept this service on the broken version while the library was fixed, so
    the fix here is to stop copying, not to re-copy the new one.

    Every knob this module used is covered by the library's signature:
    ``service_name`` replaces the local ``add_service_context`` default and
    ``suppress_loggers`` replaces the local noisy-logger loop. ``LOG_FORMAT`` and
    ``LOG_LEVEL`` are read by the library from the same env vars.

    Args:
        level: Logging level (defaults to INFO or LOG_LEVEL env var)
        log_file: Accepted for call-site back-compat (``src/server.py`` passes
            one); ignored, as it was here before — structlog logs to stdout.
    """
    global _configured

    if _configured:
        return

    _kag_common_configure_logging(
        level=level,
        log_file=log_file,
        service_name="scientific-processor",
        # Preserved verbatim from the local implementation this replaces.
        suppress_loggers=("httpx", "httpcore", "uvicorn.access", "urllib3"),
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a module-level structured logger.

    Args:
        name: Logger name (defaults to "scientific_processor")

    Returns:
        Configured structlog logger
    """
    configure_logging()
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name or "scientific_processor"))
