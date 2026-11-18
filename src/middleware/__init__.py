"""Middleware package for the Scientific Processor."""

from .logging import RequestLoggingMiddleware

__all__ = ["RequestLoggingMiddleware"]
