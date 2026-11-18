"""
CLI entry point for the scientific processor service.

Allows running `python -m scientific_processor` to start the uvicorn server.
"""

from server import main

if __name__ == "__main__":
    main()
