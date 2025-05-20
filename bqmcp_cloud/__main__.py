"""BQMCP Cloud Service - Main entry point

This module provides the main entry point for running the BQMCP Cloud service
as a Python module using `python -m bqmcp_cloud`.
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
