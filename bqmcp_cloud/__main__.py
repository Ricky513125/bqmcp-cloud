"""Entry point for running the package directly with python -m bqmcp_cloud"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
