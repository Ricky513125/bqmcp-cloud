"""
BQMCP Cloud - A cloud service for document processing and AI content generation

This package provides a cloud service for document processing and AI content generation,
built on top of FastMCP. It supports PDF processing, content extraction, and various
AI-powered content generation features.
"""

from .bqmcp_cloud import BQMCPCloud, FastMCP

def main():
    import argparse
    import asyncio 

    parser = argparse.ArgumentParser(
        description="give a model the ability to handle time queries and timezone conversions"
    )
    parser.add_argument("--local-timezone", type=str, help="Override local timezone")

    args = parser.parse_args()
    asyncio.run(serve(args.local_timezone))


if __name__ == "__main__":
    main()






__version__ = "0.1.0"
__author__ = "Ricky Li"
__email__ = "lingyuli513125@gmail.com"
__all__ = ["BQMCPCloud", "FastMCP"]
