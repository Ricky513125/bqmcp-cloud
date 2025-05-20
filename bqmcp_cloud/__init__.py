"""
BQMCP Cloud - A cloud service for document processing and AI content generation

This package provides a cloud service for document processing and AI content generation,
built on top of FastMCP. It supports PDF processing, content extraction, and various
AI-powered content generation features.
"""

from .bqmcp_cloud import BQMCPCloud, FastMCP

__version__ = "0.1.0"
__author__ = "Ricky Li"
__email__ = "lingyuli513125@gmail.com"
__all__ = ["BQMCPCloud", "FastMCP"]
