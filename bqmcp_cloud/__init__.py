"""
BQMCP Cloud - A cloud service for document processing and AI content generation
"""

from .bqmcp_cloud import (
    FastMCP,
    BQMCPCloud,
    generate_ppt,
    generate_title,
    generate_abstract,
    generate_quick_read,
    generate_mind_map,
    generate_deep_read,
    extract_release_date,
    generate_pdf,
    check_key,
)

__version__ = "0.1.0"
__all__ = [
    "FastMCP",
    "BQMCPCloud",
    "generate_ppt",
    "generate_title",
    "generate_abstract",
    "generate_quick_read",
    "generate_mind_map",
    "generate_deep_read",
    "extract_release_date",
    "generate_pdf",
    "check_key",
]
