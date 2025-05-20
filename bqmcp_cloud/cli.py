"""Command line interface for BQMCP Cloud"""

import argparse
import os
import sys
from typing import Optional

from .bqmcp_cloud import BQMCPCloud

def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="BQMCP Cloud - Document processing and AI content generation service")
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)",
        default=os.getenv("OPENAI_API_KEY")
    )
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL (default: HTTP_PROXY environment variable)",
        default=os.getenv("HTTP_PROXY")
    )
    parser.add_argument(
        "--output-dir",
        help="Base output directory (default: 'outputs')",
        default="outputs"
    )
    parser.add_argument(
        "--log-level",
        help="Logging level (default: INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )
    parser.add_argument(
        "--name",
        help="MCP server name (default: 'bigquant')",
        default="bigquant"
    )
    parser.add_argument(
        "--transport",
        help="Transport type (default: 'stdio')",
        choices=["stdio", "http"],
        default="stdio"
    )
    
    args = parser.parse_args(argv)
    
    try:
        cloud = BQMCPCloud(
            api_key=args.api_key,
            proxy=args.proxy,
            base_output_path=args.output_dir,
            log_level=args.log_level,
            mcp_name=args.name
        )
        cloud.run(transport=args.transport)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 