"""Command line interface for BQMCP Cloud"""

import argparse
import asyncio
import logging
import os
import sys
import structlog

from .bqmcp_cloud import BQMCPCloud

def setup_logging(log_level: str = "INFO"):
    """Configure logging with structlog"""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set log level
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper()),
    )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="BQMCP Cloud service")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--name",
        default="bigquant",
        help="Name for the MCP server (default: bigquant)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (default: from OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL (default: from HTTP_PROXY environment variable)"
    )
    return parser.parse_args()

async def async_main():
    """Async main entry point for the CLI"""
    args = parse_args()
    setup_logging(args.log_level)
    logger = structlog.get_logger(__name__)
    
    try:
        # Initialize service
        service = BQMCPCloud(
            api_key=args.api_key,
            proxy=args.proxy,
            log_level=args.log_level,
            name=args.name
        )
        
        # Start service
        logger.info("Starting BQMCP Cloud service", 
                   name=args.name)
        
        await service.serve()
            
    except Exception as e:
        logger.error("Failed to start service", error=str(e), exc_info=True)
        sys.exit(1)

def main():
    """Main entry point for the CLI"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main() 