"""Command line interface for BQMCP Cloud"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Optional

import structlog
from bqmcp_cloud import BQMCPCloud

def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        stream=sys.stderr  # 将日志输出到 stderr
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="BQMCP Cloud Service")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "http"],
        help="Transport method (stdio or http)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="bigquant",
        help="Name for the MCP server"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (optional, can also use OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--proxy",
        type=str,
        help="HTTP proxy URL (optional, can also use HTTP_PROXY environment variable)"
    )
    return parser.parse_args()

def main() -> int:
    """主函数"""
    args = parse_args()
    setup_logging(args.log_level)
    logger = structlog.get_logger(__name__)
    
    try:
        # 获取 API key
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not provided. Please set it via --api-key or OPENAI_API_KEY environment variable")
            return 1
            
        # 获取代理设置
        proxy = args.proxy or os.getenv("HTTP_PROXY")
        
        # 初始化服务
        service = BQMCPCloud(
            api_key=api_key,
            proxy=proxy,
            log_level=args.log_level,
            mcp_name=args.name
        )
        
        # 运行服务
        logger.info("Starting BQMCP Cloud service", transport=args.transport)
        service.run(transport=args.transport)
        return 0
        
    except Exception as e:
        logger.error("Service failed", error=str(e), exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 