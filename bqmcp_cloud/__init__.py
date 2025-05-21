"""
BQMCP Cloud - 基于 FastMCP 的文档处理和 AI 内容生成云服务

这个包提供了一个基于 FastMCP 的云服务，用于文档处理和 AI 内容生成。
它支持多种 AI 驱动的内容生成功能，包括：

- PPT 生成
- 标题生成
- 摘要生成
- 快速阅读内容生成
- 思维导图生成
- 深度阅读内容生成
- 发布日期提取
- PDF 生成

使用示例：
    >>> from bqmcp_cloud import BQMCPCloud
    >>> cloud = BQMCPCloud(api_key="your-openai-api-key")
    >>> await cloud.generate_title("你的内容")
"""

import asyncio
from .bqmcp_cloud import (
    BQMCPCloud,
    BQTools,
    ContentInput,
    KeyCheckResult,
    GenerationResult,
    ReleaseDateResult,
)

def main():
    """命令行入口点，用于启动 BQMCP Cloud 服务"""
    import argparse
    import sys
    import structlog

    parser = argparse.ArgumentParser(
        description="BQMCP Cloud - 文档处理和 AI 内容生成云服务"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (默认从 OPENAI_API_KEY 环境变量获取)"
    )
    parser.add_argument(
        "--proxy",
        help="HTTP 代理 URL (默认从 HTTP_PROXY 环境变量获取)"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="输出文件的基础目录 (默认: outputs)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    parser.add_argument(
        "--name",
        default="bigquant",
        help="MCP 服务器名称 (默认: bigquant)"
    )

    args = parser.parse_args()

    # 配置日志
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

    try:
        # 初始化服务
        cloud = BQMCPCloud(
            api_key=args.api_key,
            proxy=args.proxy,
            base_output_path=args.output_dir,
            log_level=args.log_level,
            name=args.name
        )
        
        # 运行服务
        asyncio.run(cloud.serve())
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

__version__ = "0.1.5"  # 与 pyproject.toml 保持一致
__author__ = "Ricky Li"
__email__ = "lingyuli513125@gmail.com"

__all__ = [
    "BQMCPCloud",
    "BQTools",
    "ContentInput",
    "KeyCheckResult",
    "GenerationResult",
    "ReleaseDateResult",
]
