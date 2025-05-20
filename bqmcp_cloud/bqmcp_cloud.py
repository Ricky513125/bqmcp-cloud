import asyncio
import base64
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import aiofiles
import httpx
import jieba
import structlog
from openai import AsyncOpenAI, OpenAI
from mcp.server.fastmcp import FastMCP
import fitz
from dotenv import load_dotenv

load_dotenv()

class BQMCPCloud(FastMCP):
    """BQMCP Cloud service for document processing and AI content generation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        base_output_path: str = "outputs",
        log_level: str = "INFO",
        name: str = "bigquant"
    ):
        """
        Initialize BQMCP Cloud service
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            proxy: HTTP proxy URL. If None, will try to get from environment variable HTTP_PROXY
            base_output_path: Base path for output files
            log_level: Logging level
            name: Name for the MCP server
        """
        super().__init__(name, log_level=log_level)
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")
            
        self.proxy = proxy or os.getenv("HTTP_PROXY", "http://39.104.58.112:31701")
        self.base_output_path = base_output_path
        self.base_image_local_path = os.path.join(base_output_path, "images")
        
        # Initialize logger
        self.logger = structlog.get_logger(__name__)
        
        # Load system prompts
        self.prompts = self._load_system_prompts()
        
        # Ensure output directories exist
        self._ensure_output_dirs()
        
        # Register tools
        self._register_tools()
    
    def _ensure_output_dirs(self):
        """Create output directories if they don't exist"""
        paths = [self.base_output_path, self.base_image_local_path]
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def _load_system_prompts(self) -> Dict[str, str]:
        """Load all system prompts from files"""
        prompts = {}
        prompt_files = {
            "DOC_SYSTEM_PROMPT": "prompts/doc_system_prompt.md",
            "DEEP_READ_SYSTEM_PROMPT": "prompts/deep_read_system_prompt.md",
            "PPT_SYSTEM_PROMPT": "prompts/ppt_system_prompt.md",
            "STRATEGY_SYSTEM_PROMPT": "prompts/strategy_system_prompt.md",
            "RELEASE_DATE_SYSTEM_PROMPT": "prompts/release_date_system_prompt.md"
        }
        
        for name, path in prompt_files.items():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    prompts[name] = f.read()
            except FileNotFoundError:
                self.logger.error(f"Could not find prompt file: {path}")
                prompts[name] = "System prompt unavailable"
        
        return prompts
    
    def get_openai_client(self) -> AsyncOpenAI:
        """Create and return an AsyncOpenAI client"""
        return AsyncOpenAI(
            api_key=self.api_key,
            http_client=httpx.AsyncClient(proxy=self.proxy),
        )
    
    def _register_tools(self):
        """Register all MCP tools"""
        # Register all the tool methods
        self.tool()(self.check_key)
        self.tool()(self.generate_ppt)
        self.tool()(self.generate_title)
        self.tool()(self.generate_abstract)
        self.tool()(self.generate_quick_read)
        self.tool()(self.generate_mind_map)
        self.tool()(self.generate_deep_read)
        self.tool()(self.extract_release_date)
        self.tool()(self.generate_pdf)
    
    @FastMCP.tool()
    async def check_key(self) -> dict:
        """
        Name: Check API Key
        Description: Check if the current API key is valid
        Returns: API key status
        """
        return {"status": "valid", "key": self.api_key[:8] + "..."}
    
    @FastMCP.tool()
    async def generate_ppt(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate PPT
        Description: Generate PPT content based on input text or markdown
        Args:
            user_content: Input text or markdown for PPT generation
        Returns: Generated PPT content
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        paper_ppt_content = await self._call_chatgpt(
            processed_content, 
            self.prompts["PPT_SYSTEM_PROMPT"], 
            openai_client
        )

        return {"ppt_content": paper_ppt_content}

    @FastMCP.tool()
    async def generate_title(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate Title
        Description: Generate a title based on input text or markdown
        Args:
            user_content: Input text or markdown for title generation
        Returns: Generated title
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        result = await self._call_chatgpt(
            processed_content, 
            self.prompts["DOC_SYSTEM_PROMPT"], 
            openai_client,
            pattern_tags=["paper_title"]
        )
        
        return {"title": result.get("paper_title", "")}

    @FastMCP.tool()
    async def generate_abstract(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate Abstract
        Description: Generate an abstract based on input text or markdown
        Args:
            user_content: Input text or markdown for abstract generation
        Returns: Generated abstract
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        result = await self._call_chatgpt(
            processed_content, 
            self.prompts["DOC_SYSTEM_PROMPT"], 
            openai_client,
            pattern_tags=["paper_abstract"]
        )
        
        return {"abstract": result.get("paper_abstract", "")}

    @FastMCP.tool()
    async def generate_quick_read(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate Quick Read
        Description: Generate quick read content based on input text or markdown
        Args:
            user_content: Input text or markdown for quick read generation
        Returns: Generated quick read content
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        result = await self._call_chatgpt(
            processed_content, 
            self.prompts["DOC_SYSTEM_PROMPT"], 
            openai_client,
            pattern_tags=["quick_read"]
        )
        
        return {"quick_read": result.get("quick_read", "")}

    @FastMCP.tool()
    async def generate_mind_map(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate Mind Map
        Description: Generate mind map content based on input text or markdown
        Args:
            user_content: Input text or markdown for mind map generation
        Returns: Generated mind map content
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        result = await self._call_chatgpt(
            processed_content, 
            self.prompts["DOC_SYSTEM_PROMPT"], 
            openai_client,
            pattern_tags=["mind_map"]
        )
        
        return {"mind_map": result.get("mind_map", "")}

    @FastMCP.tool()
    async def generate_deep_read(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate Deep Read
        Description: Generate deep read content based on input text or markdown
        Args:
            user_content: Input text or markdown for deep read generation
        Returns: Generated deep read content
        """
        openai_client = self.get_openai_client()
        
        if isinstance(user_content, str):
            processed_content = [{"type": "text", "text": user_content}]
        else:
            processed_content = user_content
        
        deep_read_content = await self._call_chatgpt(
            processed_content, 
            self.prompts["DEEP_READ_SYSTEM_PROMPT"], 
            openai_client
        )
        
        return {"deep_read": deep_read_content}

    @FastMCP.tool()
    async def extract_release_date(self, content: Union[List, str]) -> dict:
        """
        Name: Extract Release Date
        Description: Extract release date from input content
        Args:
            content: Input content to extract date from
        Returns: Extracted release date
        """
        openai_client = self.get_openai_client()
        
        if isinstance(content, list):
            content = self._extract_json_to_str(content)
        
        result = await self._call_chatgpt(
            content, 
            self.prompts["RELEASE_DATE_SYSTEM_PROMPT"], 
            openai_client,
            pattern_tags=["release_date"]
        )
        
        release_date = result.get("release_date", "")
        normalized_date = self._normalize_datetime_string(release_date)
        
        return {"release_date": normalized_date or release_date}

    @FastMCP.tool()
    async def generate_pdf(self, user_content: Union[List, str]) -> dict:
        """
        Name: Generate PDF
        Description: Generate PDF content based on input text or markdown
        Args:
            user_content: Input text or markdown for PDF generation
        Returns: Generated PDF content
        """
        client = OpenAI(
            api_key=self.api_key,
            http_client=httpx.Client(proxy=self.proxy),
        )
        messages = [{"role": "system", "content": "根据描述的文本或者markdown，执行生成任务。返回生成的PDF的内容"}]
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=messages,
            stream=False,
        )
        return {"pdf_content": response.choices[0].message.content}
    
    async def _call_chatgpt(
        self,
        user_content: Union[List, str],
        system_prompt: str,
        openai_client: AsyncOpenAI,
        pattern_tags: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> Union[Dict, str]:
        """Call OpenAI ChatGPT API with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
                response = await openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )
                break
            except Exception as e:
                self.logger.error(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    raise
                
        response = response.choices[0].message.content
        if not pattern_tags:
            return response
            
        result = {}
        for tag in pattern_tags:
            content = ""
            pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
            match = pattern.search(response)
            if match:
                content = match.group(1)
            result[tag] = content
        return result

    def _extract_json_to_str(self, obj) -> str:
        """Convert JSON object to string format"""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            return " ".join(self._extract_json_to_str(v) for v in obj.values())
        elif isinstance(obj, list):
            return " ".join(self._extract_json_to_str(i) for i in obj)
        return ""

    def _normalize_datetime_string(self, dt_str: str) -> Optional[str]:
        """Normalize datetime string to standard format"""
        if not dt_str or not isinstance(dt_str, str) or dt_str.strip() == "":
            return None

        dt_str = dt_str.strip()
        formats = [
            ("%Y", "%Y-01-01 00:00:00.000000"),
            ("%Y-%m", "%Y-%m-01 00:00:00.000000"),
            ("%Y-%m-%d", "%Y-%m-%d 00:00:00.000000"),
            ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:00.000000"),
            ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.000000"),
            ("%Y-%m-%d %H:%M:%S.%f", None),
        ]

        for fmt, fill_template in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                if fill_template:
                    filled_str = dt.strftime(fill_template)
                    filled_dt = datetime.strptime(filled_str, "%Y-%m-%d %H:%M:%S.%f")
                    return filled_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                return dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                continue

        return None

# 启动MCP服务器
if __name__ == "__main__":
    # 创建 BQMCPCloud 实例
    cloud = BQMCPCloud()
    # 运行服务器
    cloud.run(transport='stdio')