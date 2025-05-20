import asyncio
import base64
import json
import os
import re
from datetime import datetime
from typing import Any, List, Dict, Union, Optional

import aiofiles
import httpx
import jieba
import structlog
from openai import AsyncOpenAI, OpenAI
from mcp.server.fastmcp import FastMCP 
import fitz
from dotenv import load_dotenv
load_dotenv()

class BQMCPCloud:
    """BQMCP Cloud service for document processing and AI content generation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        proxy: Optional[str] = None,
        base_output_path: str = "outputs",
        log_level: str = "INFO",
        mcp_name: str = "bigquant"
    ):
        """
        Initialize BQMCP Cloud service
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            proxy: HTTP proxy URL. If None, will try to get from environment variable HTTP_PROXY
            base_output_path: Base path for output files
            log_level: Logging level
            mcp_name: Name for the MCP server
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")
            
        self.proxy = proxy or os.getenv("HTTP_PROXY", "http://39.104.58.112:31701")
        self.base_output_path = base_output_path
        self.base_image_local_path = os.path.join(base_output_path, "images")
        
        # Initialize logger
        self.logger = structlog.get_logger(__name__)
        
        # Initialize MCP server
        self.mcp = FastMCP(mcp_name, log_level=log_level)
        
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
        self.mcp.tool()(self.check_key)
        self.mcp.tool()(self.generate_ppt)
        self.mcp.tool()(self.generate_title)
        self.mcp.tool()(self.generate_abstract)
        self.mcp.tool()(self.generate_quick_read)
        self.mcp.tool()(self.generate_mind_map)
        self.mcp.tool()(self.generate_deep_read)
        self.mcp.tool()(self.extract_release_date)
        self.mcp.tool()(self.generate_pdf)
    
    async def check_key(self) -> str:
        """Check the current API key configuration"""
        return self.api_key
    
    def run(self, transport: str = 'stdio'):
        """Run the MCP server"""
        self.mcp.run(transport=transport)

# 为了向后兼容，保留全局函数
def get_openai_client(api_key: Optional[str] = None, proxy: Optional[str] = None) -> AsyncOpenAI:
    """Create and return an AsyncOpenAI client (legacy function)"""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    proxy = proxy or os.getenv("HTTP_PROXY", "http://39.104.58.112:31701")
    return AsyncOpenAI(
        api_key=api_key,
        http_client=httpx.AsyncClient(proxy=proxy),
    )

# 为了向后兼容，保留全局变量和函数
BASE_OUTPUT_PATH = "outputs"
BASE_IMAGE_LOCAL_PATH = os.path.join(BASE_OUTPUT_PATH, "images")

# 为了向后兼容，保留全局函数
ensure_output_dirs = lambda: os.makedirs(BASE_OUTPUT_PATH, exist_ok=True) and os.makedirs(BASE_IMAGE_LOCAL_PATH, exist_ok=True)
ensure_output_dirs()

# Create output directory for a specific PDF
def create_pdf_output_dir(pdf_path: str) -> str:
    """Create an output directory based on the PDF filename"""
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(BASE_OUTPUT_PATH, pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    return output_dir 


async def extrac_pdf_content(pdf_path: str) -> dict:
    """
    Extract text and images from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary with extracted content and metadata
    
    """
    # Create output directory based on PDF name 
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = create_pdf_output_dir(pdf_path)
    images_dir = os.path.join(output_dir, "images")

    # Open PDF file 
    try:
        doc = fitz.open(pdf_path)

        content_list = [] 
        file_id = pdf_name # Using PDF name as file_id 

        # Extract content from each page 
        for page_idx, page in enumerate(doc):
            # Extract text 
            text = page.get_text() 
            if text.strip():
                content_list.append({
                    "type": "text", 
                    "text": text, 
                    "page_idx": page_idx 
                })
                # tuple 

            # Extract images 
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                xref = img[0] 
                base_img = doc.extract_image(xref)
                image_bytes = base_img["image"]

                # Determine file extension based on image type 
                ext = base_img["ext"]
                img_filename = f"page{page_idx}_img{img_idx}.{ext}"
                img_path = os.path.join(images_dir, img_filename)

                # Save image to file 
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # Add image referrence content list 
                content_list.append({
                    "type": "image",
                    "img_path": f"images/{img_filename}",
                    "page_idx": page_idx 
                })

        # Save extracted content as JSON 
        content_json_path = os.path.join(output_dir, f"{pdf_name}_content.json")
        with open(content_json_path, "w", encoding="utf-8") as f:
            json.dump(content_list, f, ensure_ascii=False, indent=2)

        return {
            "pdf_name": pdf_name,
            "file_id": file_id,
            "content_list": content_list,
            "output_dir": output_dir, 
            "num_pages": len(doc),
            "content_json_path": content_json_path
        }
    
    except Exception as e:
        logger.error(f"Failded to  extract content from PDF: {e}")
        raise 


# Load all prompts
PROMPTS = load_system_prompts()

# Save output content to file 
async def save_content_to_file(content: str, output_dir: str, filename: str) -> str:
    """
    Save content to a file in the specified output directory

    Args:
        content: Content to save 
        output_dir: Directory to save the file in 
        filename: Name of the file to save 

    Returns:
        Full path to the saved file.    
    """
    filepath = os.join(output_dir, filename)

    # Ensure the directory exists 
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write the content to the file 
    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        await f.write(content)

    return filepath


def tokenize(text: str) -> str:
    """使用 jieba 对文本进行分词并去除空白符"""
    cleaned = re.sub(r"\[(pidx|page)::[^\]]+\]", "", text)
    return " ".join([word for word in jieba.cut(cleaned) if word.strip()])

def extract_json_to_str(obj):
    """将json转化为纯字符串格式"""
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return " ".join(extract_json_to_str(v) for v in obj.values())
    elif isinstance(obj, list):
        return " ".join(extract_json_to_str(i) for i in obj)
    return ""

def normalize_datetime_string(dt_str: str) -> Optional[str]:
    """时间处理函数，防止大模型抽取出错导致报错"""
    if not dt_str or not isinstance(dt_str, str) or dt_str.strip() == "":
        return None

    dt_str = dt_str.strip()

    # 可接受的格式以及补全后的标准格式
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

def markdown_to_openai_message(md_content: str, output_dir: str, pdf_name: str):
    """将 Markdown 文本（可能包含单个图片链接）转换为 OpenAI 消息格式。
    如果找到图片，使用基础 URL 构建完整的图片 URL。
    如果未找到图片，则生成纯文本消息。

    Args: 
        md_content: Markdown content 
        output_dir: Directory where outputs are stored 
        pdf_name: Name of the PDF file(without extension)

    Returns:
        List of content parts for OpenAI API 
    """
    content_parts = []
    last_index = 0
    image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    matches = list(image_pattern.finditer(md_content))

    if not matches:
        full_text = md_content.strip()
        if full_text:
            content_parts.append({"type": "text", "text": full_text})
    else:
        for match in matches:
            start_index = match.start()
            text_segment = md_content[last_index:start_index].strip()
            if text_segment:
                content_parts.append({"type": "text", "text": text_segment})

            relative_image_path = match.group(1)
            if relative_image_path:
                # Use local image path 
                image_path = os.path.join(output_dir, "images", relative_image_path)
                if os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    content_parts.append({
                        "type": "text", "text": f"image_path: {relative_image_path}"
                    })
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                
            last_index = match.end()
            
        remaining_text = md_content[last_index:].strip()
        if remaining_text:
            content_parts.append({"type": "text", "text": remaining_text})
    return content_parts

def _add_content_item_fully_merged(content_list: list, new_item: dict):
    """Adds an item to the content list. Merges the new item if both it
    and the last item are of type 'text'.
    """
    if new_item["type"] == "text" and not new_item.get("text", "").strip():
        return

    can_merge = (
        content_list  # 列表不为空
        and content_list[-1]["type"] == "text"  # 最后一项是文本
        and new_item["type"] == "text"  # 新项也是文本
    )

    if can_merge:
        content_list[-1]["text"] += "\n\n" + new_item["text"]
    else:
        content_list.append(new_item)

# 这一个貌似被上面的取代了
def content_to_openai_message(content_list: list, file_id: str) -> list:
    """将 mineru 识别的 content list 转换为 OpenAI 消息格式。"""
    openai_content_list = []
    current_page_idx = -1

    for item in content_list:
        page_idx = item.get("page_idx", -1)

        if page_idx != current_page_idx:
            if current_page_idx != -1:
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": f"[page_idx:{current_page_idx}:end]"})
            _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": f"[page_idx:{page_idx}:begin]"})
            current_page_idx = page_idx

        item_type = item.get("type")

        if item_type == "text":
            text = item.get("text", "")
            text_level = item.get("text_level", 0)
            formatted_text = f"{'#' * text_level} {text}" if text_level > 0 else text
            if formatted_text:
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": formatted_text})

        elif item_type == "equation":
            _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": item.get("text", "")})

        elif item_type == "image":
            img_path = item.get("img_path", "")

            img_caption = item.get("img_caption", [])
            if img_caption:
                caption_text = "\n".join(img_caption)
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": caption_text})

            if img_path:
                # Using BASE_IMAGE_LOCAL_PATH variable directly
                image_local_path = f"{BASE_IMAGE_LOCAL_PATH}/{file_id[:2]}/{file_id}/{img_path}"
                text_image_path = f"{img_path}?page={page_idx}"
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": f"image_path: {text_image_path}"})
                try:
                    with open(image_local_path, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    _add_content_item_fully_merged(openai_content_list, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
                except FileNotFoundError:
                    logger.error(f"Could not find image file: {image_local_path}")

            img_footnote = item.get("img_footnote", [])
            if img_footnote:
                footnote_text = "\n".join(img_footnote)
                if footnote_text:
                    _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": footnote_text})

        elif item_type == "table":
            table_body = item.get("table_body", "")

            table_caption = item.get("table_caption", [])
            if table_caption:
                caption_text = "\n".join(table_caption)
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": caption_text})

            if table_body:
                stripped_body = table_body.strip()
                if stripped_body:
                    _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": stripped_body})

            table_footnote = item.get("table_footnote", [])
            if table_footnote:
                footnote_text = "\n".join(table_footnote)
                _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": footnote_text})

    if current_page_idx != -1:
        _add_content_item_fully_merged(openai_content_list, {"type": "text", "text": f"[page_idx:{current_page_idx}:end]"})

    return openai_content_list

async def call_chatgpt(
    user_content: Union[List, str],
    system_prompt: str,
    openai_client: AsyncOpenAI,
    pattern_tags: Optional[List[str]] = None,
    model: str = "gpt-4.1",
) -> Union[Dict, str]:
    """访问 openai chatgpt"""
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
            logger.error(f"Request failed on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
    response = response.choices[0].message.content
    result = {}
    if pattern_tags:
        for tag in pattern_tags:
            content = ""
            pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
            match = pattern.search(response)
            if match:
                content = match.group(1)
            result[tag] = content
    return result or response

@mcp.tool()
async def generate_ppt(user_content: Union[List, str]) -> dict:
    """
    Name:
        PPT生成
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的PPT的内容
    Args:
        user_content: 输入描述的文本或markdown，生成PPT 
    Returns:
        生成的PPT内容
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        # 如果有必要，进行处理，例如将纯文本转换为合适的格式
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用PPT系统提示生成PPT
    paper_ppt_content = await call_chatgpt(
        processed_content, 
        PROMPTS["PPT_SYSTEM_PROMPT"], 
        openai_client
    )

    return {"ppt_content": paper_ppt_content}

@mcp.tool()
async def generate_title(user_content: Union[List, str]) -> dict:
    """
    Name:
        生成研报标题
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的标题的字符串
    Args:
        user_content: 输入描述的文本或markdown，生成文本的标题
    Returns:
        生成的标题
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用系统提示生成标题
    result = await call_chatgpt(
        processed_content, 
        PROMPTS["DOC_SYSTEM_PROMPT"], 
        openai_client,
        pattern_tags=["paper_title"]
    )
    
    return {"title": result.get("paper_title", "")}

@mcp.tool()
async def generate_abstract(user_content: Union[List, str]) -> dict:
    """
    Name:
        研报或论文的摘要生成
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的摘要
    Args:
        user_content: 输入描述的文本或markdown，生成摘要
    Returns:
        生成的摘要
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用系统提示生成摘要
    result = await call_chatgpt(
        processed_content, 
        PROMPTS["DOC_SYSTEM_PROMPT"], 
        openai_client,
        pattern_tags=["paper_abstract"]
    )
    
    return {"abstract": result.get("paper_abstract", "")}

@mcp.tool()
async def generate_quick_read(user_content: Union[List, str]) -> dict:
    """
    Name:
        速读内容生成
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的速读内容
    Args:
        user_content: 输入描述的文本或markdown，生成速读内容
    Returns:
        生成的速读内容
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用系统提示生成速读内容
    result = await call_chatgpt(
        processed_content, 
        PROMPTS["DOC_SYSTEM_PROMPT"], 
        openai_client,
        pattern_tags=["quick_read"]
    )
    
    return {"quick_read": result.get("quick_read", "")}

@mcp.tool()
async def generate_mind_map(user_content: Union[List, str]) -> dict:
    """
    Name:
        思维导图生成
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的思维导图
    Args:
        user_content: 输入描述的文本或markdown，生成思维导图
    Returns:
        生成的思维导图内容
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用系统提示生成思维导图内容
    result = await call_chatgpt(
        processed_content, 
        PROMPTS["DOC_SYSTEM_PROMPT"], 
        openai_client,
        pattern_tags=["mind_map"]
    )
    
    return {"mind_map": result.get("mind_map", "")}

@mcp.tool()
async def generate_deep_read(user_content: Union[List, str]) -> dict:
    """
    Name:
        深度阅读内容生成
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的深度阅读内容
    Args:
        user_content: 输入描述的文本或markdown，生成深度阅读内容
    Returns:
        生成的深度阅读内容
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 如果输入是字符串，转换为适当格式
    if isinstance(user_content, str):
        processed_content = [{"type": "text", "text": user_content}]
    else:
        processed_content = user_content
    
    # 使用系统提示生成深度阅读内容
    deep_read_content = await call_chatgpt(
        processed_content, 
        PROMPTS["DEEP_READ_SYSTEM_PROMPT"], 
        openai_client
    )
    
    return {"deep_read": deep_read_content}

@mcp.tool()
async def extract_release_date(content: Union[List, str]) -> dict:
    """
    Name:
        提取发布日期
    Description:
        从文本内容中提取研报或论文的发布日期
    Args:
        content: 输入文本内容
    Returns:
        提取的发布日期
    """
    # 初始化 OpenAI 客户端
    openai_client = get_openai_client()
    
    # 处理输入内容
    if isinstance(content, list):
        content = extract_json_to_str(content)
    
    # 使用系统提示提取发布日期
    result = await call_chatgpt(
        content, 
        PROMPTS["RELEASE_DATE_SYSTEM_PROMPT"], 
        openai_client,
        pattern_tags=["release_date"]
    )
    
    release_date = result.get("release_date", "")
    normalized_date = normalize_datetime_string(release_date)
    
    return {"release_date": normalized_date or release_date}

@mcp.tool()
async def generate_pdf(user_content: Union[List, str]) -> dict:
    """
    Name:
        生成PDF
    Description:
        根据描述的文本或者markdown，执行生成任务。返回生成的PDF的内容
    Args:
        user_content: 输入描述的文本或markdown，生成PDF
    Returns:
        生成的PDF内容
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),  # 使用环境变量
        http_client=httpx.Client(proxy=os.getenv("HTTP_PROXY", "http://39.104.58.112:31701")),
    )
    messages = [{"role": "system", "content": "根据描述的文本或者markdown，执行生成任务。返回生成的PDF的内容"}]
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=messages,
        stream=False,
    )
    return response.choices[0].message.content
    


# 启动MCP服务器
if __name__ == "__main__":
    # 创建 BQMCPCloud 实例
    cloud = BQMCPCloud()
    # 运行服务器
    cloud.run(transport='stdio')