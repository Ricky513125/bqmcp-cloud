# BQMCP Cloud

A cloud service for document processing and AI content generation, built on top of FastMCP.

## Features

- PDF document processing and content extraction
- AI-powered content generation:
  - PPT generation
  - Title generation
  - Abstract generation
  - Quick read content
  - Mind map generation
  - Deep reading content
  - Release date extraction
  - PDF generation

## Installation

```bash
pip install bqmcp-cloud
```

## Usage

### As a Python Package

```python
from bqmcp_cloud import BQMCPCloud

# Initialize the cloud service
cloud = BQMCPCloud(
    api_key="your-openai-api-key",  # or set OPENAI_API_KEY environment variable
    proxy="your-proxy-url",         # or set HTTP_PROXY environment variable
    base_output_path="outputs",     # optional
    log_level="INFO",              # optional
    mcp_name="bigquant"            # optional
)

# Run the server
cloud.run(transport='stdio')  # or 'http'
```

### Command Line Interface

```bash
# Basic usage
bqmcp-cloud

# With custom configuration
bqmcp-cloud --api-key "your-key" --proxy "your-proxy" --output-dir "custom-outputs" --log-level DEBUG
```

## Configuration

The service can be configured through:

1. Environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `HTTP_PROXY`: HTTP proxy URL

2. Constructor parameters:
   - `api_key`: OpenAI API key
   - `proxy`: HTTP proxy URL
   - `base_output_path`: Base directory for output files
   - `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - `mcp_name`: Name for the MCP server

3. Command line arguments:
   - `--api-key`: OpenAI API key
   - `--proxy`: HTTP proxy URL
   - `--output-dir`: Base output directory
   - `--log-level`: Logging level
   - `--name`: MCP server name
   - `--transport`: Transport type (stdio or http)

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bqmcp-cloud.git
cd bqmcp-cloud
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Run tests:
```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
