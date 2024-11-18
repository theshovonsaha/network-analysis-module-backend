# Network Analysis Module Backend

A powerful backend service for network analysis built with FastAPI and Python. This service integrates with multiple AI providers and offers network monitoring capabilities.

## Features

### AI Integration

- Support for multiple AI providers:
  - Mistral AI
  - Google Generative AI
  - Anthropic Claude
  - OpenAI

### Network Analysis

- Real-time network monitoring
- Interface statistics collection
- Network traffic analysis using Scapy
- System resource monitoring with psutil
- Network interface management via netifaces

## Tech Stack

- **FastAPI**: High-performance web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **Python-dotenv**: Environment variable management
- **aiohttp**: Async HTTP client/server framework
- **AI SDKs**: Multiple AI provider integrations
- **Network Tools**: psutil, netifaces, scapy

## Requirements

- Python 3.8 or higher

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/network-analysis-module-backend.git
cd network-analysis-module-backend
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:

```bash
touch .env
echo "MISTRAL_API_KEY=your_mistral_key" >> .env
echo "GOOGLE_API_KEY=your_google_key" >> .env
echo "ANTHROPIC_API_KEY=your_anthropic_key" >> .env
echo "OPENAI_API_KEY=your_openai_key" >> .env
```

## Running the Application

Start the development server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## Development

This project follows modern Python development practices:

- Type hints for better code quality
- Async/await for efficient I/O operations
- Modular design for easy maintenance
- Comprehensive API documentation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FastAPI for the excellent web framework
- All AI providers for their powerful APIs
- Network analysis tool maintainers

## Contact

Shovon Saha - [@theshovonsaha](https://www.theshovonsaha.com)
Project Link: [https://github.com/theshovonsaha/network-analysis-module-backend](https://github.com/theshovonsaha/network-analysis-module-backend)

I referenced the following code blocks for this README:

- [FastAPI Documentation](https://fastapi.tiangolo.com/tutorial/first-steps/)
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [Python Environment Variables](https://docs.python.org/3/library/os.html#os.getenv)
  fastapi
  uvicorn
  python-dotenv
  aiohttp
  mistralai
  google-generativeai
  anthropic
  openai
  psutil
  netifaces
  scapy

<p align="center">
  <a href="https://fastapi.tiangolo.com"><img src="https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png" alt="FastAPI"></a>
</p>
<p align="center">
    <em>FastAPI framework, high performance, easy to learn, fast to code, ready for production</em>
</p>
<p align="center">
<a href="https://github.com/tiangolo/fastapi/actions?query=workflow%3ATest+event%3Apush+branch%3Amaster" target="_blank">
    <img src="https://github.com/tiangolo/fastapi/workflows/Test/badge.svg?event=push&branch=master" alt="Test">
</a>
<a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/tiangolo/fastapi" target="_blank">
    <img src="https://coverage-badge.samuelcolvin.workers.dev/tiangolo/fastapi.svg" alt="Coverage">
</a>
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/v/fastapi?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://pypi.org/project/fastapi" target="_blank">
    <img src="https://img.shields.io/pypi/pyversions/fastapi.svg?color=%2334D058" alt="Supported Python versions">
</a>
</p>
