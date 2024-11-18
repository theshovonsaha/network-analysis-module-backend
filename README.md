# Network Analysis Module Backend

A sophisticated backend service for network analysis built with FastAPI and Python. This service provides comprehensive network testing capabilities with AI-powered analysis through multiple providers.

## Features

### Network Testing

- Multi-source speed testing using:
  - Direct downloads from multiple CDNs
  - Speedtest.net integration
  - iPerf3 testing (when available)
- Latency analysis with multiple targets
- Packet loss detection
- Jitter measurement
- Composite speed calculations

### AI Integration

- Support for multiple AI providers:
  - Mistral AI
  - Google Gemini
  - Anthropic Claude
  - OpenAI
- Customizable analysis prompts
- Real-time performance insights
- Gaming and streaming optimization recommendations

### System Monitoring

- CPU utilization tracking
- Memory usage analysis
- Network interface statistics
- Packet transmission metrics

## Tech Stack

- **FastAPI**: High-performance async web framework
- **Uvicorn**: Lightning-fast ASGI server
- **Python-dotenv**: Environment variable management
- **AI SDKs**: Multiple provider integrations
- **Network Tools**:
  - speedtest-cli: Internet speed testing
  - psutil: System metrics
  - iperf3: Advanced network testing (optional)
  - aiohttp: Async HTTP operations

## Requirements

- Python 3.8 or higher
- pip package manager
- (Optional) iPerf3 client

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

- Interactive API docs: `http://localhost:8000/docs`
- Alternative API docs: `http://localhost:8000/redoc`

## Deployment

The service is configured for deployment on Render:

- Auto-deployment on push to main branch
- Environment variable management
- CORS configuration for frontend integration
- Health check endpoints

## Future Improvements

### Network Analysis

- WebRTC-based peer-to-peer speed testing
- Network path analysis and traceroute visualization
- QoS (Quality of Service) measurement
- IPv6 performance metrics
- Historical data tracking and trending

### AI Integration

- Custom AI model training for network analysis
- Anomaly detection using machine learning
- Predictive performance analytics
- Network optimization recommendations
- Bandwidth usage forecasting

### System Features

- Real-time WebSocket updates
- Distributed testing nodes
- API rate limiting and caching
- Extended metrics collection
- Custom testing profiles

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Your Name - [@yourusername](https://twitter.com/yourusername)
Project Link: [https://github.com/yourusername/network-analysis-module-backend](https://github.com/yourusername/network-analysis-module-backend)

## Acknowledgments

- FastAPI for the excellent web framework
- AI providers for their powerful APIs
- Network analysis tool maintainers
- Open source community
