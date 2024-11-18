from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mistralai.client import MistralClient
import speedtest
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import logging
import asyncio
import json
import google.generativeai as genai
from anthropic import Anthropic
import openai
import psutil  # for system metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.netlify.app",
        "https://rawspeedtest.netlify.app",
        "https://your-custom-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize API keys
mistral_api_key = os.getenv("MISTRAL_API_KEY", "testmistral")
google_api_key = os.getenv("GOOGLE_API_KEY", "your-google-api-key")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key")
openai_api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# Initialize clients with default error handling
try:
    if mistral_api_key:
        mistral_client = MistralClient(api_key=mistral_api_key)
    if google_api_key:
        genai.configure(api_key=google_api_key)
    if anthropic_api_key:
        anthropic_client = Anthropic(api_key=anthropic_api_key)
    if openai_api_key:
        openai.api_key = openai_api_key
except Exception as e:
    logger.error(f"Failed to initialize AI clients: {str(e)}")

class AIProvider:
    @staticmethod
    async def analyze_with_mistral(api_key: str, prompt: str) -> str:
        client = MistralClient(api_key=api_key)
        messages = [{"role": "user", "content": prompt}]
        response = client.chat(
            model="mistral-tiny",
            messages=messages,
            temperature=0.7,
            max_tokens=250
        )
        return response.choices[0].message.content

    @staticmethod
    async def analyze_with_claude(api_key: str, prompt: str) -> str:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    @staticmethod
    async def analyze_with_openai(api_key: str, prompt: str) -> str:
        from openai import AsyncOpenAI  # Import AsyncOpenAI for async support
        
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250
        )
        return response.choices[0].message.content

    @staticmethod
    async def analyze_with_gemini(api_key: str, prompt: str) -> str:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text

async def speed_test_generator():
    status_updates = [
        {"status": "Initializing speed test...", "progress": 10},
        {"status": "Testing download speed...", "progress": 30},
        {"status": "Testing upload speed...", "progress": 60},
        {"status": "Measuring latency...", "progress": 80},
        {"status": "Finalizing results...", "progress": 90},
    ]
    
    for status in status_updates:
        yield f"data: {json.dumps(status)}\n\n"
        await asyncio.sleep(1)

@app.get("/speedtest/status")
async def speedtest_status():
    return StreamingResponse(
        speed_test_generator(),
        media_type="text/event-stream"
    )

@app.get("/")
async def root():
    return {"message": "Network Analysis API is running"}

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy"}

@app.get("/speedtest")
async def run_speedtest(
    x_api_key: Optional[str] = Header(None),
    x_provider: Optional[str] = Header(None)
) -> Dict:
    # Validate provider and API key
    if x_provider and x_provider.lower() not in provider_map:
        raise HTTPException(status_code=400, detail="Invalid AI provider specified")
    
    if x_provider and not x_api_key:
        raise HTTPException(status_code=400, detail="API key required for AI analysis")
    
    # Use dummy keys for testing if not provided
    if not x_api_key:
        dummy_keys = {
            'mistral': 'testmistral',
            'google': 'your-google-api-key',
            'anthropic': 'your-anthropic-api-key',
            'openai': 'your-openai-api-key'
        }
        x_api_key = dummy_keys.get(x_provider.lower()) if x_provider else None
    
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        
        download_speed = st.download() / 1_000_000
        upload_speed = st.upload() / 1_000_000
        ping = st.results.ping

        result = {
            "download_speed": download_speed,
            "upload_speed": upload_speed,
            "ping_ms": ping,
            "network_analysis": analyze_network_performance(download_speed, upload_speed, ping)
        }

        # Only attempt AI analysis if both provider and API key are provided
        if x_provider and x_api_key:
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            network_io = psutil.net_io_counters()
            
            prompt = f"""You are an advanced network performance analyst. Analyze these speed test results:
            Download Speed: {download_speed:.1f} Mbps
            Upload Speed: {upload_speed:.1f} Mbps
            Latency: {ping:.1f} ms"""

            try:
                analyze_func = provider_map.get(x_provider.lower())
                if analyze_func:
                    result["ai_analysis"] = await analyze_func(x_api_key, prompt)
                    result["system_metrics"] = {
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory.percent,
                        "network_stats": {
                            "packets_sent": network_io.packets_sent,
                            "packets_received": network_io.packets_recv,
                            "packet_loss": network_io.dropin + network_io.dropout
                        }
                    }
            except Exception as e:
                logger.error(f"AI analysis failed: {str(e)}")
                pass

        return result
    except Exception as e:
        logger.error(f"Speed test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add provider mapping
provider_map = {
    'mistral': AIProvider.analyze_with_mistral,
    'claude': AIProvider.analyze_with_claude,
    'openai': AIProvider.analyze_with_openai,
    'gemini': AIProvider.analyze_with_gemini
}

def analyze_network_performance(download_speed: float, upload_speed: float, ping: float) -> dict:
    analysis = {
        "gaming": {
            "status": "Poor" if ping > 100 or download_speed < 15 else
                     "Good" if ping < 50 and download_speed >= 25 else "Fair",
            "details": "Gaming requires low latency (<50ms) and stable download speeds (>25 Mbps)"
        },
        "streaming": {
            "status": "Poor" if download_speed < 10 else
                     "Good" if download_speed >= 25 else "Fair",
            "details": "4K streaming needs 25+ Mbps, HD needs 5-10 Mbps"
        },
        "video_calls": {
            "status": "Poor" if download_speed < 1.5 or upload_speed < 1.5 else
                     "Good" if download_speed >= 3.0 and upload_speed >= 3.0 else "Fair",
            "details": "Video calls need balanced upload/download (3+ Mbps each)"
        }
    }
    return analysis

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )