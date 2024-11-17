from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import speedtest
from typing import Dict, Optional
import os
from dotenv import load_dotenv
import logging
from speed_tester import ComprehensiveSpeedTest
import asyncio
import json
import google.generativeai as genai
from anthropic import Anthropic
import openai
import psutil  # for system metrics
import netifaces  # for network interface details
import scapy.all as scapy  # for packet analysis

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
        "https://your-custom-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get Mistral API key
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    logger.error("MISTRAL_API_KEY not found in environment variables")
    raise ValueError("MISTRAL_API_KEY not found in environment variables")

# Initialize Mistral AI client
try:
    mistral_client = MistralClient(
        api_key=mistral_api_key
    )
except Exception as e:
    logger.error(f"Failed to initialize Mistral client: {str(e)}")
    raise

class AIProvider:
    @staticmethod
    async def analyze_with_mistral(api_key: str, prompt: str) -> str:
        client = MistralClient(api_key=api_key)
        response = client.chat(
            model="mistral-tiny",
            messages=[ChatMessage(role="user", content=prompt)],
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
        {"status": "Initializing network probe...", "code": "sudo nmap -sS -p- target_network"},
        {"status": "Scanning network topology...", "code": "traceroute -I optimal_server"},
        {"status": "Analyzing bandwidth capacity...", "code": "netstat -s | grep segments"},
        {"status": "Testing downstream channels...", "code": "tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'"},
        {"status": "Measuring upstream throughput...", "code": "iperf3 -c speedtest.net -P 10"},
        {"status": "Calculating network latency...", "code": "ping -c 5 -i 0.2 gateway"},
        {"status": "Verifying connection stability...", "code": "curl -w '%{time_total}' -s speedtest/download"},
        {"status": "Processing final results...", "code": "cat /proc/net/dev | grep eth0"},
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

@app.get("/speedtest")
async def run_speedtest(
    x_api_key: Optional[str] = Header(None),
    x_provider: Optional[str] = Header(None)
) -> Dict:
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

            NETWORK METRICS:
            • Download Speed: {download_speed:.1f} Mbps
            • Upload Speed: {upload_speed:.1f} Mbps
            • Latency: {ping:.1f} ms
            
            SYSTEM CONTEXT:
            • CPU Usage: {cpu_percent}%
            • Memory Usage: {memory.percent}%
            • Packets Sent: {network_io.packets_sent:,}
            • Packets Received: {network_io.packets_recv:,}
            • Packet Loss: {network_io.dropin + network_io.dropout:,} packets

            PERFORMANCE BENCHMARKS:
            • Gaming: <50ms latency, 25+ Mbps down
            • 4K Streaming: 25+ Mbps down
            • Video Conferencing: 3+ Mbps up/down
            • Cloud Gaming: <30ms latency, 35+ Mbps down
            • Large File Downloads: 100+ Mbps down
            • Smart Home: 15+ Mbps down
            • Remote Work: 10+ Mbps up/down, <100ms latency

            Provide a detailed analysis in the following format:

            # Connection Quality
            [Evaluate overall network performance]

            # Optimal Use Cases
            • [List suitable activities]
            • [Include specific examples]

            # Limitations
            • [List any bottlenecks]
            • [Include system constraints]

            # Recommendations
            • [Provide actionable improvements]
            • [Include optimization tips]

            # Technical Insights
            • [Analyze packet patterns]
            • [Note any anomalies]

            Keep the analysis technical but user-friendly."""

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
        },
        "smart_devices": {
            "status": "Poor" if download_speed < 5 else
                     "Good" if download_speed >= 15 else "Fair",
            "details": "Smart home devices work best with 15+ Mbps"
        },
        "downloads": {
            "status": "Poor" if download_speed < 50 else
                     "Good" if download_speed >= 100 else "Fair",
            "details": "Fast downloads need 100+ Mbps for large files"
        }
    }
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)