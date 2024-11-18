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
        {
            "status": "🚀 Initializing speed test module...",
            "code": "import speedtest\nst = speedtest.Speedtest()\nprint('SpeedTest CLI Version 2.1.3 initialized')",
            "progress": 5
        },
        {
            "status": "🔍 Finding optimal server...",
            "code": """servers = st.get_servers()
best = st.get_best_server()
print(f'Selected Server: {best["host"]}')
print(f'Location: {best["country"]}, {best["cc"]}')
print(f'Sponsor: {best["sponsor"]}')
print(f'Latency: {best["latency"]:.2f} ms')""",
            "progress": 10
        },
        {
            "status": "🌐 Testing DNS resolution...",
            "code": """import dns.resolver
results = dns.resolver.resolve('google.com', 'A')
print('DNS Resolution Times:')
for rdata in results:
    print(f' → {rdata.address}: {dns.resolver.timings.get(rdata.address):.3f}ms')
print(f'Average DNS Response: {sum(dns.resolver.timings.values())/len(dns.resolver.timings):.2f}ms')""",
            "progress": 15
        },
        {
            "status": "📡 Testing download speed...",
            "code": """download = st.download()
speed_mbps = download / 1_000_000
print(f'Download Test Results:')
print(f'Raw Speed: {speed_mbps:.2f} Mbps')
print(f'Chunks received: {st.results.bytes_received:,} bytes')
print(f'Test Duration: {st.results.timestamp - st.results.starttime:.2f}s')""",
            "progress": 25
        },
        {
            "status": "📊 Analyzing packet sizes...",
            "code": """from scapy.all import sniff
packets = sniff(timeout=2)
sizes = [len(pkt) for pkt in packets]
print(f'Packet Analysis:')
print(f'Total Packets: {len(packets)}')
print(f'Avg Size: {sum(sizes)/len(sizes):.0f} bytes')
print(f'Max Size: {max(sizes)} bytes')
print(f'Min Size: {min(sizes)} bytes')""",
            "progress": 35
        },
        {
            "status": "🔍 Checking MTU optimization...",
            "code": """import subprocess
result = subprocess.run(['ping', '-f', '-l', '1472', '8.8.8.8'], capture_output=True)
print(f'MTU Test Results:')
print(f'Optimal MTU: {1472 if result.returncode == 0 else "1450"} bytes')
print(f'Fragmentation: {"Not Required" if result.returncode == 0 else "Required"}')""",
            "progress": 45
        },
        {
            "status": "📶 Measuring signal strength...",
            "code": """import wifi
networks = wifi.Cell.all('wlan0')
current = [n for n in networks if n.ssid == wifi.get_current_network()][0]
print(f'WiFi Signal Analysis:')
print(f'SSID: {current.ssid}')
print(f'Signal: {current.signal} dBm')
print(f'Quality: {current.quality}/70')
print(f'Frequency: {current.frequency} GHz')""",
            "progress": 55
        },
        {
            "status": "⚡ Running latency tests...",
            "code": """for server in ['8.8.8.8', '1.1.1.1', 'google.com']:
    result = ping(server, count=5)
    print(f'Ping results for {server}:')
    print(f' Min: {result.min_rtt:.2f}ms')
    print(f' Avg: {result.avg_rtt:.2f}ms')
    print(f' Max: {result.max_rtt:.2f}ms')
    print(f' Jitter: {result.jitter:.2f}ms')""",
            "progress": 65
        },
        {
            "status": "🔄 Testing TCP connection quality...",
            "code": """import socket
tcp_results = []
for port in [80, 443, 8080]:
    start = time.time()
    s = socket.create_connection(('google.com', port), timeout=2)
    tcp_results.append(time.time() - start)
    s.close()
print(f'TCP Connection Times:')
print(f' Port 80: {tcp_results[0]*1000:.2f}ms')
print(f' Port 443: {tcp_results[1]*1000:.2f}ms')
print(f' Port 8080: {tcp_results[2]*1000:.2f}ms')""",
            "progress": 75
        },
        {
            "status": "📤 Testing upload capabilities...",
            "code": """upload = st.upload()
speed_mbps = upload / 1_000_000
print(f'Upload Test Results:')
print(f'Raw Speed: {speed_mbps:.2f} Mbps')
print(f'Chunks sent: {st.results.bytes_sent:,} bytes')
print(f'Test Duration: {st.results.timestamp - st.results.starttime:.2f}s')""",
            "progress": 85
        },
        {
            "status": "📊 Analyzing network buffer...",
            "code": """import psutil
net_io = psutil.net_io_counters()
print(f'Network Buffer Statistics:')
print(f'Packets Received: {net_io.packets_recv:,}')
print(f'Packets Sent: {net_io.packets_sent:,}')
print(f'Errors In: {net_io.errin:,}')
print(f'Errors Out: {net_io.errout:,}')
print(f'Drops In: {net_io.dropin:,}')
print(f'Drops Out: {net_io.dropout:,}')""",
            "progress": 90
        },
        {
            "status": "🌐 Testing IPv6 connectivity...",
            "code": """import socket
ipv6_capable = socket.has_ipv6
if ipv6_capable:
    ipv6_addr = socket.getaddrinfo('google.com', None, socket.AF_INET6)
print(f'IPv6 Connectivity:')
print(f'IPv6 Capable: {"Yes" if ipv6_capable else "No"}')
print(f'IPv6 Addresses: {len(ipv6_addr) if ipv6_capable else 0}')
print(f'Native IPv6: {"Yes" if ipv6_capable and len(ipv6_addr) > 0 else "No"}')""",
            "progress": 92
        },
        {
            "status": "🧠 Running AI analysis...",
            "code": """analysis = await analyze_network_metrics(results)
print(f'AI Analysis Results:')
print(f'Network Rating: {analysis["rating"]}/10')
print(f'Stability Score: {analysis["stability_score"]}%')
print(f'Recommended Uses:')
for use in analysis["recommended_uses"]:
    print(f' → {use}')""",
            "progress": 95
        },
        {
            "status": "📈 Generating performance report...",
            "code": """report = {
    "download_speed": f"{speed_mbps:.2f} Mbps",
    "upload_speed": f"{upload_speed:.2f} Mbps",
    "latency": f"{ping:.2f}ms",
    "jitter": f"{jitter:.2f}ms",
    "packet_loss": f"{packet_loss:.2f}%",
    "mtu_size": optimal_mtu,
    "tcp_performance": tcp_results,
    "dns_response": dns_times,
    "ipv6_ready": ipv6_capable,
    "network_stability": stability_score
}
print(json.dumps(report, indent=2))""",
            "progress": 98
        },
        {
            "status": "✨ Finalizing results...",
            "code": """print('Speed Test Complete!')
print(f'Total Duration: {total_time:.2f}s')
print(f'Tests Completed: {len(status_updates)}')
print(f'Data Processed: {total_bytes_processed:,} bytes')
return json.dumps(results, indent=2)""",
            "progress": 100
        }
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
            
            prompt = f"""You are an advanced network performance analyst. Analyze these speed test results and provide insights in the following format:

# Connection Quality
Evaluate the overall connection quality based on the metrics:
Download: {download_speed:.1f} Mbps
Upload: {upload_speed:.1f} Mbps
Latency: {ping:.1f} ms

# Optimal Use Cases
List the best use cases for this connection.

# Limitations
Identify potential limitations and bottlenecks.

# Recommendations
Provide specific recommendations for improvement.

# Technical Insights
Network Stability: [Analysis of jitter and packet loss]
Bandwidth Utilization: [Analysis of speed metrics]
Latency Profile: [Analysis of ping and response times]
Performance Rating: [Overall rating out of 10]
"""

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