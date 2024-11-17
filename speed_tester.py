import asyncio
import time
import statistics
from typing import Dict, List
import aiohttp
import speedtest
import logging

logger = logging.getLogger(__name__)

try:
    import iperf3
    IPERF_AVAILABLE = True
except ImportError:
    IPERF_AVAILABLE = False
    logger.warning("iperf3 Python package not available. iPerf tests will be skipped.")

class ComprehensiveSpeedTest:
    def __init__(self):
        self.download_urls = [
            "https://speed.cloudflare.com/10mb",
            "https://speed.hetzner.de/100MB.bin",
            "https://proof.ovh.net/files/100Mb.dat"
        ]
        
        # Initialize iperf client settings
        self.iperf_servers = [
            {"host": "iperf.he.net", "port": 5201},
            {"host": "iperf.scottlinux.com", "port": 5201}
        ]
        self.iperf_enabled = IPERF_AVAILABLE

    async def _test_direct_download(self) -> Dict:
        speeds = []
        async with aiohttp.ClientSession() as session:
            for url in self.download_urls:
                try:
                    start_time = time.time()
                    async with session.get(url) as response:
                        size = 0
                        async for chunk in response.content.iter_chunked(8192):
                            size += len(chunk)
                        
                        duration = time.time() - start_time
                        speed_mbps = (size * 8) / (1000000 * duration)  # Convert to Mbps
                        speeds.append(speed_mbps)
                except Exception as e:
                    logger.error(f"Error testing {url}: {str(e)}")
        
        return {
            "avg_speed": statistics.mean(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "min_speed": min(speeds) if speeds else 0
        }

    def _test_iperf(self) -> Dict:
        if not self.iperf_enabled:
            logger.info("iPerf testing disabled - package not available")
            return {}
            
        results = []
        for server in self.iperf_servers:
            try:
                client = iperf3.Client()
                client.server_hostname = server["host"]
                client.port = server["port"]
                client.duration = 5
                result = client.run()
                if result:
                    results.append({
                        "sent_mbps": result.sent_Mbps,
                        "received_mbps": result.received_Mbps,
                        "jitter_ms": getattr(result, 'jitter_ms', 0),
                        "lost_packets": getattr(result, 'lost_packets', 0)
                    })
            except Exception as e:
                logger.error(f"iPerf3 error with {server['host']}: {str(e)}")
        
        return results[0] if results else {}

    def _test_speedtest(self) -> Dict:
        try:
            st = speedtest.Speedtest()
            st.get_best_server()
            
            download_speed = st.download() / 1_000_000  # Convert to Mbps
            upload_speed = st.upload() / 1_000_000  # Convert to Mbps
            ping_latency = st.results.ping
            
            return {
                "download": download_speed,
                "upload": upload_speed,
                "ping": ping_latency
            }
        except Exception as e:
            logger.error(f"Speedtest.net error: {str(e)}")
            return {}

    async def _test_latency(self) -> Dict:
        targets = ["8.8.8.8", "1.1.1.1", "208.67.222.222"]
        results = []
        
        for target in targets:
            try:
                ping_result = ping(target, count=5)
                results.append({
                    "target": target,
                    "min_ms": ping_result.rtt_min_ms,
                    "avg_ms": ping_result.rtt_avg_ms,
                    "max_ms": ping_result.rtt_max_ms,
                    "packet_loss": ping_result.packet_loss
                })
            except Exception as e:
                logger.error(f"Ping error to {target}: {str(e)}")
        
        return results

    async def run_comprehensive_test(self) -> Dict:
        try:
            # Run tests concurrently where possible
            direct_download = await self._test_direct_download()
            latency = await self._test_latency()
            
            # Run synchronous tests
            speedtest_results = self._test_speedtest()
            iperf_results = self._test_iperf()
            
            return {
                "timestamp": time.time(),
                "direct_download": direct_download,
                "speedtest": speedtest_results,
                "iperf": iperf_results,
                "latency": latency,
                "composite_speed": self._calculate_composite_speed(
                    direct_download, speedtest_results, iperf_results
                )
            }
        except Exception as e:
            logger.error(f"Comprehensive test error: {str(e)}")
            raise

    def _calculate_composite_speed(self, direct, speedtest, iperf) -> Dict:
        speeds = []
        
        if direct.get("avg_speed"):
            speeds.append(direct["avg_speed"])
        if speedtest.get("download"):
            speeds.append(speedtest["download"])
        if iperf.get("received_mbps"):
            speeds.append(iperf["received_mbps"])
        
        return {
            "download_mbps": statistics.mean(speeds) if speeds else 0,
            "reliability_score": len(speeds) / 3 * 100  # Percentage of successful tests
        } 