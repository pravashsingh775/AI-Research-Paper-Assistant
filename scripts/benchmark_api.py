"""Performance, Latency and Concurrency Benchmark Runner for Lumen Research API.

Measures:
- Latency distribution: min, p50, p90, p95, p99, max (ms)
- Throughput (requests/second)
- Concurrency scaling: 10, 25, 50 parallel workers
- Error rates and HTTP status code distribution
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from typing import Any
from uuid import uuid4

import httpx

BASE_URL = "http://localhost:8000"


async def benchmark_endpoint(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
    total_requests: int = 100,
    concurrency: int = 10,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    status_counts: dict[int, int] = {}
    errors: int = 0

    async def single_request() -> None:
        nonlocal errors
        async with semaphore:
            t0 = time.perf_counter()
            try:
                if method.upper() == "GET":
                    resp = await client.get(path, headers=headers)
                elif method.upper() == "POST":
                    resp = await client.post(path, headers=headers, json=json_data)
                else:
                    resp = await client.request(method, path, headers=headers, json=json_data)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                latencies.append(elapsed_ms)
                status_counts[resp.status_code] = status_counts.get(resp.status_code, 0) + 1
            except Exception:
                errors += 1

    start_wall = time.perf_counter()
    tasks = [asyncio.create_task(single_request()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    total_wall = time.perf_counter() - start_wall

    if not latencies:
        return {"error": "All requests failed"}

    latencies.sort()
    n = len(latencies)

    def percentile(p: float) -> float:
        idx = int(p * (n - 1))
        return latencies[idx]

    return {
        "endpoint": f"{method.upper()} {path}",
        "total_requests": total_requests,
        "concurrency": concurrency,
        "completed": len(latencies),
        "errors": errors,
        "success_rate": f"{(len(latencies) / total_requests) * 100:.1f}%",
        "duration_seconds": round(total_wall, 3),
        "throughput_rps": round(len(latencies) / total_wall, 1),
        "latency_min_ms": round(latencies[0], 2),
        "latency_p50_ms": round(percentile(0.50), 2),
        "latency_p90_ms": round(percentile(0.90), 2),
        "latency_p95_ms": round(percentile(0.95), 2),
        "latency_p99_ms": round(percentile(0.99), 2),
        "latency_max_ms": round(latencies[-1], 2),
        "status_distribution": status_counts,
    }


async def main() -> None:
    print("=" * 70)
    print("  LUMEN RESEARCH — API LATENCY & CONCURRENCY BENCHMARK SUITE")
    print("=" * 70)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        # Register benchmark user
        email = f"benchmark_{uuid4().hex[:8]}@example.com"
        reg_resp = await client.post(
            "/api/auth/register", json={"email": email, "password": "BenchPassword123!"}
        )
        token = reg_resp.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}

        benchmarks = [
            # 1. Health liveness
            ("GET", "/liveness", None, None, 100, 10),
            # 2. Deep readiness (DB + Redis + Storage)
            ("GET", "/readiness", None, None, 100, 10),
            # 3. Authenticated Auth Me
            ("GET", "/api/auth/me", auth_headers, None, 100, 10),
            # 4. Search endpoint (cached / fast path)
            ("POST", "/api/search", None, {"topic": "machine learning quantum"}, 50, 10),
            # 5. Concurrency stress test: 25 workers
            ("GET", "/liveness", None, None, 200, 25),
            # 6. Concurrency stress test: 50 workers
            ("GET", "/liveness", None, None, 250, 50),
            # 7. Deep readiness under 25 concurrency
            ("GET", "/readiness", None, None, 100, 25),
        ]

        results = []
        for method, path, headers, json_data, total, conc in benchmarks:
            print(f"\n[*] Benchmarking {method} {path} (Requests: {total}, Concurrency: {conc})...")
            res = await benchmark_endpoint(
                client=client,
                method=method,
                path=path,
                headers=headers,
                json_data=json_data,
                total_requests=total,
                concurrency=conc,
            )
            results.append(res)
            print(
                f"    RPS: {res['throughput_rps']} req/s | p50: {res['latency_p50_ms']}ms | p95: {res['latency_p95_ms']}ms | p99: {res['latency_p99_ms']}ms | Success: {res['success_rate']}"
            )

        # Save summary
        output_file = "docs/API_BENCHMARK_RESULTS.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Benchmark results written to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
