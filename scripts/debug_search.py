import sys
import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

client = httpx.Client(base_url="http://localhost:8000", timeout=20)
res = client.post("/api/search", json={"topic": "quantum machine learning"})
print("Search status:", res.status_code)
data = res.json()
papers = data.get("papers", [])
print(f"Total papers returned: {len(papers)}")
for idx, p in enumerate(papers):
    title = p.get("title", "")
    pid = p.get("id")
    ev = p.get("evidence_state")
    summary = p.get("summary", "")
    print(
        f"[{idx + 1}] ID: {pid} | Title: {title[:50]} | State: {ev} | Summary length: {len(summary)}"
    )
