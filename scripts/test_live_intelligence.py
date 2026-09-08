import httpx
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    client = httpx.Client(base_url="http://localhost:8000", timeout=30)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print("[PASS] Health Check: OK")

    # 2. Login or register
    login_res = client.post("/api/auth/token", data={"username": "researcher@lumen.ai", "password": "Password123!"})
    if login_res.status_code != 200:
        reg = client.post("/api/auth/register", json={"email": "researcher@lumen.ai", "password": "Password123!"})
        token = reg.json()["access_token"]
    else:
        token = login_res.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    print("[PASS] Auth: OK")

    # 3. AI Status
    status_res = client.get("/api/settings/ai-status", headers=headers)
    assert status_res.status_code == 200, f"AI status failed: {status_res.text}"
    print("[PASS] AI Status:", status_res.json())

    # 4. Verify Key endpoint
    verify_res = client.post("/api/settings/verify-key", json={"api_key": "dummy-key-test", "provider": "gemini"}, headers=headers)
    assert verify_res.status_code == 200, f"Verify key failed: {verify_res.text}"
    print("[PASS] Verify Key Check:", verify_res.json())

    # 5. Research Tools Test Payload
    papers = [
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "title": "Attention Is All You Need",
            "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer based solely on attention mechanisms.",
            "authors": "Vaswani et al.",
            "year": 2017,
            "venue": "NeurIPS",
            "citation_count": 95000,
        },
        {
            "id": "00000000-0000-0000-0000-000000000002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "summary": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
            "authors": "Devlin et al.",
            "year": 2018,
            "venue": "NAACL",
            "citation_count": 80000,
        }
    ]

    # 6. Compare
    comp = client.post("/api/compare", json={"topic": "Transformers", "papers": papers}, headers=headers)
    assert comp.status_code == 200, f"Compare failed: {comp.text}"
    diff_text = comp.json().get("key_differences", "")
    assert "Compare the paper-specific" not in diff_text, "Found static mock string in compare!"
    print("[PASS] Compare Matrix: OK (Genuine Synthesis)")

    # 7. Trends
    trends = client.post("/api/trends", json={"topic": "Transformers", "papers": papers}, headers=headers)
    assert trends.status_code == 200, f"Trends failed: {trends.text}"
    trend_narrative = trends.json().get("trend_analysis", "")
    assert len(trend_narrative) > 20, "Missing trend analysis narrative"
    print("[PASS] Trends Trajectory: OK")

    # 8. Research Gaps
    gaps = client.post("/api/research-gaps", json={"topic": "Transformers", "papers": papers}, headers=headers)
    assert gaps.status_code == 200, f"Gaps failed: {gaps.text}"
    findings = gaps.json().get("evidence_based_findings", [])
    assert len(findings) > 0, "No evidence based findings returned"
    assert "Evaluation gap" not in str(findings[0]), "Found static mock string in gaps!"
    print("[PASS] Research Gaps: OK")

    # 9. Research Ideas
    ideas = client.post("/api/research-ideas", json={"topic": "Transformers", "papers": papers}, headers=headers)
    assert ideas.status_code == 200, f"Ideas failed: {ideas.text}"
    idea_list = ideas.json().get("ideas", [])
    assert len(idea_list) > 0, "No ideas returned"
    assert "Robust evaluation for" not in idea_list[0].get("title", ""), "Found static mock string in ideas!"
    print(f"[PASS] Research Ideas: OK (Generated {len(idea_list)} directions)")

    # 10. Proposals
    prop = client.post("/api/proposals", json={"topic": "Transformers", "papers": papers}, headers=headers)
    assert prop.status_code == 200, f"Proposals failed: {prop.text}"
    prop_data = prop.json()
    assert "A systematic study of" not in prop_data.get("title", ""), "Found static mock string in proposal title!"
    assert len(prop_data.get("specific_hypotheses", [])) > 0, "Missing specific hypotheses in proposal"
    print(f"[PASS] Research Proposal: OK ('{prop_data.get('title')}')")

    print("\n*** ALL 10 LIVE INTELLIGENCE TESTS PASSED SUCCESSFULLY! ***")

if __name__ == "__main__":
    main()
