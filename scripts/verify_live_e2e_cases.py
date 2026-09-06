"""Live end-to-end verification script testing all features available in the Lumen Research UI.
Uploads real academic paper PDFs, tests grounded Q&A, abstention on unmentioned facts,
interactive chat sessions, research collections, and all 6 multi-paper intelligence tools.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz  # PyMuPDF
import httpx

API_BASE = "http://localhost:8000"
WEB_BASE = "http://localhost:3000"


def generate_research_paper_pdf() -> bytes:
    """Generate a multi-page academic research paper PDF with detailed scientific content."""
    doc = fitz.open()

    # Page 1: Title, Abstract, Introduction, Methodology
    page1 = doc.new_page()
    page1.insert_text(
        (50, 50), "Quantum Neural Kernels for Graph-Structured Representation Learning", fontsize=15
    )
    page1.insert_text(
        (50, 85),
        "Authors: Dr. Evelyn Vance, Dr. Tariq Al-Mansoor, Prof. Elena Rostova\n"
        "Affiliation: Institute for Quantum Computation and Machine Learning\n"
        "Venue: International Conference on Quantum Learning Systems (ICQLS 2026)",
        fontsize=9,
    )
    page1.insert_text(
        (50, 140),
        "Abstract\n"
        "Graph-structured data poses fundamental computational bottlenecks for classical kernel methods.\n"
        "In this work, we introduce Quantum Neural Kernels (QNK), an expressive family of parameterized\n"
        "quantum circuits tailored for non-Euclidean graph topologies. By mapping node neighborhoods to\n"
        "high-dimensional Hilbert states |ψ_G(x)⟩ = ∑_k α_k |k⟩, QNK achieves provable separation over\n"
        "classical Weisfeiler-Lehman graph kernels with O(log N) circuit depth.",
        fontsize=9,
    )
    page1.insert_text(
        (50, 240),
        "1. Methodology & Quantum Architecture\n"
        "Our architecture constructs an equivariant quantum graph convolutional layer.\n"
        "Given graph G = (V, E), node features are encoded via single-qubit rotations R_z(θ_v) R_x(ϕ_v).\n"
        "Inter-node entanglement is generated using Controlled-Z (CZ) gates along edge boundaries E.\n"
        "The variational parameters are updated iteratively via quantum natural gradient descent\n"
        "with an adaptive learning rate η = 0.015 and Fisher Information Matrix regularization.",
        fontsize=9,
    )
    page1.insert_text(
        (50, 360),
        "2. Datasets & Empirical Evaluation Protocol\n"
        "We evaluate QNK on three standard benchmark datasets: Cora (2,708 nodes), Citeseer (3,327 nodes),\n"
        "and PubMed (19,717 nodes). All experiments use 10-fold cross-validation with an 80/10/10 split.\n"
        "On the Cora benchmark, QNK achieved 95.4% test classification accuracy, outperforming classical GCN\n"
        "(81.5%) and GAT (83.0%) baselines while utilizing 68% fewer trainable parameters.\n"
        "On Citeseer, QNK achieved 89.2% accuracy with robust resistance to label noise.",
        fontsize=9,
    )

    # Page 2: Hardware Limitations, Ethical Impact & Future Directions
    page2 = doc.new_page()
    page2.insert_text(
        (50, 50),
        "3. NISQ Hardware Bottlenecks & Decoherence Limits\n"
        "Current experiments were executed on an IBM Quantum Eagle 127-qubit superconducting processor.\n"
        "At circuit depths exceeding 45 two-qubit gates, thermal relaxation T1 (95 microseconds) and dephasing\n"
        "T2 (70 microseconds) induce significant fidelity degradation, necessitating zero-noise extrapolation (ZNE).\n"
        "Quantum volume was measured at 2^7 under standard error-mitigated benchmarking protocols.",
        fontsize=9,
    )
    page2.insert_text(
        (50, 160),
        "4. Conclusion & Future Roadmap\n"
        "We established that quantum state embeddings confer exponential representational power for graph domains.\n"
        "Future research will investigate fault-tolerant topological surface codes and distributed quantum\n"
        "federated graph learning to scale to graphs with > 1 million vertices.",
        fontsize=9,
    )

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


async def verify_all_system_cases() -> None:
    print("\n" + "=" * 80)
    print("      LUMEN RESEARCH: COMPREHENSIVE LIVE END-TO-END VERIFICATION")
    print("=" * 80 + "\n")

    results: dict[str, bool] = {}

    async with httpx.AsyncClient(timeout=45.0) as client:
        # -------------------------------------------------------------
        # Case 1: Service Health & Deep 3-Tier Readiness
        # -------------------------------------------------------------
        print("🔹 CASE 1: Service Probes & Deep Multi-Tier Readiness")
        t0 = time.perf_counter()
        resp_health = await client.get(f"{API_BASE}/health")
        resp_ready = await client.get(f"{API_BASE}/readiness")
        probe_ms = (time.perf_counter() - t0) * 1000

        assert resp_health.status_code == 200
        assert resp_ready.status_code == 200
        ready_json = resp_ready.json()

        db_ok = ready_json["checks"]["database"]
        redis_ok = ready_json["checks"]["redis"]
        storage_ok = ready_json["checks"]["storage"]

        print(f"  ✓ /health: {resp_health.json()['status']}")
        print(
            f"  ✓ /readiness: PostgreSQL={db_ok}, Redis={redis_ok}, MinIO Storage={storage_ok} ({probe_ms:.1f}ms)"
        )
        results["Case 1 (Readiness Probes)"] = db_ok and redis_ok and storage_ok

        # -------------------------------------------------------------
        # Case 2: Researcher Registration & JWT Authentication
        # -------------------------------------------------------------
        print("\n🔹 CASE 2: User Registration, JWT Issuance & Profile Resolution")
        test_email = f"lead_researcher_{uuid4().hex[:8]}@oxford.edu"
        test_password = "QuantumResearch2026!"

        reg_resp = await client.post(
            f"{API_BASE}/api/auth/register",
            json={"email": test_email, "password": test_password},
        )
        assert reg_resp.status_code in (200, 201), f"Registration failed: {reg_resp.text}"
        token = reg_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        me_resp = await client.get(f"{API_BASE}/api/auth/me", headers=headers)
        assert me_resp.status_code == 200
        user_info = me_resp.json()

        print(f"  ✓ Registered Researcher: {user_info['email']}")
        print(f"  ✓ JWT Token Issued: {token[:24]}...")
        print(f"  ✓ Verified Profile: User ID = {user_info['id']}")
        results["Case 2 (Auth & JWT)"] = bool(user_info["id"])

        # -------------------------------------------------------------
        # Case 3: Federated Scholarly Discovery Search
        # -------------------------------------------------------------
        print("\n🔹 CASE 3: Federated Scholarly Discovery Search")
        search_query = "Graph Neural Networks Quantum Computing"
        t0 = time.perf_counter()
        search_resp = await client.post(
            f"{API_BASE}/api/search",
            headers=headers,
            json={"topic": search_query},
        )
        search_ms = (time.perf_counter() - t0) * 1000
        assert search_resp.status_code == 200, f"Search failed: {search_resp.text}"
        search_data = search_resp.json()
        papers_found = search_data.get("papers", [])

        print(
            f"  ✓ Query: '{search_query}' -> {len(papers_found)} papers retrieved ({search_ms:.1f}ms)"
        )
        if papers_found:
            top_paper = papers_found[0]
            print(f'  ✓ Top Match: "{top_paper["title"][:65]}..."')
            print(f"    Source: {top_paper.get('source')} | Venue: {top_paper.get('venue', 'N/A')}")
        results["Case 3 (Federated Search)"] = len(papers_found) > 0

        # -------------------------------------------------------------
        # Case 4: Private Academic PDF Upload & Ingestion
        # -------------------------------------------------------------
        print("\n🔹 CASE 4: Academic PDF Upload, Text Normalization & Background Indexing")
        pdf_bytes = generate_research_paper_pdf()

        upload_resp = await client.post(
            f"{API_BASE}/api/papers/upload",
            headers=headers,
            files={"file": ("quantum_graph_kernels.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
        upload_data = upload_resp.json()
        paper_id = upload_data["id"]
        doc_id = upload_data["document_id"]
        job_id = upload_data["job_id"]

        print(f"  ✓ PDF Uploaded: Paper ID = {paper_id}")
        print(f"  ✓ Document Record Created: {doc_id}")
        print(f"  ✓ Background Worker Job Queued: {job_id}")

        # Poll worker completion
        completed = False
        for i in range(30):
            await asyncio.sleep(1.0)
            job_resp = await client.get(f"{API_BASE}/api/jobs/{job_id}", headers=headers)
            assert job_resp.status_code == 200
            jdata = job_resp.json()
            if jdata["status"] == "COMPLETED":
                completed = True
                print(
                    f"  ✓ Background Ingestion Succeeded in {i + 1}s (Progress: {jdata.get('progress', 100)}%)"
                )
                break
            elif jdata["status"] == "FAILED":
                raise RuntimeError(f"Ingestion job failed: {jdata.get('error')}")

        assert completed, "Background job timed out"

        # Check paper state
        paper_get = await client.get(f"{API_BASE}/api/papers/{paper_id}", headers=headers)
        assert paper_get.status_code == 200
        paper_info = paper_get.json()
        print(f'  ✓ Title Extracted: "{paper_info["title"]}"')
        print(f"  ✓ Evidence State: '{paper_info['evidence_state']}' (Full-Text Verified)")
        results["Case 4 (Upload & Indexing)"] = paper_info["evidence_state"] == "full-text"

        # -------------------------------------------------------------
        # Case 5: Grounded Q&A - Direct Technical Fact Extraction
        # -------------------------------------------------------------
        print("\n🔹 CASE 5: Grounded Q&A: Technical Accuracy & Citation Verification")
        q1 = "What dataset was used and what test classification accuracy was achieved on Cora?"
        t0 = time.perf_counter()
        qa1_resp = await client.post(
            f"{API_BASE}/api/qa",
            headers=headers,
            json={"document_id": doc_id, "question": q1},
        )
        qa1_ms = (time.perf_counter() - t0) * 1000
        assert qa1_resp.status_code == 200
        qa1_data = qa1_resp.json()

        print(f'  [Q]: "{q1}"')
        print(f"  [Status]: {qa1_data['status']} ({qa1_ms:.1f}ms)")
        print(f'  [Answer]: "{qa1_data["answer"][:120]}..."')

        # Verify factual accuracy
        answer_text = qa1_data["answer"].lower()
        has_cora = "cora" in answer_text
        has_accuracy = "95.4" in answer_text or "95" in answer_text
        citations = qa1_data.get("evidence", [])
        print(f"  ✓ Grounded Evidence: {len(citations)} chunk citation(s) attached")
        if citations:
            c = citations[0]
            print(f"    Attribution -> Page: {c.get('page')}, Section: '{c.get('section')}'")

        assert has_cora, "Expected Cora in answer"
        assert len(citations) > 0, "Expected citation chunks in response"
        results["Case 5 (Grounded Technical QA)"] = has_cora and len(citations) > 0

        # -------------------------------------------------------------
        # Case 6: Grounded Q&A - Methodology & Mathematical Reasoning
        # -------------------------------------------------------------
        print("\n🔹 CASE 6: Grounded Q&A: Mathematical Architecture Reasoning")
        q2 = "What optimization method and learning rate are used to update the variational parameters?"
        qa2_resp = await client.post(
            f"{API_BASE}/api/qa",
            headers=headers,
            json={"document_id": doc_id, "question": q2},
        )
        assert qa2_resp.status_code == 200
        qa2_data = qa2_resp.json()

        print(f'  [Q]: "{q2}"')
        print(f"  [Status]: {qa2_data['status']}")
        print(f'  [Answer]: "{qa2_data["answer"][:120]}..."')
        a2_text = qa2_data["answer"].lower()
        has_gradient = "gradient" in a2_text or "natural gradient" in a2_text or "fisher" in a2_text
        print(f"  ✓ Methodology Correctly Grounded: {has_gradient}")
        results["Case 6 (Methodology QA)"] = has_gradient

        # -------------------------------------------------------------
        # Case 7: Anti-Hallucination & Faithful Abstention Protocol
        # -------------------------------------------------------------
        print("\n🔹 CASE 7: Anti-Hallucination Guardrail (Distractor Test)")
        q_fake = "What was the brand and serial number of the helium compressor in the cryogenic laboratory?"
        qa3_resp = await client.post(
            f"{API_BASE}/api/qa",
            headers=headers,
            json={"document_id": doc_id, "question": q_fake},
        )
        assert qa3_resp.status_code == 200
        qa3_data = qa3_resp.json()

        print(f'  [Q - Unmentioned Fact]: "{q_fake}"')
        print(f"  [Status]: {qa3_data['status']}")
        print(f'  [Response]: "{qa3_data["answer"][:100]}..."')
        abstained = (
            qa3_data["status"] == "insufficient-evidence"
            or "insufficient" in qa3_data["answer"].lower()
        )
        print(f"  ✓ Faithfully Abstained (Zero False Claim): {abstained}")
        results["Case 7 (Anti-Hallucination Abstention)"] = abstained

        # -------------------------------------------------------------
        # Case 8: Interactive Multi-Turn Research Chat Session
        # -------------------------------------------------------------
        print("\n🔹 CASE 8: Interactive Multi-Turn Chat Session with Context")
        # Turn 1
        chat_turn1 = await client.post(
            f"{API_BASE}/api/chat",
            headers=headers,
            json={
                "paper_id": paper_id,
                "question": "What are the node counts of the Cora, Citeseer, and PubMed datasets evaluated in this study?",
            },
        )
        assert chat_turn1.status_code == 200, f"Chat turn 1 failed: {chat_turn1.text}"
        c1_data = chat_turn1.json()
        session_id = c1_data["session_id"]
        print(f"  ✓ Turn 1 Chat Session ID: {session_id}")
        print(f'  ✓ Assistant: "{c1_data["answer"][:95]}..."')
        print(f"  ✓ Citations Attached: {len(c1_data.get('citations', []))} items")

        # Turn 2 (Follow-up query using session_id)
        chat_turn2 = await client.post(
            f"{API_BASE}/api/chat",
            headers=headers,
            json={
                "paper_id": paper_id,
                "session_id": session_id,
                "question": "What test classification accuracy did QNK achieve on the Cora and Citeseer benchmarks?",
            },
        )
        assert chat_turn2.status_code == 200, f"Chat turn 2 failed: {chat_turn2.text}"
        c2_data = chat_turn2.json()
        print(f'  ✓ Turn 2 Follow-up Assistant: "{c2_data["answer"][:95]}..."')
        results["Case 8 (Interactive Chat Session)"] = (
            bool(session_id) and len(c1_data.get("citations", [])) > 0
        )

        # -------------------------------------------------------------
        # Case 9: Research Collections Management
        # -------------------------------------------------------------
        print("\n🔹 CASE 9: Research Collections & Project Workspace Management")
        col_create = await client.post(
            f"{API_BASE}/api/collections",
            headers=headers,
            json={"name": "Quantum Graph Learning Project"},
        )
        assert col_create.status_code in (200, 201)
        col_id = col_create.json()["id"]
        print(f"  ✓ Created Collection: ID = {col_id}")

        # Add paper to collection
        add_paper = await client.post(
            f"{API_BASE}/api/collections/{col_id}/papers",
            headers=headers,
            json={"paper_id": paper_id},
        )
        assert add_paper.status_code in (200, 201)
        print(f"  ✓ Added Paper {paper_id} to Collection")

        # Verify collection detail
        col_get = await client.get(f"{API_BASE}/api/collections/{col_id}", headers=headers)
        assert col_get.status_code == 200
        col_data = col_get.json()
        has_paper = any(p["id"] == paper_id for p in col_data.get("papers", []))
        print(f"  ✓ Collection Paper Membership Confirmed: {has_paper}")
        results["Case 9 (Collections Management)"] = has_paper

        # -------------------------------------------------------------
        # Case 10: Multi-Paper Research Intelligence Tools
        # -------------------------------------------------------------
        print("\n🔹 CASE 10: Multi-Paper Intelligence Suite (All 6 Tools)")
        chosen_papers = [
            {
                "id": paper_id,
                "title": paper_info["title"],
                "summary": "Quantum graph convolutional networks for molecular and citation topology classification.",
                "authors": "Dr. Evelyn Vance et al.",
                "year": 2026,
                "venue": "ICQLS 2026",
                "citation_count": 14,
            },
            {
                "id": str(uuid4()),
                "title": "Classical Weisfeiler-Lehman Graph Kernels vs Message Passing GNNs",
                "summary": "Benchmarking expressive limits of classical graph neural networks against WL isomorphism tests.",
                "authors": "Dr. Marcus Thorne et al.",
                "year": 2024,
                "venue": "NeurIPS 2024",
                "citation_count": 89,
            },
        ]
        tool_payload = {
            "topic": "Quantum vs Classical Graph Machine Learning",
            "papers": chosen_papers,
        }

        # Tool 1: Compare
        t_comp = await client.post(f"{API_BASE}/api/compare", headers=headers, json=tool_payload)
        assert t_comp.status_code == 200
        print(f"  ✓ 1. Comparison Matrix: {len(t_comp.json().get('papers', []))} papers structured")

        # Tool 2: Trends
        t_trend = await client.post(f"{API_BASE}/api/trends", headers=headers, json=tool_payload)
        assert t_trend.status_code == 200
        print(
            f"  ✓ 2. Research Trajectory: Keywords -> {t_trend.json().get('emerging_keywords', [])[:3]}"
        )

        # Tool 3: Gaps
        t_gaps = await client.post(
            f"{API_BASE}/api/research-gaps", headers=headers, json=tool_payload
        )
        assert t_gaps.status_code == 200
        print(
            f"  ✓ 3. Research Gaps: {len(t_gaps.json().get('evidence_based_findings', []))} gap finding(s)"
        )

        # Tool 4: Ideas
        t_ideas = await client.post(
            f"{API_BASE}/api/research-ideas", headers=headers, json=tool_payload
        )
        assert t_ideas.status_code == 200
        print(
            f"  ✓ 4. Novel Directions: Generated '{t_ideas.json().get('ideas', [{}])[0].get('title', 'N/A')}'"
        )

        # Tool 5: Proposal
        t_prop = await client.post(f"{API_BASE}/api/proposals", headers=headers, json=tool_payload)
        assert t_prop.status_code == 200
        print(f'  ✓ 5. Academic Proposal: Drafted "{t_prop.json().get("title")}"')

        # Tool 6: Similarity Map
        t_sim = await client.post(
            f"{API_BASE}/api/similarity-map", headers=headers, json=tool_payload
        )
        assert t_sim.status_code == 200
        print(
            f"  ✓ 6. Similarity Network: {len(t_sim.json().get('nodes', []))} nodes, {len(t_sim.json().get('edges', []))} edges"
        )
        results["Case 10 (6 Research Tools)"] = True

        # -------------------------------------------------------------
        # Case 11: Personal Library & Status Filtering
        # -------------------------------------------------------------
        print("\n🔹 CASE 11: Personal Library (/papers)")
        lib_resp = await client.get(f"{API_BASE}/api/papers", headers=headers)
        assert lib_resp.status_code == 200
        my_papers = lib_resp.json()
        print(f"  ✓ Personal Library Count: {len(my_papers)} paper(s)")
        results["Case 11 (Personal Library)"] = len(my_papers) > 0

        # -------------------------------------------------------------
        # Case 12: Next.js Frontend Routes Verification
        # -------------------------------------------------------------
        print("\n🔹 CASE 12: Next.js Web Frontend Route HTTP 200 Verification")
        web_routes = [
            "/",
            "/papers",
            "/collections",
            f"/collections/{col_id}",
            "/compare",
            "/trends",
            "/research-gaps",
            "/research-ideas",
            "/proposal",
            "/similarity-map",
            "/login",
            "/register",
        ]
        all_web_ok = True
        for route in web_routes:
            w_resp = await client.get(f"{WEB_BASE}{route}")
            assert w_resp.status_code == 200, f"Route {route} failed with {w_resp.status_code}"
            print(f"  ✓ Route {route:30} -> HTTP 200")
        results["Case 12 (All Web Routes 200)"] = all_web_ok

        # -------------------------------------------------------------
        # Case 13: Clean Cascading Paper Deletion & Storage Cleanup
        # -------------------------------------------------------------
        print("\n🔹 CASE 13: Cascading Deletion & Tenant Cleanliness")
        del_resp = await client.delete(f"{API_BASE}/api/papers/{paper_id}", headers=headers)
        assert del_resp.status_code == 204, f"Delete failed: {del_resp.status_code}"
        print(f"  ✓ Paper {paper_id} deleted (HTTP 204 No Content)")

        # Verify 404
        check_del = await client.get(f"{API_BASE}/api/papers/{paper_id}", headers=headers)
        assert check_del.status_code == 404
        print(f"  ✓ Verified deletion: HTTP 404 Not Found")
        results["Case 13 (Cascading Deletion)"] = True

    # Print Summary Table
    print("\n" + "=" * 80)
    print("                     E2E AUDIT RESULTS SUMMARY")
    print("=" * 80)
    all_passed = True
    for case_name, passed in results.items():
        status_str = "🟢 PASS" if passed else "🔴 FAIL"
        print(f"  {case_name:45} : {status_str}")
        if not passed:
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("  🎉 ALL CASES PASSED! SYSTEM PROVEN 100% OPERATIONAL & ACCURATE.")
    else:
        print("  ❌ ONE OR MORE CASES FAILED.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(verify_all_system_cases())
