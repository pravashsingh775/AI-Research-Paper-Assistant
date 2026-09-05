"""Manual Docker integration test for the private PDF workflow.

Run inside the backend container while the worker is running:
python /tmp/integration_e2e.py
"""

import asyncio
import uuid

import fitz
import httpx


async def main() -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Methodology\nWe evaluate graph retrieval using the Cora dataset. "
        "The methodology uses dense retrieval with a bi-encoder.\nResults\n"
        "The Cora dataset improves retrieval evaluation reproducibility.",
    )
    document.save("/tmp/integration-paper.pdf")
    email = f"integration-{uuid.uuid4().hex[:12]}@example.com"

    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.post("/api/search", json={"topic": "retrieval augmented generation"})).status_code == 200
        registration = await client.post("/api/auth/register", json={"email": email, "password": "IntegrationPass123!"})
        assert registration.status_code == 201, registration.text
        headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
        assert (await client.get("/api/auth/me", headers=headers)).status_code == 200
        with open("/tmp/integration-paper.pdf", "rb") as handle:
            upload = await client.post("/api/papers/upload", headers=headers, files={"file": ("integration-paper.pdf", handle, "application/pdf")})
        assert upload.status_code == 200, upload.text
        payload = upload.json()
        for _ in range(30):
            job = await client.get(f"/api/jobs/{payload['job_id']}", headers=headers)
            assert job.status_code == 200, job.text
            if job.json()["status"] == "COMPLETED":
                break
            await asyncio.sleep(1)
        else:
            raise AssertionError(job.text)
        dataset = await client.post("/api/qa", headers=headers, json={"document_id": payload["document_id"], "question": "What dataset was used?"})
        assert dataset.status_code == 200 and dataset.json()["status"] == "evidence-backed" and dataset.json()["evidence"], dataset.text
        methodology = await client.post("/api/qa", headers=headers, json={"document_id": payload["document_id"], "question": "What is the main methodology of this paper?"})
        assert methodology.status_code == 200 and methodology.json()["status"] == "evidence-backed", methodology.text
        unsupported = await client.post("/api/qa", headers=headers, json={"document_id": payload["document_id"], "question": "What color were the laboratory walls?"})
        assert unsupported.status_code == 200 and unsupported.json()["status"] == "insufficient-evidence", unsupported.text
        print({"document_id": payload["document_id"], "job_id": payload["job_id"], "dataset": dataset.json()["status"], "methodology": methodology.json()["status"], "unsupported": unsupported.json()["status"]})


if __name__ == "__main__":
    asyncio.run(main())
