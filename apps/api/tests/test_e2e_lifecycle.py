"""End-to-end integration and paper lifecycle test suite against live services."""

from __future__ import annotations

import asyncio
import io
from uuid import uuid4

import httpx
import pytest


def create_minimal_pdf_bytes(
    title: str = "Test Academic Paper", content: str = "Sample content."
) -> bytes:
    """Create valid binary PDF bytes using pymupdf."""
    import pymupdf as fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), title, fontsize=18)
    page.insert_text((50, 120), "Abstract\n" + content, fontsize=11)
    page.insert_text(
        (50, 200),
        "Materials and Methods\nWe evaluated our approach across 100 samples.",
        fontsize=11,
    )
    page.insert_text((50, 280), "Results\nThe method achieved 95% accuracy.", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def base_url() -> str:
    return "http://localhost:8000"


@pytest.mark.asyncio
async def test_e2e_user_registration_and_login(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        email = f"e2e_{uuid4().hex[:8]}@example.com"
        password = "TestPassword123!"

        # 1. Register
        reg_resp = await client.post(
            "/api/auth/register", json={"email": email, "password": password}
        )
        assert reg_resp.status_code == 201, reg_resp.text
        data = reg_resp.json()
        assert "access_token" in data
        token = data["access_token"]

        # 2. Login
        login_resp = await client.post(
            "/api/auth/token",
            data={"username": email, "password": password},
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        assert login_resp.status_code == 200, login_resp.text
        assert login_resp.json()["access_token"] == token

        # 3. Auth me
        me_resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_resp.status_code == 200, me_resp.text
        assert me_resp.json()["email"] == email


@pytest.mark.asyncio
async def test_pdf_upload_validation_and_lifecycle(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, timeout=30) as client:
        email = f"uploader_{uuid4().hex[:8]}@example.com"
        reg = await client.post(
            "/api/auth/register", json={"email": email, "password": "Password123!"}
        )
        assert reg.status_code == 201, reg.text
        token = reg.json()["access_token"]
        auth_header = {"Authorization": f"Bearer {token}"}

        # 1. Negative: Non-PDF rejected
        bad_resp = await client.post(
            "/api/papers/upload",
            files={"file": ("malicious.txt", b"This is not a PDF", "text/plain")},
            headers=auth_header,
        )
        assert bad_resp.status_code == 422
        assert "not a valid PDF" in bad_resp.json()["detail"]

        # 2. Negative: Empty PDF rejected
        empty_resp = await client.post(
            "/api/papers/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
            headers=auth_header,
        )
        assert empty_resp.status_code == 422
        assert "empty" in empty_resp.json()["detail"]

        # 3. Valid PDF Upload
        pdf_bytes = create_minimal_pdf_bytes(
            "Quantum Genomic Transformer", "Long sequence modeling of DNA."
        )
        upload_resp = await client.post(
            "/api/papers/upload",
            files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
            headers=auth_header,
        )
        assert upload_resp.status_code == 200, upload_resp.text
        data = upload_resp.json()
        paper_id = data["id"]
        document_id = data["document_id"]
        job_id = data["job_id"]
        assert data["evidence_state"] == "processing"

        # 4. Get Paper while or after processing
        get_resp = await client.get(f"/api/papers/{paper_id}", headers=auth_header)
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["id"] == paper_id

        # 5. Wait for worker processing to complete
        for _ in range(30):
            job_resp = await client.get(f"/api/jobs/{job_id}", headers=auth_header)
            if job_resp.status_code == 200 and job_resp.json()["status"] == "COMPLETED":
                break
            await asyncio.sleep(1)

        # 6. Create Collection and Add Paper
        col_resp = await client.post(
            "/api/collections", json={"name": "Genomics Review"}, headers=auth_header
        )
        assert col_resp.status_code == 201, col_resp.text
        col_id = col_resp.json()["id"]

        add_resp = await client.post(
            f"/api/collections/{col_id}/papers", json={"paper_id": paper_id}, headers=auth_header
        )
        assert add_resp.status_code == 201, add_resp.text

        # Verify collection contains paper
        col_get = await client.get(f"/api/collections/{col_id}", headers=auth_header)
        assert col_get.status_code == 200, col_get.text
        assert len(col_get.json()["papers"]) == 1
        assert col_get.json()["papers"][0]["id"] == paper_id

        # 7. Delete Paper and Verify Cascade Cleanup
        del_resp = await client.delete(f"/api/papers/{paper_id}", headers=auth_header)
        assert del_resp.status_code == 204, del_resp.text

        # Paper should now be 404
        get_del = await client.get(f"/api/papers/{paper_id}", headers=auth_header)
        assert get_del.status_code == 404, get_del.text

        # Collection should no longer contain paper
        col_get_after = await client.get(f"/api/collections/{col_id}", headers=auth_header)
        assert len(col_get_after.json()["papers"]) == 0
