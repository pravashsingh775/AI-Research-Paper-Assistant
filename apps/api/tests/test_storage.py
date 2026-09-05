from __future__ import annotations

import pytest
from pathlib import Path
from uuid import uuid4

from apps.api.app.core.storage import (
    LocalStorageBackend,
    StorageService,
    generate_object_key,
    sanitize_filename,
    validate_pdf_bytes,
)


def test_sanitize_filename() -> None:
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("paper name (2024)!?.pdf") == "paper_name__2024___.pdf"
    assert sanitize_filename("safe_file.pdf") == "safe_file.pdf"


def test_generate_object_key() -> None:
    owner_id = uuid4()
    paper_id = uuid4()
    key = generate_object_key(owner_id, paper_id, "my paper.pdf")
    assert key.startswith(f"papers/{owner_id}/{paper_id}/")
    assert ".." not in key


def test_validate_pdf_bytes_valid() -> None:
    import pymupdf as fitz

    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    valid, err = validate_pdf_bytes(pdf_bytes)
    assert valid is True
    assert err is None


def test_validate_pdf_bytes_invalid() -> None:
    valid, err = validate_pdf_bytes(b"not a pdf file")
    assert valid is False
    assert "not a valid PDF" in str(err)

    valid, err = validate_pdf_bytes(b"")
    assert valid is False
    assert "empty" in str(err)

    valid, err = validate_pdf_bytes(b"%PDF-1.4 header but corrupt bytes")
    assert valid is False
    assert "Corrupted" in str(err) or "zero pages" in str(err)


@pytest.mark.asyncio
async def test_local_storage_crud(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    key = "papers/test-user/test-paper/document.pdf"
    data = b"%PDF-1.4 test binary data"

    # Put
    res = await backend.put(key, data)
    assert res == key

    # Exists
    assert await backend.exists(key) is True
    assert await backend.exists("nonexistent.pdf") is False

    # Get
    retrieved = await backend.get(key)
    assert retrieved == data

    # Delete
    deleted = await backend.delete(key)
    assert deleted is True
    assert await backend.exists(key) is False
