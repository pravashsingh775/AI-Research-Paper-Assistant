from __future__ import annotations

import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID
import logging

import httpx

from apps.api.app.core.config import get_settings

logger = logging.getLogger(__name__)

MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_PDF_PAGES = 500


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and unsafe characters."""
    clean = Path(filename).name
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    return clean[:200] or "document.pdf"


def generate_object_key(owner_id: UUID | str | None, paper_id: UUID | str, filename: str) -> str:
    """Generate deterministic, partitioned object key."""
    owner_part = str(owner_id) if owner_id else "public"
    safe_name = sanitize_filename(filename)
    return f"papers/{owner_part}/{paper_id}/{safe_name}"


def validate_pdf_bytes(
    content: bytes, max_bytes: int = MAX_PDF_SIZE_BYTES
) -> tuple[bool, str | None]:
    """Validate PDF magic bytes, size limits, and basic structure."""
    if not content:
        return False, "Uploaded file is empty."
    if len(content) > max_bytes:
        return (
            False,
            f"File size ({len(content)} bytes) exceeds the {max_bytes // (1024 * 1024)} MB limit.",
        )
    if not content.startswith(b"%PDF-") and not content.startswith(b"%PDF"):
        return False, "File is not a valid PDF document (missing %PDF header)."
    try:
        import pymupdf as fitz

        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return False, "PDF contains zero pages."
        if page_count > MAX_PDF_PAGES:
            doc.close()
            return (
                False,
                f"PDF exceeds maximum page limit ({page_count} > {MAX_PDF_PAGES}).",
            )
        doc.close()
    except Exception as exc:
        return False, f"Corrupted or unreadable PDF structure: {exc}"

    return True, None


class LocalStorageBackend:
    """Stores binary objects on local filesystem."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, object_key: str) -> Path:
        safe_rel = Path(object_key).as_posix().lstrip("/")
        return self.base_dir / safe_rel

    async def put(self, object_key: str, data: bytes, content_type: str = "application/pdf") -> str:
        target_path = self._resolve_path(object_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        return object_key

    async def get(self, object_key: str) -> bytes:
        target_path = self._resolve_path(object_key)
        if not target_path.exists():
            raise FileNotFoundError(f"Object not found: {object_key}")
        return target_path.read_bytes()

    async def delete(self, object_key: str) -> bool:
        target_path = self._resolve_path(object_key)
        if target_path.exists():
            target_path.unlink()
            return True
        return False

    async def exists(self, object_key: str) -> bool:
        return self._resolve_path(object_key).exists()


class S3StorageBackend:
    """Stores binary objects in S3 / MinIO using standard REST API and AWS SigV4."""

    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self._bucket_verified = False

    @property
    def host_header(self) -> str:
        netloc = httpx.URL(self.endpoint_url).netloc
        if isinstance(netloc, bytes):
            return netloc.decode("ascii")
        return str(netloc)

    def _sign_request(
        self,
        method: str,
        url_path: str,
        headers: dict[str, str],
        payload: bytes,
    ) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        payload_hash = hashlib.sha256(payload).hexdigest()
        clean_headers: dict[str, str] = {}
        for k, v in headers.items():
            str_val = v.decode("ascii") if isinstance(v, bytes) else str(v)
            clean_headers[k] = str_val

        clean_headers["x-amz-date"] = amz_date
        clean_headers["x-amz-content-sha256"] = payload_hash

        canonical_headers = "".join(
            f"{k.lower()}:{clean_headers[k].strip()}\n" for k in sorted(clean_headers)
        )
        signed_headers = ";".join(sorted(k.lower() for k in clean_headers))

        canonical_request = (
            f"{method.upper()}\n{url_path}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        )

        algorithm = "AWS4-HMAC-SHA256"
        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"

        def _sign(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        k_date = _sign(("AWS4" + self.secret_key).encode("utf-8"), date_stamp)
        k_region = _sign(k_date, self.region)
        k_service = _sign(k_region, "s3")
        k_signing = _sign(k_service, "aws4_request")

        signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        authorization = (
            f"{algorithm} Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        clean_headers["Authorization"] = authorization
        return clean_headers

    async def _ensure_bucket(self) -> None:
        if self._bucket_verified:
            return
        url_path = f"/{self.bucket}"
        headers = {"Host": self.host_header}
        signed = self._sign_request("HEAD", url_path, dict(headers), b"")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.head(f"{self.endpoint_url}{url_path}", headers=signed)
            if resp.status_code in (403, 404):
                # Create bucket
                put_headers = dict(headers)
                signed_put = self._sign_request("PUT", url_path, put_headers, b"")
                put_resp = await client.put(f"{self.endpoint_url}{url_path}", headers=signed_put)
                if put_resp.status_code not in (200, 409):
                    put_resp.raise_for_status()
        self._bucket_verified = True

    async def put(self, object_key: str, data: bytes, content_type: str = "application/pdf") -> str:
        await self._ensure_bucket()
        url_path = f"/{self.bucket}/{object_key.lstrip('/')}"
        headers = {
            "Host": self.host_header,
            "Content-Type": content_type,
            "Content-Length": str(len(data)),
        }
        signed = self._sign_request("PUT", url_path, headers, data)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.put(f"{self.endpoint_url}{url_path}", headers=signed, content=data)
            resp.raise_for_status()
        return object_key

    async def get(self, object_key: str) -> bytes:
        url_path = f"/{self.bucket}/{object_key.lstrip('/')}"
        headers = {"Host": self.host_header}
        signed = self._sign_request("GET", url_path, headers, b"")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{self.endpoint_url}{url_path}", headers=signed)
            if resp.status_code == 404:
                raise FileNotFoundError(f"Object not found: {object_key}")
            resp.raise_for_status()
            return resp.content

    async def delete(self, object_key: str) -> bool:
        url_path = f"/{self.bucket}/{object_key.lstrip('/')}"
        headers = {"Host": self.host_header}
        signed = self._sign_request("DELETE", url_path, headers, b"")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.delete(f"{self.endpoint_url}{url_path}", headers=signed)
            return resp.status_code in (200, 204)

    async def exists(self, object_key: str) -> bool:
        url_path = f"/{self.bucket}/{object_key.lstrip('/')}"
        headers = {"Host": self.host_header}
        signed = self._sign_request("HEAD", url_path, headers, b"")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.head(f"{self.endpoint_url}{url_path}", headers=signed)
            return resp.status_code == 200


class StorageService:
    """Unified storage service with auto-selection and local fallback."""

    def __init__(self) -> None:
        settings = get_settings()
        self.local_backend = LocalStorageBackend(settings.storage_local_dir)
        self.s3_backend: S3StorageBackend | None = None

        if settings.storage_backend in ("s3", "auto") and settings.object_storage_endpoint:
            try:
                self.s3_backend = S3StorageBackend(
                    endpoint_url=settings.object_storage_endpoint,
                    bucket=settings.object_storage_bucket,
                    access_key=settings.object_storage_access_key,
                    secret_key=settings.object_storage_secret_key,
                    region=settings.object_storage_region,
                )
            except Exception as exc:
                logger.warning(
                    f"Could not initialize S3 storage backend: {exc}. Using local storage."
                )

    async def put(self, object_key: str, data: bytes, content_type: str = "application/pdf") -> str:
        if self.s3_backend and get_settings().storage_backend != "local":
            try:
                return await self.s3_backend.put(object_key, data, content_type)
            except Exception as exc:
                logger.warning(
                    f"S3 put failed for {object_key}: {exc}. Falling back to local storage."
                )
        return await self.local_backend.put(object_key, data, content_type)

    async def get(self, object_key: str) -> bytes:
        if self.s3_backend and get_settings().storage_backend != "local":
            try:
                return await self.s3_backend.get(object_key)
            except Exception as exc:
                logger.warning(f"S3 get failed for {object_key}: {exc}. Trying local storage.")
        return await self.local_backend.get(object_key)

    async def delete(self, object_key: str) -> bool:
        deleted = False
        if self.s3_backend and get_settings().storage_backend != "local":
            try:
                deleted = await self.s3_backend.delete(object_key)
            except Exception as exc:
                logger.warning(f"S3 delete failed for {object_key}: {exc}.")
        local_del = await self.local_backend.delete(object_key)
        return deleted or local_del

    async def exists(self, object_key: str) -> bool:
        if self.s3_backend and get_settings().storage_backend != "local":
            try:
                if await self.s3_backend.exists(object_key):
                    return True
            except Exception:
                pass
        return await self.local_backend.exists(object_key)


_storage_instance: StorageService | None = None


def get_storage() -> StorageService:
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = StorageService()
    return _storage_instance
