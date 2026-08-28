"""Small HTTP client for the FastAPI backend."""

from __future__ import annotations

from typing import Any

import requests


class ApiError(RuntimeError):
	"""A readable backend error suitable for displaying in the UI."""


class ResearchApi:
	def __init__(self, base_url: str, timeout: int = 300) -> None:
		self.base_url = base_url.rstrip("/")
		self.timeout = timeout

	def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
		try:
			response = requests.request(
				method,
				f"{self.base_url}{path}",
				timeout=self.timeout,
				**kwargs,
			)
		except requests.RequestException as exc:
			raise ApiError(f"Backend is unreachable: {exc}") from exc

		if not response.ok:
			try:
				detail = response.json().get("detail", response.text)
			except ValueError:
				detail = response.text
			raise ApiError(f"{response.status_code}: {detail or 'Request failed.'}")

		try:
			payload = response.json()
		except ValueError as exc:
			raise ApiError("Backend returned an invalid JSON response.") from exc
		if not isinstance(payload, dict):
			raise ApiError("Backend returned an unexpected response.")
		return payload

	def health(self) -> dict[str, Any]:
		return self._request("GET", "/health")

	def search(self, query: str, candidate_k: int, final_k: int) -> dict[str, Any]:
		return self._request(
			"POST",
			"/research/search",
			json={"query": query, "candidate_k": candidate_k, "final_k": final_k, "include_analysis": False},
		)

	def upload(self, file_name: str, file_bytes: bytes) -> dict[str, Any]:
		if not file_name.lower().endswith(".pdf"):
			raise ApiError("Only PDF research papers are currently supported by the backend.")
		content_types = {
			".pdf": "application/pdf",
			".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
			".txt": "text/plain",
		}
		suffix = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
		return self._request(
			"POST",
			"/api/papers/upload",
			files={"file": (file_name, file_bytes, content_types.get(suffix, "application/octet-stream"))},
		)

	def ask(self, document_id: str, question: str, session_id: str) -> dict[str, Any]:
		return self._request(
			"POST",
			"/api/qa",
			json={
				"document_id": document_id,
				"question": question,
				"session_id": session_id,
			},
		)
