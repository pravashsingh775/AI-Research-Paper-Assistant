from __future__ import annotations

import os
from typing import Any

import httpx


class ScholarlyProvider:
    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        raise NotImplementedError


class OpenAlexProvider(ScholarlyProvider):
    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "per-page": min(limit, 50),
            "mailto": os.getenv("OPENALEX_EMAIL", ""),
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get("https://api.openalex.org/works", params=params)
            response.raise_for_status()
        return [
            _normalize_openalex(item) for item in response.json().get("results", [])
        ]


class SemanticScholarProvider(ScholarlyProvider):
    async def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        headers = {"x-api-key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")}
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,abstract,authors,year,venue,externalIds,openAccessPdf,citationCount,url",
        }
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
        return [_normalize_semantic(item) for item in response.json().get("data", [])]


def reconstruct_inverted_index(inv_index: Any) -> str:
    """Reconstruct plain text from OpenAlex positional abstract_inverted_index."""
    if not isinstance(inv_index, dict) or not inv_index:
        return ""
    pos_words: list[tuple[int, str]] = []
    for word, positions in inv_index.items():
        if not word or not isinstance(positions, (list, tuple)):
            continue
        word_str = str(word).strip()
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                pos_words.append((pos, word_str))
    if not pos_words:
        return ""
    # Sort deterministically by word position, breaking ties with word text
    pos_words.sort(key=lambda item: (item[0], item[1]))
    # Deduplicate positions if malformed external data supplies duplicates
    ordered_words: list[str] = []
    seen_positions: set[int] = set()
    for pos, word in pos_words:
        if pos not in seen_positions:
            seen_positions.add(pos)
            ordered_words.append(word)
    return " ".join(ordered_words)


def _normalize_openalex(item: dict[str, Any]) -> dict[str, Any]:
    locations = item.get("locations") or []
    best_location = next(
        (location for location in locations if location.get("is_oa")),
        locations[0] if locations else {},
    )
    abstract = reconstruct_inverted_index(item.get("abstract_inverted_index"))
    primary_location = item.get("primary_location") or {}
    venue = primary_location.get("source") or {}
    authors = ", ".join(
        (author.get("author") or {}).get("display_name", "")
        for author in item.get("authorships", [])
    )
    return {
        "id": str(item.get("id", "")).rsplit("/", 1)[-1],
        "title": item.get("title") or "Untitled paper",
        "summary": abstract,
        "authors_raw": authors,
        "published_date": item.get("publication_date"),
        "venue": venue.get("display_name"),
        "doi": item.get("doi"),
        "citation_count": item.get("cited_by_count", 0),
        "url": item.get("doi") or best_location.get("landing_page_url"),
        "pdf_url": best_location.get("pdf_url"),
        "source": "OpenAlex",
        "open_access": bool((item.get("open_access") or {}).get("is_oa")),
    }


def _normalize_semantic(item: dict[str, Any]) -> dict[str, Any]:
    external_ids = item.get("externalIds") or {}
    open_access = item.get("openAccessPdf") or {}
    return {
        "id": item.get("paperId") or external_ids.get("DOI") or item.get("title"),
        "title": item.get("title") or "Untitled paper",
        "summary": item.get("abstract") or "",
        "authors_raw": ", ".join(
            author.get("name", "") for author in item.get("authors", [])
        ),
        "published_date": str(item.get("year")) if item.get("year") else None,
        "venue": item.get("venue"),
        "doi": external_ids.get("DOI"),
        "citation_count": item.get("citationCount", 0),
        "url": item.get("url"),
        "pdf_url": open_access.get("url"),
        "source": "Semantic Scholar",
        "open_access": bool(open_access.get("url")),
    }


async def search_scholarly(query: str) -> list[dict[str, Any]]:
    providers: list[ScholarlyProvider] = [OpenAlexProvider()]
    if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
        providers.append(SemanticScholarProvider())
    results: list[dict[str, Any]] = []
    for provider in providers:
        try:
            results.extend(await provider.search(query))
        except httpx.HTTPError:
            continue
    deduplicated: dict[str, dict[str, Any]] = {}
    for paper in results:
        key = str(paper.get("doi") or paper.get("title", "")).lower().strip()
        existing = deduplicated.get(key)
        if not existing or paper.get("citation_count", 0) > existing.get(
            "citation_count", 0
        ):
            deduplicated[key] = paper
    return list(deduplicated.values())
