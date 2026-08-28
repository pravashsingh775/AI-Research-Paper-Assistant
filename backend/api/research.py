"""
FastAPI API router for Mode-1 research discovery.

HTTP -> Pydantic request -> ResearchSearchService -> ResearchPaperAnalyzer
     -> typed aggregate response -> HTTP

This module intentionally does NOT implement embeddings, FAISS, retrieval,
reranking, ranking, RAG, LLM calls, PDF processing, or scientific analysis.
Expensive AI dependencies must be initialized once by the application layer
and exposed through ``app.state``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.research.analyzer import (
    ResearchAnalysisResponse as DomainAnalysisResponse,
    ResearchPaperAnalyzer,
)
from backend.research.search import (
    FinalRankingStageError,
    InvalidSearchRequestError,
    ResearchSearchError,
    ResearchSearchService,
    RetrievalStageError,
    RerankingStageError,
    SearchResultValidationError,
)
from backend.schemas.research import (
    PaperAnalysis,
    ResearchPaper,
    ResearchResponse,
    ResearchSearchRequest,
    ResearchSearchResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


def get_research_service(request: Request) -> ResearchSearchService:
    """Resolve the already-initialized Mode-1 search service."""
    service = getattr(request.app.state, "research_service", None)
    if service is None or not callable(getattr(service, "search", None)):
        logger.error("Research search service is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research search service is unavailable.",
        )
    return service


def get_research_analyzer(request: Request) -> ResearchPaperAnalyzer:
    """Resolve the already-initialized Top-K paper analyzer."""
    analyzer = getattr(request.app.state, "research_analyzer", None)
    if analyzer is None or not callable(getattr(analyzer, "analyze", None)):
        logger.error("Research paper analyzer is not configured.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research analysis service is unavailable.",
        )
    return analyzer


def _validate_search_response(raw: Any) -> ResearchSearchResponse:
    """Validate the domain search result without changing its ordering."""
    try:
        return ResearchSearchResponse.model_validate(
            raw,
            from_attributes=True,
        )
    except Exception as exc:
        logger.exception("Research search returned an invalid response contract.")
        raise SearchResultValidationError(
            "Research search returned an invalid response contract."
        ) from exc


def _validate_analysis_response(raw: Any) -> DomainAnalysisResponse:
    """Require the actual analyzer response contract."""
    if not isinstance(raw, DomainAnalysisResponse):
        raise SearchResultValidationError(
            "Research analyzer returned an unexpected response type."
        )
    return raw


def _build_aggregate_response(
    *,
    search_response: ResearchSearchResponse,
    analysis_response: DomainAnalysisResponse,
) -> ResearchResponse:
    """
    Join search and analysis strictly by canonical document_id.

    Search order remains authoritative. No sorting, truncation, deduplication,
    score recalculation, or fabricated metadata is performed.
    """
    analyses_by_document_id: dict[str, PaperAnalysis] = {}

    for raw_analysis in analysis_response.papers:
        analysis = PaperAnalysis.model_validate(
            raw_analysis,
            from_attributes=True,
        )
        if analysis.document_id in analyses_by_document_id:
            raise SearchResultValidationError(
                f"Analyzer returned duplicate document_id={analysis.document_id!r}."
            )
        analyses_by_document_id[analysis.document_id] = analysis

    papers: list[ResearchPaper] = []

    for result in search_response.results:
        analysis = analyses_by_document_id.get(result.document_id)

        papers.append(
            ResearchPaper(
                document_id=result.document_id,
                paper_id=result.paper_id,
                rank=result.rank,
                title=result.title,
                summary=result.summary,
                retrieval_score=result.retrieval_score,
                reranker_score=result.reranker_score,
                final_score=result.final_score,
                analysis=analysis,
            )
        )

    return ResearchResponse(
        query=search_response.query,
        total_candidates=search_response.candidate_count,
        returned_count=len(papers),
        papers=papers,
        trace=search_response.trace,
        # No validation result is fabricated. Attach it here when the existing
        # research validation service is actually integrated upstream.
        validation=None,
    )


def _build_search_only_response(
    search_response: ResearchSearchResponse,
) -> ResearchResponse:
    """Return ranked paper metadata without running expensive analysis."""
    return ResearchResponse(
        query=search_response.query,
        total_candidates=search_response.candidate_count,
        returned_count=len(search_response.results),
        papers=[
            ResearchPaper(
                document_id=result.document_id,
                paper_id=result.paper_id,
                rank=result.rank,
                title=result.title,
                summary=result.summary,
                retrieval_score=result.retrieval_score,
                reranker_score=result.reranker_score,
                final_score=result.final_score,
                analysis=None,
            )
            for result in search_response.results
        ],
        trace=search_response.trace,
        validation=None,
    )


def _raise_http_error(exc: Exception) -> None:
    """Map known service failures to safe HTTP responses."""
    if isinstance(exc, InvalidSearchRequestError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid research search request.",
        ) from exc

    if isinstance(
        exc,
        (
            RetrievalStageError,
            RerankingStageError,
            FinalRankingStageError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Research search service is temporarily unavailable.",
        ) from exc

    if isinstance(exc, SearchResultValidationError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research search returned an invalid result.",
        ) from exc

    if isinstance(exc, ResearchSearchError):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Research search failed.",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Research search failed unexpectedly.",
    ) from exc


@router.post(
    "/search",
    response_model=ResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search research papers",
    description=(
        "Run the existing Mode-1 research discovery pipeline, preserve its "
        "authoritative ranking, analyze the selected papers, and return a "
        "typed Top-K response."
    ),
    response_description="Validated research discovery response.",
)
def search_research(
    request: ResearchSearchRequest,
    service: ResearchSearchService = Depends(get_research_service),
    analyzer: ResearchPaperAnalyzer = Depends(get_research_analyzer),
) -> ResearchResponse:
    """HTTP orchestration only: schema -> search service -> analyzer -> schema."""
    started = time.perf_counter()

    try:
        raw_search = service.search(
            request.query,
            candidate_k=request.candidate_k,
            final_k=request.final_k,
        )
        search_response = _validate_search_response(raw_search)

        if request.include_analysis:
            # The analyzer accepts the existing search response and preserves
            # the selected-paper order; it owns the seven analysis components.
            raw_analysis = analyzer.analyze(raw_search)
            analysis_response = _validate_analysis_response(raw_analysis)
            response = _build_aggregate_response(
                search_response=search_response,
                analysis_response=analysis_response,
            )
        else:
            response = _build_search_only_response(search_response)

        logger.info(
            "Research API completed: results=%d total_time=%.4fs.",
            response.returned_count,
            time.perf_counter() - started,
        )
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Research API request failed.")
        _raise_http_error(exc)


def run_self_test() -> None:
    """Offline API contract tests; no ML models, FAISS, LLM, or dataset."""
    from dataclasses import dataclass

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.research.analyzer import PaperAnalysis

    @dataclass
    class FakeResult:
        document_id: str
        rank: int
        final_score: float
        retrieval_score: float
        reranker_score: float
        title: str
        summary: str

    class FakeSearchService:
        def search(self, query: str, *, candidate_k: int, final_k: int) -> Any:
            assert query == "transformer medical segmentation"
            assert candidate_k == 50
            assert final_k == 10
            return type(
                "SearchResponse",
                (),
                {
                    "query": query,
                    "candidate_count": 2,
                    "result_count": 2,
                    "results": [
                        FakeResult(
                            "paper-A", 1, 0.95, 0.90, 0.94,
                            "Paper A", "Summary A"
                        ),
                        FakeResult(
                            "paper-B", 2, 0.91, 0.87, 0.90,
                            "Paper B", "Summary B"
                        ),
                    ],
                    "trace": None,
                },
            )()

    class FakeAnalyzer:
        def analyze(self, search_response: Any) -> DomainAnalysisResponse:
            return DomainAnalysisResponse(
                query=search_response.query,
                analyzed_count=2,
                papers=[
                    PaperAnalysis(
                        document_id="paper-A",
                        rank=1,
                        title="Paper A",
                        summary="A summary",
                        model={"name": "Model A"},
                        dataset={"name": "Dataset A"},
                        methodology={"steps": ["training"]},
                        findings=["Finding A"],
                        strengths=["Strength A"],
                        weaknesses=["Weakness A"],
                    ),
                    PaperAnalysis(
                        document_id="paper-B",
                        rank=2,
                        title="Paper B",
                        summary="B summary",
                        model={"name": "Model B"},
                        dataset={"name": "Dataset B"},
                        methodology={"steps": ["evaluation"]},
                        findings=["Finding B"],
                        strengths=["Strength B"],
                        weaknesses=["Weakness B"],
                    ),
                ],
                requested_count=2,
                failed_count=0,
                partial_count=0,
                total_latency_seconds=0.001,
            )

    app = FastAPI()
    app.state.research_service = FakeSearchService()
    app.state.research_analyzer = FakeAnalyzer()
    app.include_router(router)

    client = TestClient(app)

    response = client.post(
        "/research/search",
        json={
            "query": "transformer medical segmentation",
            "candidate_k": 50,
            "final_k": 10,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query"] == "transformer medical segmentation"
    assert payload["returned_count"] == 2
    assert [p["document_id"] for p in payload["papers"]] == [
        "paper-A", "paper-B"
    ]
    assert payload["papers"][0]["rank"] == 1
    assert payload["papers"][1]["rank"] == 2
    assert payload["papers"][0]["analysis"]["model"]["name"] == "Model A"

    bad = client.post(
        "/research/search",
        json={"query": "   ", "candidate_k": 50, "final_k": 10},
    )
    assert bad.status_code == 422

    unavailable_app = FastAPI()
    unavailable_app.include_router(router)
    unavailable_client = TestClient(unavailable_app)
    unavailable = unavailable_client.post(
        "/research/search",
        json={"query": "valid query", "candidate_k": 50, "final_k": 10},
    )
    assert unavailable.status_code == 503

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/research/search" in openapi.json()["paths"]

    print("api/research.py self-test: PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_self_test()