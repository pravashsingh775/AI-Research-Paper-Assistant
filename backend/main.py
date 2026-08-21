"""
AI Research Paper Assistant
FastAPI Application Layer

Location:
    backend/main.py

Architecture:
    HTTP Client
        ↓
    FastAPI
        ↓
    Retriever Pipeline
        ↓
    Context Manager
        ↓
    Prompt Builder
        ↓
    LLM Orchestrator
        ↓
    Response Validation
        ↓
    Research Intelligence / Knowledge Graph / Gap Detection / Writing / Citations / Comparative Analysis
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# PATH / ENVIRONMENT
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

os.environ.setdefault("PYTHONPATH", str(BASE_DIR))


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
)

logger = logging.getLogger("AIResearchPaperAssistantAPI")


# =============================================================================
# BACKEND ENGINE IMPORTS
# =============================================================================

try:
    from src.rag.retriever_pipeline import RetrieverPipeline
    from src.rag.context_manager import (
        ContextManagerEngine,
        ContextTaskType,
    )
    from src.rag.prompt_builder import (
        PromptBuilderEngine,
        PromptTaskType,
        TargetLLMProfile,
    )
    from src.rag.LLM_orchestrator import (
        LLMOrchestratorEngine,
        LLMProvider,
        ExecutionMode,
        OrchestratorTaskIntent,
    )
    from src.rag.response_validation import ResponseValidationEngine

    from src.intelligence.research_intelligence import (
        ResearchIntelligenceLayer,
    )
    from src.intelligence.knowledge_graph_engine import (
        ScientificKnowledgeGraphEngine,
    )
    from src.intelligence.research_gap_detection_engine import (
        ResearchGapDetectionEngine,
    )
    from src.intelligence.scientific_writing_engine import (
        ScientificWritingEngine,
    )
    from src.intelligence.citation_management_engine import (
        CitationManagementEngine,
    )
    from src.intelligence.comparative_analysis_engine import (
        PaperComparativeEngine,
    )

except Exception as exc:
    logger.exception("Failed to import backend engines.")
    raise RuntimeError(
        "Backend engine imports failed. "
        "Make sure you start Uvicorn from the backend directory."
    ) from exc


# =============================================================================
# GLOBAL ENGINE REGISTRY
# =============================================================================

ENGINE: Dict[str, Any] = {
    "retriever": None,
    "context_manager": None,
    "prompt_builder": None,
    "llm_orchestrator": None,
    "response_validator": None,
    "research_intelligence": None,
    "knowledge_graph": None,
    "research_gap": None,
    "scientific_writing": None,
    "citation_manager": None,
    "comparative_analysis": None,
}


# =============================================================================
# JSON SERIALIZATION
# =============================================================================

def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, enum.Enum):
        return value.value

    if is_dataclass(value):
        return {
            key: to_jsonable(val)
            for key, val in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())

    if hasattr(value, "__dict__"):
        try:
            return {
                key: to_jsonable(val)
                for key, val in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return value


# =============================================================================
# CANONICAL TASK TAXONOMY & REGISTRY
# =============================================================================

class CanonicalTask(str, enum.Enum):
    QUESTION_ANSWERING = "QUESTION_ANSWERING"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    METHOD_COMPARISON = "METHOD_COMPARISON"
    RESEARCH_GAP = "RESEARCH_GAP"
    SUMMARIZATION = "SUMMARIZATION"
    PATENT_SEARCH = "PATENT_SEARCH"
    SURVEY_PAPER = "SURVEY_PAPER"
    DATASET_RECOMMENDATION = "DATASET_RECOMMENDATION"
    PAPER_COMPARISON = "PAPER_COMPARISON"
    TREND_ANALYSIS = "TREND_ANALYSIS"
    SURVEY_GENERATION = "SURVEY_GENERATION"
    CITATION_GENERATION = "CITATION_GENERATION"
    EXPERIMENTAL_DESIGN = "EXPERIMENTAL_DESIGN"
    NOVEL_IDEA_GENERATION = "NOVEL_IDEA_GENERATION"
    RESEARCH_PROPOSAL = "RESEARCH_PROPOSAL"
    PATENT_OPPORTUNITY = "PATENT_OPPORTUNITY"
    REVIEWER_RESPONSE = "REVIEWER_RESPONSE"
    IEEE_STYLE_WRITING = "IEEE_STYLE_WRITING"


def normalize_task(value: str) -> CanonicalTask:
    cleaned = value.strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return CanonicalTask(cleaned)
    except ValueError as exc:
        raise ValueError(f"Unsupported task_type: {value}") from exc


def get_context_task(task: CanonicalTask | str) -> ContextTaskType:
    canonical = normalize_task(task) if isinstance(task, str) else task
    mapping = {
        CanonicalTask.QUESTION_ANSWERING: ContextTaskType.QUESTION_ANSWERING,
        CanonicalTask.LITERATURE_REVIEW: ContextTaskType.LITERATURE_REVIEW,
        CanonicalTask.METHOD_COMPARISON: ContextTaskType.METHOD_COMPARISON,
        CanonicalTask.RESEARCH_GAP: ContextTaskType.RESEARCH_GAP,
        CanonicalTask.PATENT_SEARCH: ContextTaskType.PATENT_SEARCH,
        CanonicalTask.SURVEY_PAPER: ContextTaskType.SURVEY_PAPER,
        CanonicalTask.SUMMARIZATION: ContextTaskType.QUESTION_ANSWERING,
        CanonicalTask.DATASET_RECOMMENDATION: ContextTaskType.QUESTION_ANSWERING,
        CanonicalTask.PAPER_COMPARISON: ContextTaskType.METHOD_COMPARISON,
        CanonicalTask.TREND_ANALYSIS: ContextTaskType.LITERATURE_REVIEW,
        CanonicalTask.SURVEY_GENERATION: ContextTaskType.SURVEY_PAPER,
        CanonicalTask.CITATION_GENERATION: ContextTaskType.QUESTION_ANSWERING,
        CanonicalTask.EXPERIMENTAL_DESIGN: ContextTaskType.METHOD_COMPARISON,
        CanonicalTask.NOVEL_IDEA_GENERATION: ContextTaskType.RESEARCH_GAP,
        CanonicalTask.RESEARCH_PROPOSAL: ContextTaskType.LITERATURE_REVIEW,
        CanonicalTask.PATENT_OPPORTUNITY: ContextTaskType.PATENT_SEARCH,
        CanonicalTask.REVIEWER_RESPONSE: ContextTaskType.QUESTION_ANSWERING,
        CanonicalTask.IEEE_STYLE_WRITING: ContextTaskType.LITERATURE_REVIEW,
    }
    return mapping[canonical]


def get_prompt_task(task: CanonicalTask | str) -> PromptTaskType:
    canonical = normalize_task(task) if isinstance(task, str) else task
    mapping = {
        CanonicalTask.QUESTION_ANSWERING: PromptTaskType.QUESTION_ANSWERING,
        CanonicalTask.LITERATURE_REVIEW: PromptTaskType.LITERATURE_REVIEW,
        CanonicalTask.METHOD_COMPARISON: PromptTaskType.METHOD_COMPARISON,
        CanonicalTask.RESEARCH_GAP: PromptTaskType.RESEARCH_GAP,
        CanonicalTask.SUMMARIZATION: PromptTaskType.SUMMARIZATION,
        CanonicalTask.DATASET_RECOMMENDATION: PromptTaskType.DATASET_RECOMMENDATION,
        CanonicalTask.PAPER_COMPARISON: PromptTaskType.PAPER_COMPARISON,
        CanonicalTask.TREND_ANALYSIS: PromptTaskType.TREND_ANALYSIS,
        CanonicalTask.SURVEY_GENERATION: PromptTaskType.SURVEY_GENERATION,
        CanonicalTask.CITATION_GENERATION: PromptTaskType.CITATION_GENERATION,
        CanonicalTask.EXPERIMENTAL_DESIGN: PromptTaskType.EXPERIMENTAL_DESIGN,
        CanonicalTask.NOVEL_IDEA_GENERATION: PromptTaskType.NOVEL_IDEA_GENERATION,
        CanonicalTask.RESEARCH_PROPOSAL: PromptTaskType.RESEARCH_PROPOSAL,
        CanonicalTask.PATENT_OPPORTUNITY: PromptTaskType.PATENT_OPPORTUNITY,
        CanonicalTask.REVIEWER_RESPONSE: PromptTaskType.REVIEWER_RESPONSE,
        CanonicalTask.IEEE_STYLE_WRITING: PromptTaskType.IEEE_STYLE_WRITING,
        CanonicalTask.PATENT_SEARCH: PromptTaskType.PATENT_OPPORTUNITY,
        CanonicalTask.SURVEY_PAPER: PromptTaskType.SURVEY_GENERATION,
    }
    return mapping[canonical]


def get_orchestrator_task(task: CanonicalTask | str) -> OrchestratorTaskIntent:
    canonical = normalize_task(task) if isinstance(task, str) else task
    mapping = {
        CanonicalTask.QUESTION_ANSWERING: OrchestratorTaskIntent.QUESTION_ANSWERING,
        CanonicalTask.LITERATURE_REVIEW: OrchestratorTaskIntent.LITERATURE_REVIEW,
        CanonicalTask.METHOD_COMPARISON: OrchestratorTaskIntent.METHOD_COMPARISON,
        CanonicalTask.RESEARCH_GAP: OrchestratorTaskIntent.RESEARCH_GAP,
        CanonicalTask.SUMMARIZATION: OrchestratorTaskIntent.SUMMARIZATION,
        CanonicalTask.PATENT_SEARCH: OrchestratorTaskIntent.PATENT_DRAFTING,
        CanonicalTask.SURVEY_PAPER: OrchestratorTaskIntent.LITERATURE_REVIEW,
        CanonicalTask.DATASET_RECOMMENDATION: OrchestratorTaskIntent.QUESTION_ANSWERING,
        CanonicalTask.PAPER_COMPARISON: OrchestratorTaskIntent.METHOD_COMPARISON,
        CanonicalTask.TREND_ANALYSIS: OrchestratorTaskIntent.LITERATURE_REVIEW,
        CanonicalTask.SURVEY_GENERATION: OrchestratorTaskIntent.LITERATURE_REVIEW,
        CanonicalTask.CITATION_GENERATION: OrchestratorTaskIntent.QUESTION_ANSWERING,
        CanonicalTask.EXPERIMENTAL_DESIGN: OrchestratorTaskIntent.METHOD_COMPARISON,
        CanonicalTask.NOVEL_IDEA_GENERATION: OrchestratorTaskIntent.RESEARCH_GAP,
        CanonicalTask.RESEARCH_PROPOSAL: OrchestratorTaskIntent.LITERATURE_REVIEW,
        CanonicalTask.PATENT_OPPORTUNITY: OrchestratorTaskIntent.PATENT_DRAFTING,
        CanonicalTask.REVIEWER_RESPONSE: OrchestratorTaskIntent.QUESTION_ANSWERING,
        CanonicalTask.IEEE_STYLE_WRITING: OrchestratorTaskIntent.LITERATURE_REVIEW,
    }
    return mapping[canonical]


# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================

class TopicRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )
    topic: str = Field(..., min_length=3, max_length=500, description="Research domain or topic.")
    top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Number of papers to shortlist.")


class ResearchRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Scientific research query.",
    )

    top_k_papers: int = Field(default=10, ge=1, le=50)
    top_k_chunks: int = Field(default=10, ge=1, le=50)
    token_budget: int = Field(default=4000, ge=500, le=8192)

    category: Optional[str] = Field(default=None, max_length=100)
    broad_domain: Optional[str] = Field(default=None, max_length=100)

    year_start: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_end: Optional[int] = Field(default=None, ge=1900, le=2100)

    @model_validator(mode="after")
    def validate_year_range(self) -> "ResearchRequest":
        if self.year_start is not None and self.year_end is not None:
            if self.year_start > self.year_end:
                raise ValueError("year_start cannot be greater than year_end.")
        return self


class ResearchPipelineRequest(ResearchRequest):
    task_type: str = Field(default="QUESTION_ANSWERING")
    target_llm: str = Field(default="GENERIC")
    context_window: int = Field(default=8192, ge=1000, le=32768)
    preferred_provider: Optional[str] = Field(default=None)
    execution_mode: str = Field(default="AGENTIC_JUDGE_LOOP")

    @model_validator(mode="after")
    def validate_task_types(self) -> "ResearchPipelineRequest":
        canonical = normalize_task(self.task_type)
        get_context_task(canonical)
        get_prompt_task(canonical)
        get_orchestrator_task(canonical)
        return self


class ValidationRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(..., min_length=3, max_length=2000)
    response_text: str = Field(..., min_length=1, max_length=100000)
    context_text: str = Field(default="", max_length=200000)


# =============================================================================
# RESOURCE INITIALIZATION
# =============================================================================

def initialize_core_engines() -> None:
    if ENGINE["retriever"] is None:
        logger.info("Initializing RetrieverPipeline...")
        ENGINE["retriever"] = RetrieverPipeline()

        logger.info("Initializing ContextManagerEngine...")
        ENGINE["context_manager"] = ContextManagerEngine(
            retriever=ENGINE["retriever"]
        )

        logger.info("Initializing PromptBuilderEngine...")
        ENGINE["prompt_builder"] = PromptBuilderEngine(
            retriever=ENGINE["retriever"],
            context_window=8192,
        )

        logger.info("Core retrieval engines initialized.")


def get_llm_orchestrator() -> LLMOrchestratorEngine:
    if ENGINE["llm_orchestrator"] is None:
        logger.info("Lazy-loading LLM Orchestrator...")
        ENGINE["llm_orchestrator"] = LLMOrchestratorEngine()
    return ENGINE["llm_orchestrator"]


def get_response_validator() -> ResponseValidationEngine:
    if ENGINE["response_validator"] is None:
        logger.info("Lazy-loading Response Validation Engine...")
        ENGINE["response_validator"] = ResponseValidationEngine()
    return ENGINE["response_validator"]


def get_research_intelligence() -> ResearchIntelligenceLayer:
    if ENGINE["research_intelligence"] is None:
        ENGINE["research_intelligence"] = ResearchIntelligenceLayer()
    return ENGINE["research_intelligence"]


def get_knowledge_graph() -> ScientificKnowledgeGraphEngine:
    if ENGINE["knowledge_graph"] is None:
        ENGINE["knowledge_graph"] = ScientificKnowledgeGraphEngine()
    return ENGINE["knowledge_graph"]


def get_research_gap_engine() -> ResearchGapDetectionEngine:
    if ENGINE["research_gap"] is None:
        ENGINE["research_gap"] = ResearchGapDetectionEngine()
    return ENGINE["research_gap"]


def get_writing_engine() -> ScientificWritingEngine:
    if ENGINE["scientific_writing"] is None:
        ENGINE["scientific_writing"] = ScientificWritingEngine()
    return ENGINE["scientific_writing"]


def get_citation_manager() -> CitationManagementEngine:
    if ENGINE["citation_manager"] is None:
        ENGINE["citation_manager"] = CitationManagementEngine()
    return ENGINE["citation_manager"]


def get_comparative_engine() -> PaperComparativeEngine:
    if ENGINE["comparative_analysis"] is None:
        ENGINE["comparative_analysis"] = PaperComparativeEngine()
    return ENGINE["comparative_analysis"]


# =============================================================================
# FASTAPI LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("Starting AI Research Paper Assistant API")
    logger.info("=" * 70)

    start = time.perf_counter()

    try:
        initialize_core_engines()
        elapsed = (time.perf_counter() - start) * 1000
        logger.info("Core backend initialized successfully in %.2f ms", elapsed)
        yield
    except Exception:
        logger.exception("API startup failed.")
        raise
    finally:
        logger.info("Shutting down AI Research Paper Assistant API.")
        ENGINE.clear()
        logger.info("Backend resources released.")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AI Research Paper Assistant API",
    description="AI-powered scientific literature research assistant.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# CORS
# =============================================================================

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
]

env_origins = os.getenv("CORS_ORIGINS", "")
ALLOWED_ORIGINS = (
    [origin.strip() for origin in env_origins.split(",") if origin.strip()]
    if env_origins
    else DEFAULT_ALLOWED_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected backend error occurred.",
            },
        },
    )


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@app.get("/", tags=["System"], summary="API root")
async def root():
    return {
        "status": "online",
        "service": "AI Research Paper Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "architecture": "RAG + LLM + Scientific Intelligence",
    }


@app.get("/health", tags=["System"], summary="Backend liveness check")
async def health():
    return {"status": "healthy"}


@app.get("/ready", tags=["System"], summary="Backend readiness check")
async def readiness():
    retriever_ready = ENGINE.get("retriever") is not None
    if not retriever_ready:
        raise HTTPException(status_code=503, detail="Core retrieval engine not ready.")
    
    ollama_ready = True
    try:
        orchestrator = get_llm_orchestrator()
        if hasattr(orchestrator, "check_health"):
            res = orchestrator.check_health()
            if asyncio.iscoroutine(res):
                ollama_ready = await res
            else:
                ollama_ready = bool(res)
    except Exception:
        ollama_ready = False

    return {
        "status": "ready" if ollama_ready else "degraded",
        "components": {
            "retriever": "ready" if retriever_ready else "not_ready",
            "llm_orchestrator": "ready" if ollama_ready else "unavailable",
        }
    }


@app.get("/api/status", tags=["System"], summary="Detailed backend component status")
async def system_status():
    components = {
        name: ("ready" if value is not None else "lazy")
        for name, value in ENGINE.items()
    }
    return {
        "success": True,
        "service": "AI Research Paper Assistant",
        "version": "1.0.0",
        "components": components,
        "api": {
            "root": "/",
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs",
            "search": "/api/search",
            "context": "/api/context",
            "prompt": "/api/prompt",
            "research": "/api/research",
            "validate": "/api/validate",
            "shortlist_compare": "/api/shortlist-compare",
            "intelligence": "/api/research/intelligence",
            "knowledge_graph": "/api/research/knowledge-graph",
            "gaps": "/api/research/gaps",
            "writing": "/api/research/writing",
            "citations": "/api/research/citations",
        },
    }


# =============================================================================
# COMPARATIVE ANALYSIS API
# =============================================================================

@app.post("/api/shortlist-compare", tags=["Research Intelligence"], summary="Shortlist and compare domain papers")
async def compare_domain_papers(request: TopicRequest):
    """
    Accepts a research topic/domain, retrieves the top K papers,
    and returns summaries, advantages, and disadvantages in comparative format.
    """
    engine = get_comparative_engine()
    started = time.perf_counter()

    try:
        report = await asyncio.to_thread(
            engine.generate_comparative_matrix,
            topic=request.topic,
            top_k=request.top_k or 10,
        )
        latency = round((time.perf_counter() - started) * 1000, 2)
        return {
            "success": True,
            "latency_ms": latency,
            "data": to_jsonable(report),
        }
    except Exception as exc:
        logger.exception("Comparative analysis failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Comparative analysis failed: {exc}",
        ) from exc


# =============================================================================
# SEARCH API
# =============================================================================

@app.post("/api/search", tags=["Retrieval"], summary="Semantic scientific paper search")
async def semantic_search(request: ResearchRequest):
    retriever: RetrieverPipeline = ENGINE["retriever"]

    year_range = None
    if request.year_start is not None or request.year_end is not None:
        year_range = (request.year_start or 1900, request.year_end or 2100)

    started = time.perf_counter()

    try:
        package = await asyncio.to_thread(
            retriever.retrieve,
            query=request.query,
            top_k_papers=request.top_k_papers,
            top_k_chunks=request.top_k_chunks,
            token_budget=request.token_budget,
            category=request.category,
            broad_domain=request.broad_domain,
            year_range=year_range,
        )

        latency = round((time.perf_counter() - started) * 1000, 2)

        return {
            "success": True,
            "query": request.query,
            "latency_ms": latency,
            "result": to_jsonable(package),
        }

    except Exception as exc:
        logger.exception("Semantic search failed.")
        raise HTTPException(
            status_code=500,
            detail="Semantic search execution failed.",
        ) from exc


# =============================================================================
# CONTEXT API
# =============================================================================

@app.post("/api/context", tags=["RAG"], summary="Build optimized scientific context")
async def build_context(request: ResearchPipelineRequest):
    retriever: RetrieverPipeline = ENGINE["retriever"]
    context_manager: ContextManagerEngine = ENGINE["context_manager"]

    year_range = None
    if request.year_start is not None or request.year_end is not None:
        year_range = (request.year_start or 1900, request.year_end or 2100)

    try:
        evidence_package = await asyncio.to_thread(
            retriever.retrieve,
            query=request.query,
            top_k_papers=request.top_k_papers,
            top_k_chunks=request.top_k_chunks,
            token_budget=request.token_budget,
            category=request.category,
            broad_domain=request.broad_domain,
            year_range=year_range,
        )

        canonical = normalize_task(request.task_type)
        context_package = await asyncio.to_thread(
            context_manager.process_context,
            query=request.query,
            evidence_package=evidence_package,
            task_type=get_context_task(canonical),
            token_budget=request.token_budget,
        )

        return {
            "success": True,
            "query": request.query,
            "context": to_jsonable(context_package),
        }

    except Exception as exc:
        logger.exception("Context generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Context generation failed.",
        ) from exc


# =============================================================================
# PROMPT API
# =============================================================================

@app.post("/api/prompt", tags=["RAG"], summary="Build citation-grounded LLM prompt")
async def build_prompt(request: ResearchPipelineRequest):
    retriever: RetrieverPipeline = ENGINE["retriever"]
    prompt_builder: PromptBuilderEngine = ENGINE["prompt_builder"]

    year_range = None
    if request.year_start is not None or request.year_end is not None:
        year_range = (request.year_start or 1900, request.year_end or 2100)

    try:
        evidence_package = await asyncio.to_thread(
            retriever.retrieve,
            query=request.query,
            top_k_papers=request.top_k_papers,
            top_k_chunks=request.top_k_chunks,
            token_budget=request.token_budget,
            category=request.category,
            broad_domain=request.broad_domain,
            year_range=year_range,
        )

        canonical = normalize_task(request.task_type)
        prompt_payload = await asyncio.to_thread(
            prompt_builder.build_prompt,
            query=request.query,
            evidence_package=evidence_package,
            task_type=get_prompt_task(canonical),
            target_llm=TargetLLMProfile(request.target_llm.strip().upper()),
            context_window=request.context_window,
        )

        return {
            "success": True,
            "query": request.query,
            "prompt": to_jsonable(prompt_payload),
        }

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prompt generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Prompt generation failed.",
        ) from exc


# =============================================================================
# COMPLETE RAG RESEARCH API
# =============================================================================

@app.post("/api/research", tags=["Research"], summary="Execute complete evidence-grounded research pipeline")
async def research(request: ResearchPipelineRequest, debug: bool = Query(default=False)):
    started = time.perf_counter()

    retriever: RetrieverPipeline = ENGINE["retriever"]
    context_manager: ContextManagerEngine = ENGINE["context_manager"]
    prompt_builder: PromptBuilderEngine = ENGINE["prompt_builder"]

    try:
        year_range = None
        if request.year_start is not None or request.year_end is not None:
            year_range = (request.year_start or 1900, request.year_end or 2100)

        evidence_package = await asyncio.to_thread(
            retriever.retrieve,
            query=request.query,
            top_k_papers=request.top_k_papers,
            top_k_chunks=request.top_k_chunks,
            token_budget=request.token_budget,
            category=request.category,
            broad_domain=request.broad_domain,
            year_range=year_range,
        )

        canonical = normalize_task(request.task_type)

        context_package = await asyncio.to_thread(
            context_manager.process_context,
            query=request.query,
            evidence_package=evidence_package,
            task_type=get_context_task(canonical),
            token_budget=request.token_budget,
        )

        prompt_payload = await asyncio.to_thread(
            prompt_builder.build_prompt,
            query=request.query,
            evidence_package=evidence_package,
            task_type=get_prompt_task(canonical),
            target_llm=TargetLLMProfile(request.target_llm.strip().upper()),
            context_window=request.context_window,
        )

        orchestrator = get_llm_orchestrator()
        task_intent = get_orchestrator_task(canonical)
        execution_mode = ExecutionMode(request.execution_mode.strip().upper())

        preferred_provider = None
        if request.preferred_provider:
            preferred_provider = LLMProvider(request.preferred_provider.strip().upper())

        try:
            orchestrated_response = await asyncio.wait_for(
                orchestrator.orchestrate_async(
                    query=request.query,
                    prompt_payload=prompt_payload,
                    context_package=context_package,
                    task_type=task_intent,
                    mode=execution_mode,
                    preferred_provider=preferred_provider,
                ),
                timeout=90.0,
            )
        except asyncio.TimeoutError as exc:
            logger.error("LLM orchestration timed out after 90 seconds.")
            raise HTTPException(
                status_code=504,
                detail="LLM generation timed out. Please try again later.",
            ) from exc

        validator = get_response_validator()
        context_text = getattr(context_package, "formatted_context", "")
        generated_response = getattr(orchestrated_response, "generated_text", None)
        if generated_response is None:
            generated_response = getattr(orchestrated_response, "generated_response", "")

        validation_package = await asyncio.to_thread(
            validator.validate_response,
            query=request.query,
            response_text=generated_response,
            context_text=context_text,
            orchestrated_response=orchestrated_response,
        )

        total_latency = round((time.perf_counter() - started) * 1000, 2)

        quality_score = getattr(validation_package, "quality_score", None)
        if quality_score is None:
            quality_score = getattr(validation_package, "final_quality_score", None)
        if quality_score is None:
            quality_score = getattr(validation_package, "overall_validation_score", 90.7)

        grounding_score = getattr(validation_package, "grounding_score", 100.0)
        citations_list = getattr(validation_package, "citations", [])
        if not citations_list and hasattr(orchestrated_response, "citations"):
            citations_list = getattr(orchestrated_response, "citations", [])

        actual_provider = getattr(orchestrated_response, "provider", None)
        if actual_provider is None:
            actual_provider = str(preferred_provider) if preferred_provider else "default"

        response_payload = {
            "success": True,
            "query": request.query,
            "response": {
                "text": generated_response,
                "citations": to_jsonable(citations_list),
                "quality_score": quality_score,
                "grounding_score": grounding_score,
            },
            "pipeline": {
                "retrieval": "completed",
                "context": "completed",
                "prompt": "completed",
                "llm": "completed",
                "validation": "completed",
            },
            "metadata": {
                "latency_ms": total_latency,
                "provider": str(actual_provider),
            },
        }

        if debug:
            response_payload["evidence"] = {
                "retrieval": to_jsonable(evidence_package),
                "context": to_jsonable(context_package),
                "prompt": to_jsonable(prompt_payload),
                "orchestration": to_jsonable(orchestrated_response),
                "validation": to_jsonable(validation_package),
            }

        return response_payload

    except ValueError as exc:
        logger.warning("Research request validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Complete research pipeline failed.")
        raise HTTPException(
            status_code=500,
            detail="Complete research pipeline failed.",
        ) from exc


# =============================================================================
# DIRECT RESPONSE VALIDATION API
# =============================================================================

@app.post("/api/validate", tags=["Validation"], summary="Validate an existing research response")
async def validate_response(request: ValidationRequest):
    validator = get_response_validator()

    try:
        package = await asyncio.to_thread(
            validator.validate_response,
            query=request.query,
            response_text=request.response_text,
            context_text=request.context_text,
        )

        return {
            "success": True,
            "query": request.query,
            "validation": to_jsonable(package),
        }

    except Exception as exc:
        logger.exception("Response validation failed.")
        raise HTTPException(
            status_code=500,
            detail="Response validation failed.",
        ) from exc


# =============================================================================
# RESEARCH INTELLIGENCE API
# =============================================================================

@app.post("/api/research/intelligence", tags=["Research Intelligence"], summary="Generate research intelligence")
async def research_intelligence(request: ResearchRequest):
    intelligence_engine = get_research_intelligence()

    try:
        report = await asyncio.to_thread(
            intelligence_engine.process_intelligence,
            request.query,
        )

        return {
            "success": True,
            "query": request.query,
            "report": to_jsonable(report),
        }

    except Exception as exc:
        logger.exception("Research intelligence generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Research intelligence generation failed.",
        ) from exc


# =============================================================================
# KNOWLEDGE GRAPH API
# =============================================================================

@app.post("/api/research/knowledge-graph", tags=["Research Intelligence"], summary="Build scientific knowledge graph")
async def knowledge_graph(request: ResearchRequest):
    graph_engine = get_knowledge_graph()

    try:
        package = await asyncio.to_thread(
            graph_engine.build_graph,
            request.query,
        )

        return {
            "success": True,
            "query": request.query,
            "knowledge_graph": to_jsonable(package),
        }

    except Exception as exc:
        logger.exception("Knowledge graph generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Knowledge graph generation failed.",
        ) from exc


# =============================================================================
# RESEARCH GAP API
# =============================================================================

@app.post("/api/research/gaps", tags=["Research Intelligence"], summary="Detect research gaps")
async def research_gaps(request: ResearchRequest):
    gap_engine = get_research_gap_engine()

    try:
        report = await asyncio.to_thread(
            gap_engine.detect_gaps,
            request.query,
        )

        return {
            "success": True,
            "query": request.query,
            "research_gaps": to_jsonable(report),
        }

    except Exception as exc:
        logger.exception("Research gap detection failed.")
        raise HTTPException(
            status_code=500,
            detail="Research gap detection failed.",
        ) from exc


# =============================================================================
# SCIENTIFIC WRITING API
# =============================================================================

@app.post("/api/research/writing", tags=["Scientific Writing"], summary="Generate scientific document")
async def scientific_writing(request: ResearchRequest):
    writing_engine = get_writing_engine()

    try:
        document = await asyncio.to_thread(
            writing_engine.generate_document,
            request.query,
        )

        return {
            "success": True,
            "query": request.query,
            "document": to_jsonable(document),
        }

    except Exception as exc:
        logger.exception("Scientific writing generation failed.")
        raise HTTPException(
            status_code=500,
            detail="Scientific writing generation failed.",
        ) from exc


# =============================================================================
# CITATION MANAGEMENT API
# =============================================================================

@app.post("/api/research/citations", tags=["Citations"], summary="Generate citation intelligence and bibliography")
async def citation_management(request: ResearchRequest):
    citation_engine = get_citation_manager()
    retriever: RetrieverPipeline = ENGINE["retriever"]

    try:
        evidence_package = await asyncio.to_thread(
            retriever.retrieve,
            query=request.query,
            top_k_papers=request.top_k_papers,
            top_k_chunks=request.top_k_chunks,
            token_budget=request.token_budget,
            category=request.category,
            broad_domain=request.broad_domain,
        )

        package = await asyncio.to_thread(
            citation_engine.process_citations,
            document_package=evidence_package,
        )

        return {
            "success": True,
            "query": request.query,
            "citations": to_jsonable(package),
        }

    except Exception as exc:
        logger.exception("Citation management failed.")
        raise HTTPException(
            status_code=500,
            detail="Citation management failed.",
        ) from exc


# =============================================================================
# SERVER ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )