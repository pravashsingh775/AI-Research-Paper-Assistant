"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";
import AppHeader from "../components/AppHeader";
import StatusBadge from "../components/StatusBadge";
import CitationCard from "../components/CitationCard";
import EvidenceDrawer from "../components/EvidenceDrawer";
import LoadingSpinner from "../components/LoadingSpinner";
import EmptyState from "../components/EmptyState";
import { api } from "../lib/api";

type Analysis = {
  summary: string;
  strengths: string[];
  weaknesses: string[];
  advantages: string;
  disadvantages: string;
  method: string;
  evidence?: string[] | string;
  confidence?: number;
};

type Paper = {
  id: string;
  document_id?: string | null;
  title: string;
  summary?: string;
  authors_raw?: string;
  first_author?: string;
  published_date?: string;
  venue?: string;
  citation_count?: number;
  source?: string;
  evidence_state?: "metadata-only" | "full-text" | "private-upload" | "processing" | "failed";
  analysis?: Analysis;
  processing_status?: Job["status"];
  doi?: string | null;
  url?: string | null;
};

type Job = {
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  progress: number;
  error?: string | null;
  result?: { analysis?: Analysis };
};

type Evidence = {
  page?: number | null;
  section?: string | null;
  chunk_index?: number;
  text?: string;
};

type QaResponse = {
  answer: string;
  evidence: Evidence[];
  status: "evidence-backed" | "insufficient-evidence";
  validation?: { status: string; question_type?: string; reranker?: string };
};

type ChatResponse = {
  answer: string;
  session_id: string;
  citations: Evidence[];
  evidence?: Evidence[];
  status: "evidence-backed" | "insufficient-evidence";
  evidence_state?: string;
};

const SUGGESTED_QUERIES = [
  { label: "Quantum Machine Learning", icon: "⚛️" },
  { label: "Retrieval Augmented Generation", icon: "⚡" },
  { label: "Graph Neural Networks", icon: "🕸️" },
  { label: "Diffusion Models", icon: "🧬" },
  { label: "Attention Mechanisms", icon: "🧠" },
];

export default function Home() {
  const [topic, setTopic] = useState("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<
    Array<{ role: "user" | "assistant"; content: string; evidence?: Evidence[]; status?: string }>
  >([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [asking, setAsking] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [jobStatus, setJobStatus] = useState<Job["status"] | null>(null);
  const [jobProgress, setJobProgress] = useState(0);
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);

  const isPrivateUpload = selected?.source === "upload";
  const canAsk = Boolean(
    selected && (!isPrivateUpload || jobStatus === "COMPLETED" || selected.evidence_state === "full-text")
  );

  function selectPaper(paper: Paper) {
    setSelected(paper);
    setQuestion("");
    setChatHistory([]);
    setError("");
    setChatSessionId(null);

    // Hydrate full paper details from server to ensure latest analysis & status
    if (paper.id) {
      api<Paper>(`/api/papers/${paper.id}`)
        .then(fullPaper => {
          setSelected(prev => (prev?.id === paper.id ? { ...prev, ...fullPaper } : prev));
          if (fullPaper.processing_status) setJobStatus(fullPaper.processing_status);
        })
        .catch(() => {});
    }
  }

  useEffect(() => {
    const paperId = new URLSearchParams(window.location.search).get("paper");
    if (!paperId) return;
    api<Paper>(`/api/papers/${paperId}`)
      .then(paper => {
        selectPaper(paper);
        setJobStatus(paper.processing_status ?? null);
      })
      .catch(reason => setError(reason instanceof Error ? reason.message : "Unable to load paper."));
  }, []);

  useEffect(() => {
    if (!jobId) return;
    const poll = window.setInterval(() => {
      api<Job>(`/api/jobs/${jobId}`)
        .then(job => {
          setJobStatus(job.status);
          setJobProgress(job.progress);
          if (job.status === "COMPLETED") {
            setSelected(current =>
              current
                ? {
                    ...current,
                    evidence_state: "full-text",
                    analysis: job.result?.analysis ?? current.analysis,
                  }
                : current
            );
            setMessage("Paper extraction and vector indexing complete. Full-text Q&A is now active.");
            setJobId(null);
          } else if (job.status === "FAILED") {
            setError(job.error || "Paper processing failed.");
            setJobId(null);
          } else {
            setMessage(`Extracting pages & generating embeddings (${job.progress}%)...`);
          }
        })
        .catch(reason => {
          setError(reason instanceof Error ? reason.message : "Unable to read processing status.");
          setJobId(null);
        });
    }, 1200);
    return () => window.clearInterval(poll);
  }, [jobId]);

  async function search(event?: FormEvent, queryText?: string) {
    if (event) event.preventDefault();
    const query = queryText || topic;
    if (query.trim().length < 2) return;
    setBusy(true);
    setError("");
    setMessage("");
    setSelected(null);
    setChatHistory([]);
    setChatSessionId(null);

    try {
      const data = await api<{ papers: Paper[]; notice?: string }>("/api/search", {
        method: "POST",
        body: JSON.stringify({ topic: query.trim() }),
      });
      setPapers(data.papers || []);
      setMessage(data.notice || "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Search failed.");
    } finally {
      setBusy(false);
    }
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.endsWith(".pdf")) {
      setError("Please select a valid PDF file.");
      return;
    }
    setBusy(true);
    setError("");
    setMessage("Uploading PDF...");
    setQuestion("");
    setChatHistory([]);
    setChatSessionId(null);
    setJobStatus("PENDING");
    setJobProgress(5);

    const body = new FormData();
    body.append("file", file);

    try {
      const data = await api<Paper & { id: string; document_id: string; job_id: string; status: Job["status"] }>(
        "/api/papers/upload",
        { method: "POST", body }
      );
      setPapers([]);
      selectPaper({ ...data, source: "upload", evidence_state: "processing" });
      setJobId(data.job_id);
      setJobStatus(data.status);
      setMessage("Paper uploaded. Background worker is extracting pages and indexing chunks...");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Upload failed.");
      setJobStatus(null);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }

  async function ask(event: FormEvent) {
    event.preventDefault();
    if (!selected || !question.trim() || !canAsk) return;

    const q = question.trim();
    setAsking(true);
    setError("");
    setChatHistory(prev => [...prev, { role: "user", content: q }]);
    setQuestion("");

    try {
      if (isPrivateUpload && selected.document_id) {
        const resp = await api<QaResponse>("/api/qa", {
          method: "POST",
          body: JSON.stringify({ document_id: selected.document_id, question: q }),
        });
        setChatHistory(prev => [
          ...prev,
          { role: "assistant", content: resp.answer, evidence: resp.evidence, status: resp.status },
        ]);
      } else {
        const resp = await api<ChatResponse>("/api/chat", {
          method: "POST",
          body: JSON.stringify({ paper_id: selected.id, session_id: chatSessionId, question: q }),
        });
        setChatSessionId(resp.session_id);
        const evidenceItems = resp.evidence || resp.citations || [];
        setChatHistory(prev => [
          ...prev,
          { role: "assistant", content: resp.answer, evidence: evidenceItems, status: resp.status },
        ]);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Question failed.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <main className="shell">
      <AppHeader />

      {/* Hero Section */}
      <section className="hero">
        <div className="eyebrow">
          <span>✨ Next-Gen Research Intelligence</span>
          <span>·</span>
          <span>pgvector & arXiv Grounded</span>
        </div>
        <h1>
          Discover Scholarly Evidence. <br />
          <span className="text-gradient">Uncover Verified Truth.</span>
        </h1>
        <p>
          Search millions of peer-reviewed papers across OpenAlex and arXiv, upload private PDFs,
          and ask complex empirical questions with mathematically verified citations.
        </p>

        {/* Command Search Bar */}
        <form className="search" onSubmit={e => search(e)}>
          <span className="search-icon-prefix">🔍</span>
          <input
            value={topic}
            onChange={event => setTopic(event.target.value)}
            placeholder="Search topic or keywords: quantum machine learning, graph neural networks..."
            aria-label="Research topic"
          />
          <span className="shortcut-badge">↵ Enter</span>
          <button className="primary" disabled={busy}>
            {busy ? "Searching..." : "Search literature"}
          </button>
        </form>

        {/* Suggested Queries */}
        <div className="suggestion-chips">
          <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 600 }}>Suggested:</span>
          {SUGGESTED_QUERIES.map(q => (
            <button
              key={q.label}
              type="button"
              className="chip-btn"
              onClick={() => {
                setTopic(q.label);
                search(undefined, q.label);
              }}
            >
              <span>{q.icon}</span>
              <span>{q.label}</span>
            </button>
          ))}
        </div>
      </section>

      {/* Split Workspace Layout */}
      <div className="layout">
        {/* Left Column: Results or Paper Workspace */}
        <section>
          <div className="section-head">
            <h2>
              {papers.length
                ? "Ranked Literature Results"
                : selected
                ? "Active Paper Workspace"
                : "Research Desk"}
            </h2>
            <span className="count">
              {papers.length
                ? `${papers.length} scholarly papers retrieved`
                : selected
                ? "Viewing selected paper"
                : "Ready for query"}
            </span>
          </div>

          {message && <div className="notice" role="status">{message}</div>}
          {error && <p className="error" role="alert">{error}</p>}

          {/* Results List */}
          {papers.length > 0 && (
            <div className="results">
              {papers.map((paper, index) => (
                <article
                  key={paper.id}
                  className={`paper ${selected?.id === paper.id ? "active" : ""}`}
                  onClick={() => selectPaper(paper)}
                >
                  <div className="paper-meta">
                    <span className="paper-source-tag">
                      <strong>#{String(index + 1).padStart(2, "0")}</strong>
                      <span>·</span>
                      <span>{paper.source ? paper.source.toUpperCase() : "SCHOLAR"}</span>
                      {paper.published_date && (
                        <>
                          <span>·</span>
                          <span>{paper.published_date.slice(0, 4)}</span>
                        </>
                      )}
                    </span>
                    <StatusBadge state={paper.evidence_state} />
                  </div>
                  <h3>{paper.title}</h3>
                  <p>{paper.analysis?.summary || paper.summary || "Abstract unavailable in this record."}</p>
                  <div className="paper-foot">
                    <span className="paper-foot-author">
                      👤 {paper.first_author || paper.authors_raw || "Unknown author"}
                    </span>
                    <span>🏛️ {paper.venue || "Venue unspecified"}</span>
                    {paper.citation_count !== undefined && (
                      <span className="citation-count-badge" title="Verified citation count">
                        ⭐ {paper.citation_count.toLocaleString()} citations
                      </span>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}

          {/* Empty State */}
          {!papers.length && !selected && !busy && (
            <EmptyState
              title="No papers loaded yet"
              description="Enter a research topic above to search OpenAlex and arXiv, or upload a private PDF on the right to start."
              icon="📚"
            />
          )}

          {/* Loading State */}
          {busy && !papers.length && (
            <LoadingSpinner message="Searching scholarly literature databases..." size={38} />
          )}

          {/* Active Workspace */}
          {selected && (
            <PaperWorkspace
              paper={selected}
              question={question}
              setQuestion={setQuestion}
              chatHistory={chatHistory}
              ask={ask}
              asking={asking}
              canAsk={canAsk}
              isProcessing={isPrivateUpload && jobStatus !== "COMPLETED" && selected.evidence_state === "processing"}
              jobProgress={jobProgress}
              onBack={() => setSelected(null)}
            />
          )}
        </section>

        {/* Right Column: Upload & Citation Policy */}
        <aside className="panel">
          <h2>Private Document</h2>
          <p className="panel-copy">
            Upload an academic PDF for asynchronous extraction, vector chunking, and isolated evidence-backed Q&A.
          </p>

          <label className="upload">
            <div className="upload-icon-box">☁️</div>
            <strong>{busy ? "Uploading Document..." : "Upload Research PDF"}</strong>
            <span style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
              Magic bytes verified · Max 25 MB
            </span>
            <input type="file" accept="application/pdf" onChange={upload} disabled={busy} />
          </label>

          {/* Real-Time Processing Progress */}
          {jobStatus && jobStatus !== "COMPLETED" && (
            <div style={{ marginTop: 18 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 6 }}>
                <span style={{ fontWeight: 700, color: "var(--ink)" }}>Extracting & Indexing</span>
                <span style={{ fontWeight: 700, color: "var(--primary)" }}>{jobProgress}%</span>
              </div>
              <div style={{ background: "var(--border)", height: 7, borderRadius: 4, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${jobProgress}%`,
                    height: "100%",
                    background: "var(--primary-gradient)",
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
            </div>
          )}

          {/* Citation Policy Reassurance */}
          <div style={{ marginTop: 24, borderTop: "1px solid var(--border)", paddingTop: 18 }}>
            <h4 style={{ margin: "0 0 10px", fontSize: 12, textTransform: "uppercase", color: "var(--primary)", letterSpacing: "0.06em" }}>
              Citation Policy
            </h4>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 12, color: "var(--ink-secondary)" }}>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                <span>✅</span>
                <span>Type-specific evidence required before answering</span>
              </div>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                <span>✅</span>
                <span>Zero hallucination: strictly abstains on unmentioned facts</span>
              </div>
              <div style={{ display: "flex", alignItems: "flex-start", gap: 6 }}>
                <span>✅</span>
                <span>Multi-tenant isolated storage in encrypted MinIO</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}

function PaperWorkspace({
  paper,
  question,
  setQuestion,
  chatHistory,
  ask,
  asking,
  canAsk,
  isProcessing,
  jobProgress,
  onBack,
}: {
  paper: Paper;
  question: string;
  setQuestion: (val: string) => void;
  chatHistory: Array<{ role: "user" | "assistant"; content: string; evidence?: Evidence[]; status?: string }>;
  ask: (e: FormEvent) => void;
  asking: boolean;
  canAsk: boolean;
  isProcessing: boolean;
  jobProgress: number;
  onBack: () => void;
}) {
  const [activeTab, setActiveTab] = useState<"overview" | "chat">("chat");
  const analysis = paper.analysis;

  return (
    <section className="detail">
      {/* Top Header & Back Button */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <button
          type="button"
          className="secondary small-button"
          onClick={onBack}
          title="Back to search results"
        >
          ← Back to Results
        </button>
        <StatusBadge state={paper.evidence_state} />
      </div>

      <h2 style={{ fontSize: 22, fontWeight: 800, margin: "0 0 8px", color: "var(--ink)" }}>
        {paper.title}
      </h2>
      <p style={{ fontSize: 13, color: "var(--muted)", margin: "0 0 18px" }}>
        👤 {paper.authors_raw || "Unknown authors"} · 🏛️ {paper.venue || "Venue unspecified"}{" "}
        {paper.published_date ? `· 📅 ${paper.published_date}` : ""}
      </p>

      {/* Tab Controls */}
      <div style={{ display: "flex", gap: 8, borderBottom: "1px solid var(--border)", paddingBottom: 12, marginBottom: 18 }}>
        <button
          type="button"
          className={`nav-link ${activeTab === "chat" ? "active" : ""}`}
          onClick={() => setActiveTab("chat")}
        >
          💬 Grounded AI Chat
        </button>
        <button
          type="button"
          className={`nav-link ${activeTab === "overview" ? "active" : ""}`}
          onClick={() => setActiveTab("overview")}
        >
          📑 Structured Overview
        </button>
      </div>

      {/* Overview Tab */}
      {activeTab === "overview" && analysis && (
        <div className="detail-grid">
          <div className="detail-block">
            <h4>Summary</h4>
            <p>{analysis.summary}</p>
          </div>
          <div className="detail-block">
            <h4>Methodology</h4>
            <p>{analysis.method || analysis.advantages || "Extracted from paper text."}</p>
          </div>
          {analysis.strengths?.length > 0 && (
            <div className="detail-block">
              <h4>Reported Strengths</h4>
              <ul>
                {analysis.strengths.map((s, idx) => (
                  <li key={idx}>{s}</li>
                ))}
              </ul>
            </div>
          )}
          {analysis.weaknesses?.length > 0 && (
            <div className="detail-block">
              <h4>Limitations</h4>
              <ul>
                {analysis.weaknesses.map((w, idx) => (
                  <li key={idx}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {isProcessing && (
        <div className="notice" style={{ margin: "16px 0" }}>
          <strong>Processing Document ({jobProgress}%):</strong> Extracting pages, generating vector embeddings, and structuring sections. Q&A will activate once indexing completes.
        </div>
      )}

      {/* Chat Tab */}
      {activeTab === "chat" && (
        <div>
          {chatHistory.length > 0 ? (
            <div className="chat-container">
              {chatHistory.map((item, idx) => {
                const isAssistant = item.role === "assistant";
                const isInsufficient = item.status === "insufficient-evidence";
                return (
                  <div
                    key={idx}
                    className={`chat-bubble ${isAssistant ? "chat-assistant" : "chat-user"}`}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                      <strong style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", opacity: 0.85 }}>
                        {isAssistant ? "🤖 Lumen Research Assistant" : "👤 You"}
                      </strong>
                      {isAssistant && (
                        <span className={`badge ${isInsufficient ? "badge-failed" : "badge-full-text"}`}>
                          {isInsufficient ? "Abstained (Insufficient)" : "Evidence-Backed"}
                        </span>
                      )}
                    </div>

                    <p style={{ margin: "4px 0", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{item.content}</p>

                    {item.evidence && item.evidence.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <EvidenceDrawer evidence={item.evidence} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "28px 16px", color: "var(--muted)", fontSize: 14 }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>💡</div>
              <strong>Ask any empirical question about this paper</strong>
              <p style={{ fontSize: 13, margin: "4px 0 0" }}>
                Ask about datasets, empirical accuracy, baseline models, mathematical formulas, or limitations.
              </p>
            </div>
          )}

          {/* Question Input Form */}
          <form className="qa" onSubmit={ask} style={{ marginTop: 16 }}>
            <input
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder={
                canAsk
                  ? "Ask an empirical question grounded in this paper (e.g., What dataset was used?)..."
                  : "Waiting for document indexing..."
              }
              disabled={!canAsk || asking}
              aria-label="Ask paper a question"
            />
            <button className="primary" disabled={!canAsk || asking}>
              {asking ? "Checking Evidence..." : "Ask AI"}
            </button>
          </form>
        </div>
      )}
    </section>
  );
}
