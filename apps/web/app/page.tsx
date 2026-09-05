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
  "retrieval augmented generation",
  "graph neural networks",
  "attention mechanisms",
  "diffusion models",
];

export default function Home() {
  const [topic, setTopic] = useState("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Paper | null>(null);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<Array<{ role: "user" | "assistant"; content: string; evidence?: Evidence[]; status?: string }>>([]);
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
              current ? { ...current, evidence_state: "full-text", analysis: job.result?.analysis ?? current.analysis } : current
            );
            setMessage("Paper extraction and vector indexing complete. Full-text Q&A is now active.");
            setJobId(null);
          } else if (job.status === "FAILED") {
            setError(job.error || "Paper processing failed.");
            setJobId(null);
          } else {
            setMessage(`Extracting & indexing document (${job.progress}%)...`);
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
      setPapers(data.papers);
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
      setError("Please select a PDF file.");
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
      const data = await api<Paper & { document_id: string; job_id: string; status: Job["status"] }>(
        "/api/papers/upload",
        { method: "POST", body }
      );
      setPapers([]);
      selectPaper({ ...data, source: "upload", evidence_state: "processing" });
      setJobId(data.job_id);
      setJobStatus(data.status);
      setMessage("Paper uploaded. Background worker is extracting pages and generating embeddings...");
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

      <section className="hero">
        <div className="eyebrow">Academic Intelligence Platform</div>
        <h1>Find the evidence<br />behind the paper.</h1>
        <p>
          Search scholarly literature across OpenAlex, explore verified citations, upload private PDFs,
          and ask questions grounded strictly in paper text.
        </p>

        <form className="search" onSubmit={e => search(e)}>
          <input
            value={topic}
            onChange={event => setTopic(event.target.value)}
            placeholder="Search topic: retrieval augmented generation, graph neural networks..."
            aria-label="Research topic"
          />
          <button className="primary" disabled={busy}>
            {busy ? "Searching..." : "Search literature"}
          </button>
        </form>

        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 600 }}>Suggested topics:</span>
          {SUGGESTED_QUERIES.map(q => (
            <button
              key={q}
              type="button"
              className="link-button"
              style={{ fontSize: 12, background: "var(--cream)", padding: "3px 10px", borderRadius: "var(--radius-sm)" }}
              onClick={() => {
                setTopic(q);
                search(undefined, q);
              }}
            >
              {q}
            </button>
          ))}
        </div>
      </section>

      <div className="layout">
        <section>
          <div className="section-head">
            <h2>{papers.length ? "Ranked Literature Results" : selected ? "Active Paper Workspace" : "Research Desk"}</h2>
            <span className="count">
              {papers.length ? `${papers.length} scholarly papers retrieved` : selected ? "Viewing selected paper" : "Ready for query"}
            </span>
          </div>

          {message && <div className="notice" role="status">{message}</div>}
          {error && <p className="error" role="alert">{error}</p>}

          {papers.length > 0 && (
            <div className="results">
              {papers.map((paper, index) => (
                <article
                  key={paper.id}
                  className={`paper ${selected?.id === paper.id ? "active" : ""}`}
                  onClick={() => selectPaper(paper)}
                >
                  <div className="paper-meta">
                    <span>{String(index + 1).padStart(2, "0")} / {paper.source || "Scholar"}</span>
                    <StatusBadge state={paper.evidence_state} />
                  </div>
                  <h3>{paper.title}</h3>
                  <p>{paper.analysis?.summary || paper.summary || "Abstract unavailable in this record."}</p>
                  <div className="paper-foot">
                    <span>{paper.first_author || paper.authors_raw || "Unknown author"}</span>
                    <span>{paper.venue || "Venue unspecified"}</span>
                    {paper.citation_count !== undefined && (
                      <span><strong>{paper.citation_count.toLocaleString()}</strong> citations</span>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}

          {!papers.length && !selected && !busy && (
            <EmptyState
              title="No papers loaded"
              description="Enter a research topic above to search OpenAlex and the scientific corpus, or upload a PDF on the right."
              icon="🔍"
            />
          )}

          {busy && !papers.length && (
            <LoadingSpinner message="Searching scholarly literature databases..." size={36} />
          )}

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
            />
          )}
        </section>

        <aside className="panel">
          <h2>Private Document</h2>
          <p className="panel-copy">
            Upload an academic PDF for asynchronous extraction, chunking, and isolated evidence-backed Q&A.
          </p>

          <label className="upload">
            <strong>{busy ? "Uploading Document..." : "Upload Paper PDF"}</strong>
            <span className="panel-copy" style={{ margin: "4px 0 0" }}>
              Magic bytes verified · Max 25 MB
            </span>
            <input type="file" accept="application/pdf" onChange={upload} disabled={busy} />
          </label>

          {jobStatus && jobStatus !== "COMPLETED" && (
            <div style={{ marginTop: 14 }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                <span style={{ fontWeight: 600 }}>Extraction progress</span>
                <span>{jobProgress}%</span>
              </div>
              <div style={{ background: "var(--line)", height: 6, borderRadius: 3, overflow: "hidden" }}>
                <div style={{ width: `${jobProgress}%`, height: "100%", background: "var(--teal)", transition: "width 0.3s" }} />
              </div>
            </div>
          )}

          <div style={{ marginTop: 24, borderTop: "1px solid var(--line)", paddingTop: 16 }}>
            <h4 style={{ margin: "0 0 8px", fontSize: 12, textTransform: "uppercase", color: "var(--teal)" }}>
              Citation Policy
            </h4>
            <p style={{ fontSize: 12, color: "var(--muted)", margin: 0, lineHeight: 1.5 }}>
              Lumen Research strictly requires type-specific evidence before answering. If a claim cannot be verified from the extracted text, the platform explicitly abstains.
            </p>
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
}) {
  const analysis = paper.analysis;

  return (
    <section className="detail">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div className="eyebrow">
          {paper.source === "upload" ? "Private Library Paper" : "Scholarly Literature Match"}
        </div>
        <StatusBadge state={paper.evidence_state} />
      </div>

      <h2 style={{ fontSize: 24, margin: "0 0 10px" }}>{paper.title}</h2>
      <p style={{ fontSize: 13, color: "var(--muted)", margin: "0 0 16px" }}>
        {paper.authors_raw || "Unknown authors"} · {paper.venue || "Venue unspecified"} {paper.published_date ? `· ${paper.published_date}` : ""}
      </p>

      {analysis && (
        <div className="detail-grid">
          <div className="detail-block">
            <h4>Summary</h4>
            <p>{analysis.summary}</p>
          </div>
          <div className="detail-block">
            <h4>Methodology</h4>
            <p>{analysis.advantages || analysis.method || "Extracted from full paper text."}</p>
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
        <div className="notice" style={{ marginTop: 18 }}>
          <strong>Processing Document ({jobProgress}%):</strong> Extracting pages, generating vector embeddings, and structuring sections. Q&A will activate once indexing completes.
        </div>
      )}

      <div style={{ marginTop: 28 }}>
        <h3 style={{ fontSize: 18, font: "700 18px var(--font-serif)", margin: "0 0 12px" }}>
          Grounded Evidence Q&A
        </h3>

        {chatHistory.length > 0 && (
          <div style={{ display: "grid", gap: 12, marginBottom: 18 }}>
            {chatHistory.map((item, idx) => {
              const isAssistant = item.role === "assistant";
              const isInsufficient = item.status === "insufficient-evidence";
              return (
                <div
                  key={idx}
                  className={`answer-card ${isInsufficient ? "insufficient" : ""}`}
                  style={{
                    borderLeft: isAssistant ? (isInsufficient ? "4px solid var(--error)" : "4px solid var(--success)") : "4px solid var(--teal)",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <strong style={{ fontSize: 12, textTransform: "uppercase", color: isAssistant ? "var(--teal)" : "var(--muted)" }}>
                      {isAssistant ? "Lumen Verified Assistant" : "Researcher Query"}
                    </strong>
                    {isAssistant && (
                      <span className={`badge ${isInsufficient ? "badge-failed" : "badge-full-text"}`}>
                        {isInsufficient ? "Abstained (Insufficient)" : "Evidence-backed"}
                      </span>
                    )}
                  </div>

                  <p style={{ margin: "4px 0", whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{item.content}</p>

                  {item.evidence && item.evidence.length > 0 && (
                    <EvidenceDrawer evidence={item.evidence} />
                  )}
                </div>
              );
            })}
          </div>
        )}

        <form className="qa" onSubmit={ask}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder={canAsk ? "Ask an empirical question (e.g. dataset, methodology, metrics, results)..." : "Waiting for document indexing..."}
            disabled={!canAsk || asking}
            aria-label="Ask paper a question"
          />
          <button className="primary" disabled={!canAsk || asking}>
            {asking ? "Checking Evidence..." : "Ask"}
          </button>
        </form>
      </div>
    </section>
  );
}
