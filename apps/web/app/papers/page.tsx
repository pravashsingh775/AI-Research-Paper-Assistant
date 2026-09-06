"use client";

import Link from "next/link";
import { ChangeEvent, useEffect, useState } from "react";
import { api, loadPapers, deletePaper, WorkspacePaper } from "../../lib/api";
import AppHeader from "../../components/AppHeader";
import StatusBadge from "../../components/StatusBadge";
import EmptyState from "../../components/EmptyState";
import LoadingSpinner from "../../components/LoadingSpinner";

type Job = {
  id: string;
  status: string;
  progress: number;
  error?: string | null;
  result?: { paper_id?: string };
};

export default function PapersPage() {
  const [papers, setPapers] = useState<WorkspacePaper[]>([]);
  const [filter, setFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "full-text" | "metadata-only">("all");
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function refresh() {
    setError("");
    try {
      setPapers(await loadPapers());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load papers.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!localStorage.getItem("research_token")) {
      window.location.href = "/login";
      return;
    }
    refresh();
  }, []);

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.endsWith(".pdf")) {
      setError("Please select a valid PDF file.");
      return;
    }
    setUploading(true);
    setUploadProgress(10);
    setError("");
    setMessage("Uploading PDF document...");
    const body = new FormData();
    body.append("file", file);

    try {
      const created = await api<{ job_id: string }>("/api/papers/upload", {
        method: "POST",
        body,
      });
      setMessage("Extracting text, sections, and computing embeddings...");
      await poll(created.job_id);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not upload paper.");
    } finally {
      setUploading(false);
      setUploadProgress(0);
      event.target.value = "";
    }
  }

  async function poll(jobId: string) {
    for (let i = 0; i < 40; i++) {
      const job = await api<Job>(`/api/jobs/${jobId}`);
      setUploadProgress(job.progress || Math.min(20 + i * 2, 95));
      if (job.status === "COMPLETED") {
        setUploadProgress(100);
        setMessage("Paper indexing complete and ready for grounded research!");
        setTimeout(() => setMessage(""), 4000);
        return;
      }
      if (job.status === "FAILED") {
        throw new Error(job.error || "Paper processing failed.");
      }
      setMessage(`Extracting text and embeddings (${job.progress || 20}%)...`);
      await new Promise(resolve => window.setTimeout(resolve, 1200));
    }
  }

  async function handleDelete(id: string, title: string) {
    if (!window.confirm(`Delete "${title}" from your library?`)) return;
    setDeletingId(id);
    try {
      await deletePaper(id);
      setPapers(prev => prev.filter(p => p.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete paper.");
    } finally {
      setDeletingId(null);
    }
  }

  // Filter logic
  const filtered = papers.filter(p => {
    const matchesSearch =
      !filter.trim() ||
      p.title.toLowerCase().includes(filter.toLowerCase()) ||
      (p.authors_raw && p.authors_raw.toLowerCase().includes(filter.toLowerCase())) ||
      (p.venue && p.venue.toLowerCase().includes(filter.toLowerCase()));

    const matchesStatus =
      statusFilter === "all" ||
      (statusFilter === "full-text" && (p.evidence_state?.toLowerCase() === "full-text" || p.evidence_state?.toLowerCase() === "full_text")) ||
      (statusFilter === "metadata-only" && p.evidence_state?.toLowerCase() !== "full-text" && p.evidence_state?.toLowerCase() !== "full_text");

    return matchesSearch && matchesStatus;
  });

  // Calculate statistics
  const totalPapers = papers.length;
  const fullTextCount = papers.filter(
    p => p.evidence_state?.toLowerCase() === "full-text" || p.evidence_state?.toLowerCase() === "full_text"
  ).length;
  const totalCitations = papers.reduce((sum, p) => sum + (p.citation_count || 0), 0);

  return (
    <main className="shell">
      <AppHeader />

      {/* Hero Header */}
      <section className="tool-hero" style={{ paddingBottom: 20 }}>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 13 }}>📚</span> WORKSPACE LIBRARY
        </div>
        <h1 style={{ fontSize: "clamp(2rem, 3.5vw, 2.75rem)", marginTop: 8 }}>My Research Papers</h1>
        <p className="panel-copy" style={{ maxWidth: 640 }}>
          Your persisted research library. Upload research PDFs for vector embeddings and grounded citation Q&amp;A,
          or organize papers discovered across literature databases.
        </p>

        {/* Stats Row */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: 16,
            marginTop: 24,
            marginBottom: 8,
          }}
        >
          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              boxShadow: "var(--shadow-sm)",
              display: "flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: "var(--radius-sm)",
                background: "var(--primary-50)",
                color: "var(--primary-600)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 22,
              }}
            >
              📄
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Total Papers
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "var(--ink)", lineHeight: 1.2 }}>
                {loading ? "—" : totalPapers}
              </div>
            </div>
          </div>

          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              boxShadow: "var(--shadow-sm)",
              display: "flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: "var(--radius-sm)",
                background: "var(--success-bg)",
                color: "var(--success)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 22,
              }}
            >
              🔬
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Full-Text Verified
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "var(--ink)", lineHeight: 1.2 }}>
                {loading ? "—" : fullTextCount}
                <span style={{ fontSize: 13, fontWeight: 500, color: "var(--ink-muted)", marginLeft: 6 }}>
                  ({totalPapers > 0 ? Math.round((fullTextCount / totalPapers) * 100) : 0}%)
                </span>
              </div>
            </div>
          </div>

          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              boxShadow: "var(--shadow-sm)",
              display: "flex",
              alignItems: "center",
              gap: 16,
            }}
          >
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: "var(--radius-sm)",
                background: "#fef3c7",
                color: "#b45309",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 22,
              }}
            >
              📈
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Total Citations
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "var(--ink)", lineHeight: 1.2 }}>
                {loading ? "—" : totalCitations.toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Upload Progress Bar if Uploading */}
      {uploading && (
        <div
          style={{
            background: "var(--primary-50)",
            border: "1px solid var(--primary-100)",
            borderRadius: "var(--radius-md)",
            padding: "16px 20px",
            marginBottom: 24,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 14, fontWeight: 600, color: "var(--primary-700)" }}>
            <span>{message || "Processing paper PDF..."}</span>
            <span>{uploadProgress}%</span>
          </div>
          <div style={{ height: 6, background: "var(--primary-100)", borderRadius: 3, overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                background: "var(--primary)",
                width: `${uploadProgress}%`,
                transition: "width 0.3s ease",
              }}
            />
          </div>
        </div>
      )}

      {/* Control Bar (Search, Status Filter, Upload Button) */}
      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 24,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", gap: 10, flex: 1, minWidth: 280, alignItems: "center" }}>
          <div style={{ position: "relative", flex: 1 }}>
            <span
              style={{
                position: "absolute",
                left: 14,
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--ink-muted)",
                fontSize: 15,
                pointerEvents: "none",
              }}
            >
              🔍
            </span>
            <input
              type="text"
              value={filter}
              onChange={e => setFilter(e.target.value)}
              placeholder="Search by title, author, venue..."
              style={{
                width: "100%",
                padding: "10px 14px 10px 38px",
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                background: "var(--surface)",
                fontSize: 14,
                color: "var(--ink)",
                boxShadow: "var(--shadow-sm)",
              }}
            />
            {filter && (
              <button
                type="button"
                onClick={() => setFilter("")}
                style={{
                  position: "absolute",
                  right: 12,
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--ink-muted)",
                  fontSize: 13,
                }}
              >
                ✕
              </button>
            )}
          </div>

          {/* Filter Pills */}
          <div style={{ display: "flex", gap: 6 }}>
            <button
              type="button"
              onClick={() => setStatusFilter("all")}
              style={{
                padding: "8px 14px",
                borderRadius: "9999px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                border: statusFilter === "all" ? "1px solid var(--primary)" : "1px solid var(--line)",
                background: statusFilter === "all" ? "var(--primary-50)" : "var(--surface)",
                color: statusFilter === "all" ? "var(--primary-700)" : "var(--ink-muted)",
                transition: "all var(--speed-fast)",
              }}
            >
              All ({papers.length})
            </button>
            <button
              type="button"
              onClick={() => setStatusFilter("full-text")}
              style={{
                padding: "8px 14px",
                borderRadius: "9999px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                border: statusFilter === "full-text" ? "1px solid var(--success)" : "1px solid var(--line)",
                background: statusFilter === "full-text" ? "var(--success-bg)" : "var(--surface)",
                color: statusFilter === "full-text" ? "var(--success)" : "var(--ink-muted)",
                transition: "all var(--speed-fast)",
              }}
            >
              Full-Text ({fullTextCount})
            </button>
          </div>
        </div>

        {/* Upload Button */}
        <label
          className="primary small-button"
          style={{
            cursor: uploading ? "not-allowed" : "pointer",
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "10px 18px",
            fontSize: 14,
            fontWeight: 600,
            boxShadow: "var(--shadow-sm)",
          }}
        >
          <span>{uploading ? "⏳" : "⬆️"}</span>
          <span>{uploading ? "Extracting PDF..." : "Upload Paper PDF"}</span>
          <input
            type="file"
            accept="application/pdf"
            onChange={upload}
            disabled={uploading}
            style={{ display: "none" }}
          />
        </label>
      </div>

      {message && !uploading && <div className="notice" role="status">{message}</div>}
      {error && <p className="error" role="alert">{error}</p>}

      {/* Paper List Section */}
      <section aria-live="polite">
        {loading ? (
          <LoadingSpinner message="Loading your paper library..." />
        ) : filtered.length ? (
          <div style={{ display: "grid", gap: 14 }}>
            {filtered.map(paper => (
              <article
                key={paper.id}
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-md)",
                  padding: "18px 22px",
                  boxShadow: "var(--shadow-sm)",
                  transition: "transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease",
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.borderColor = "var(--primary-200)";
                  e.currentTarget.style.boxShadow = "var(--shadow-md)";
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.borderColor = "var(--line)";
                  e.currentTarget.style.boxShadow = "var(--shadow-sm)";
                }}
              >
                {/* Meta Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <StatusBadge state={paper.evidence_state} />
                    {paper.year && (
                      <span
                        style={{
                          fontSize: 12,
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: 4,
                          background: "var(--slate-100)",
                          color: "var(--ink-secondary)",
                        }}
                      >
                        {paper.year}
                      </span>
                    )}
                    {paper.venue && (
                      <span
                        style={{
                          fontSize: 12,
                          fontWeight: 500,
                          color: "var(--ink-muted)",
                        }}
                      >
                        {paper.venue}
                      </span>
                    )}
                    {paper.citation_count !== undefined && paper.citation_count > 0 && (
                      <span
                        style={{
                          fontSize: 12,
                          fontWeight: 600,
                          color: "var(--primary-600)",
                        }}
                      >
                        📊 {paper.citation_count} citations
                      </span>
                    )}
                  </div>

                  {/* Actions */}
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <Link
                      href={`/?paper=${paper.id}`}
                      className="primary small-button"
                      style={{
                        textDecoration: "none",
                        padding: "6px 14px",
                        fontSize: 13,
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      <span>Open Desk</span>
                      <span style={{ fontSize: 14 }}>&rarr;</span>
                    </Link>
                    <button
                      type="button"
                      onClick={() => handleDelete(paper.id, paper.title)}
                      disabled={deletingId === paper.id}
                      style={{
                        background: "transparent",
                        border: "none",
                        cursor: "pointer",
                        color: "var(--error)",
                        fontSize: 13,
                        padding: "6px 8px",
                        borderRadius: "var(--radius-sm)",
                      }}
                      title="Remove from library"
                    >
                      {deletingId === paper.id ? "Deleting..." : "Delete"}
                    </button>
                  </div>
                </div>

                {/* Title */}
                <div>
                  <h3
                    style={{
                      margin: 0,
                      fontSize: "1.1rem",
                      fontWeight: 700,
                      color: "var(--ink)",
                      lineHeight: 1.4,
                    }}
                  >
                    <Link
                      href={`/?paper=${paper.id}`}
                      style={{ textDecoration: "none", color: "inherit" }}
                    >
                      {paper.title}
                    </Link>
                  </h3>
                </div>

                {/* Authors */}
                <div style={{ fontSize: 13, color: "var(--ink-secondary)", display: "flex", alignItems: "center", gap: 6 }}>
                  <span>✍️</span>
                  <span>{paper.authors_raw || "Unknown authors"}</span>
                </div>

                {/* Summary / Abstract Snippet */}
                {paper.summary && (
                  <p
                    style={{
                      margin: 0,
                      fontSize: 13,
                      lineHeight: 1.6,
                      color: "var(--ink-secondary)",
                      display: "-webkit-box",
                      WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                    }}
                  >
                    {paper.summary}
                  </p>
                )}
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title={filter ? "No matching papers found" : "Your library is empty"}
            description={
              filter
                ? `No papers matched "${filter}". Try adjusting your search keywords.`
                : "Upload your research PDFs or discover literature from the home workspace to build your library."
            }
            icon="📚"
            actionText={filter ? "Clear filter" : "Explore Literature"}
            onAction={filter ? () => setFilter("") : () => (window.location.href = "/")}
          />
        )}
      </section>
    </main>
  );
}
