"use client";

import Link from "next/link";
import { ChangeEvent, useEffect, useState } from "react";
import { api, loadPapers, WorkspacePaper } from "../../lib/api";
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
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

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
    setError("");
    setMessage("Uploading PDF...");
    const body = new FormData();
    body.append("file", file);

    try {
      const created = await api<{ job_id: string }>("/api/papers/upload", {
        method: "POST",
        body,
      });
      setMessage("Processing and extracting paper text...");
      await poll(created.job_id);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not upload paper.");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  async function poll(jobId: string) {
    for (let i = 0; i < 40; i++) {
      const job = await api<Job>(`/api/jobs/${jobId}`);
      if (job.status === "COMPLETED") {
        setMessage("Paper indexing complete and available for Q&A.");
        return;
      }
      if (job.status === "FAILED") {
        throw new Error(job.error || "Paper processing failed.");
      }
      setMessage(`Extracting text and embeddings (${job.progress || 20}%)...`);
      await new Promise(resolve => window.setTimeout(resolve, 1200));
    }
  }

  const filtered = papers.filter(
    p =>
      !filter.trim() ||
      p.title.toLowerCase().includes(filter.toLowerCase()) ||
      (p.authors_raw && p.authors_raw.toLowerCase().includes(filter.toLowerCase()))
  );

  return (
    <main className="shell">
      <AppHeader />

      <section className="tool-hero">
        <div className="eyebrow">Personal Library</div>
        <h1>My Papers</h1>
        <p className="panel-copy">
          Your persisted research library. Papers uploaded here are private to your account and
          indexed with vector embeddings for grounded Q&A.
        </p>
      </section>

      <div style={{ display: "flex", gap: 16, alignItems: "center", marginBottom: 20, flexWrap: "wrap" }}>
        <input
          type="text"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter papers by title or author..."
          style={{
            flex: 1,
            minWidth: 260,
            padding: "10px 14px",
            border: "1px solid var(--line)",
            borderRadius: "var(--radius-sm)",
            background: "var(--surface)",
            fontSize: 14,
          }}
        />

        <label className="primary small-button" style={{ cursor: "pointer" }}>
          {uploading ? "Processing PDF..." : "Upload New PDF"}
          <input type="file" accept="application/pdf" onChange={upload} disabled={uploading} style={{ display: "none" }} />
        </label>
      </div>

      {message && <div className="notice" role="status">{message}</div>}
      {error && <p className="error" role="alert">{error}</p>}

      <section className="collection-list" aria-live="polite">
        {loading ? (
          <LoadingSpinner message="Loading your paper library..." />
        ) : filtered.length ? (
          filtered.map(paper => (
            <article className="collection-row" key={paper.id}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <strong style={{ fontSize: 16 }}>{paper.title}</strong>
                  <StatusBadge state={paper.evidence_state} />
                </div>
                <div className="panel-copy" style={{ margin: 0 }}>
                  {paper.authors_raw || "Unknown authors"} · {paper.year || "Year n/a"} · {paper.venue || paper.source || "Paper"}
                </div>
              </div>
              <span style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <Link href={`/?paper=${paper.id}`} className="primary small-button" style={{ textDecoration: "none" }}>
                  Open Desk
                </Link>
              </span>
            </article>
          ))
        ) : (
          <EmptyState
            title="No papers found"
            description={filter ? "No papers match your filter criteria." : "You haven't uploaded or saved any papers yet."}
            icon="📚"
          />
        )}
      </section>
    </main>
  );
}
