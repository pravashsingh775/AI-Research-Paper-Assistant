"use client";

import { FormEvent, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { api, loadPapers, WorkspacePaper } from "../../../lib/api";
import AppHeader from "../../../components/AppHeader";
import StatusBadge from "../../../components/StatusBadge";
import EmptyState from "../../../components/EmptyState";
import LoadingSpinner from "../../../components/LoadingSpinner";

type Collection = { id: string; name: string; papers: WorkspacePaper[] };

export default function CollectionDetail() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const id = params.id;

  const [collection, setCollection] = useState<Collection | null>(null);
  const [papers, setPapers] = useState<WorkspacePaper[]>([]);
  const [name, setName] = useState("");
  const [libraryFilter, setLibraryFilter] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);

  async function load() {
    try {
      const [current, available] = await Promise.all([
        api<Collection>(`/api/collections/${id}`),
        loadPapers(),
      ]);
      setCollection(current);
      setName(current.name);
      setPapers(available);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to load collection");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!localStorage.getItem("research_token")) {
      router.replace("/login");
      return;
    }
    load();
  }, [id, router]);

  async function rename(event: FormEvent) {
    event.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api(`/api/collections/${id}`, { method: "PATCH", body: JSON.stringify({ name: name.trim() }) });
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to rename collection");
    } finally {
      setBusy(false);
    }
  }

  async function add(paperId: string) {
    setBusy(true);
    try {
      await api(`/api/collections/${id}/papers`, { method: "POST", body: JSON.stringify({ paper_id: paperId }) });
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to add paper");
    } finally {
      setBusy(false);
    }
  }

  async function remove(paperId: string) {
    setBusy(true);
    try {
      await api(`/api/collections/${id}/papers/${paperId}`, { method: "DELETE" });
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to remove paper");
    } finally {
      setBusy(false);
    }
  }

  const selectedSet = new Set(collection?.papers.map(p => p.id));
  const availablePapers = papers.filter(
    p =>
      !selectedSet.has(p.id) &&
      (!libraryFilter.trim() ||
        p.title.toLowerCase().includes(libraryFilter.toLowerCase()) ||
        (p.authors_raw && p.authors_raw.toLowerCase().includes(libraryFilter.toLowerCase())))
  );

  return (
    <main className="shell">
      <AppHeader />

      {/* Back Link */}
      <div style={{ margin: "16px 0 8px" }}>
        <Link
          href="/collections"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            fontSize: 13,
            fontWeight: 600,
            color: "var(--primary-600)",
            textDecoration: "none",
          }}
        >
          <span>&larr;</span>
          <span>Back to all collections</span>
        </Link>
      </div>

      {/* Hero / Rename Section */}
      <section className="tool-hero" style={{ padding: "16px 0 24px" }}>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>📁</span> COLLECTION WORKSPACE
        </div>
        <h1 style={{ fontSize: "clamp(1.8rem, 3vw, 2.4rem)", marginTop: 6, marginBottom: 14 }}>
          {collection?.name || "Loading Collection..."}
        </h1>

        {/* Rename Form */}
        <form onSubmit={rename} style={{ display: "flex", gap: 10, maxWidth: 520 }}>
          <input
            type="text"
            value={name}
            onChange={event => setName(event.target.value)}
            placeholder="Rename collection..."
            aria-label="Collection name"
            style={{
              flex: 1,
              padding: "8px 14px",
              fontSize: 13,
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-sm)",
              background: "var(--surface)",
              color: "var(--ink)",
            }}
          />
          <button
            type="submit"
            className="secondary small-button"
            disabled={busy || !name.trim() || name === collection?.name}
            style={{
              padding: "8px 14px",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {busy ? "Saving..." : "Rename"}
          </button>
        </form>
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      {loading ? (
        <div style={{ padding: "80px 0" }}>
          <LoadingSpinner message="Loading collection papers..." />
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))",
            gap: 24,
            alignItems: "start",
          }}
        >
          {/* Column 1: Papers in Collection */}
          <section
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "20px 24px",
              boxShadow: "var(--shadow-sm)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <h2 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700, color: "var(--ink)" }}>
                  In this Collection
                </h2>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "9999px",
                    background: "var(--primary-50)",
                    color: "var(--primary-700)",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {collection?.papers.length ?? 0}
                </span>
              </div>
            </div>

            <div style={{ display: "grid", gap: 12 }}>
              {collection?.papers.length ? (
                collection.papers.map(paper => (
                  <article
                    key={paper.id}
                    style={{
                      border: "1px solid var(--line)",
                      borderRadius: "var(--radius-sm)",
                      padding: "14px 16px",
                      background: "var(--surface)",
                      boxShadow: "var(--shadow-sm)",
                      display: "flex",
                      flexDirection: "column",
                      gap: 8,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: "0 0 4px", fontSize: 14, fontWeight: 700 }}>
                          <Link href={`/?paper=${paper.id}`} style={{ color: "var(--ink)", textDecoration: "none" }}>
                            {paper.title}
                          </Link>
                        </h4>
                        <div style={{ fontSize: 12, color: "var(--ink-muted)" }}>
                          {paper.authors_raw || "Unknown authors"} · {paper.year || "n/a"}
                        </div>
                      </div>
                      <StatusBadge state={paper.evidence_state} />
                    </div>

                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        borderTop: "1px solid var(--slate-100)",
                        paddingTop: 8,
                      }}
                    >
                      <Link
                        href={`/?paper=${paper.id}`}
                        style={{ fontSize: 12, fontWeight: 600, color: "var(--primary-600)", textDecoration: "none" }}
                      >
                        Open Desk &rarr;
                      </Link>
                      <button
                        type="button"
                        onClick={() => remove(paper.id)}
                        disabled={busy}
                        style={{
                          background: "none",
                          border: "none",
                          color: "var(--error)",
                          fontSize: 12,
                          fontWeight: 600,
                          cursor: "pointer",
                          padding: "2px 6px",
                        }}
                      >
                        Remove from set
                      </button>
                    </div>
                  </article>
                ))
              ) : (
                <EmptyState
                  title="Empty Collection"
                  description="No papers in this collection yet. Add papers from your library on the right."
                  icon="📄"
                />
              )}
            </div>
          </section>

          {/* Column 2: Available from Library */}
          <section
            style={{
              background: "var(--surface)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "20px 24px",
              boxShadow: "var(--shadow-sm)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <h2 style={{ margin: 0, fontSize: "1.1rem", fontWeight: 700, color: "var(--ink)" }}>
                  Add from Library
                </h2>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "9999px",
                    background: "var(--slate-100)",
                    color: "var(--ink-muted)",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {availablePapers.length}
                </span>
              </div>
            </div>

            {/* Quick search filter for library papers */}
            <div style={{ marginBottom: 14 }}>
              <input
                type="text"
                value={libraryFilter}
                onChange={e => setLibraryFilter(e.target.value)}
                placeholder="Search library papers..."
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  fontSize: 13,
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--slate-50)",
                  color: "var(--ink)",
                }}
              />
            </div>

            <div style={{ display: "grid", gap: 12 }}>
              {availablePapers.length ? (
                availablePapers.map(paper => (
                  <article
                    key={paper.id}
                    style={{
                      border: "1px solid var(--line)",
                      borderRadius: "var(--radius-sm)",
                      padding: "14px 16px",
                      background: "var(--surface)",
                      boxShadow: "var(--shadow-sm)",
                      display: "flex",
                      flexDirection: "column",
                      gap: 8,
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
                      <div style={{ flex: 1 }}>
                        <h4 style={{ margin: "0 0 4px", fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>
                          {paper.title}
                        </h4>
                        <div style={{ fontSize: 12, color: "var(--ink-muted)" }}>
                          {paper.authors_raw || "Unknown"} · {paper.year || "n/a"}
                        </div>
                      </div>
                      <StatusBadge state={paper.evidence_state} />
                    </div>

                    <div
                      style={{
                        display: "flex",
                        justifyContent: "flex-end",
                        borderTop: "1px solid var(--slate-100)",
                        paddingTop: 8,
                      }}
                    >
                      <button
                        type="button"
                        className="primary small-button"
                        onClick={() => add(paper.id)}
                        disabled={busy}
                        style={{
                          padding: "6px 14px",
                          fontSize: 12,
                          fontWeight: 600,
                          display: "inline-flex",
                          alignItems: "center",
                          gap: 4,
                        }}
                      >
                        <span>+ Add to set</span>
                      </button>
                    </div>
                  </article>
                ))
              ) : (
                <EmptyState
                  title="All papers included"
                  description={
                    libraryFilter
                      ? `No library papers matched "${libraryFilter}".`
                      : "All papers in your library are currently added to this collection."
                  }
                  icon="✨"
                />
              )}
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
