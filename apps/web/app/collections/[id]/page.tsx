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
  const availablePapers = papers.filter(p => !selectedSet.has(p.id));

  return (
    <main className="shell">
      <AppHeader />

      <div style={{ margin: "16px 0 8px" }}>
        <Link href="/collections" className="link-button" style={{ fontSize: 13 }}>
          &larr; Back to all collections
        </Link>
      </div>

      <section className="tool-hero" style={{ padding: "20px 0 24px" }}>
        <div className="eyebrow">Project Collection</div>
        <h1>{collection?.name || "Loading collection..."}</h1>

        <form className="search" onSubmit={rename} style={{ maxWidth: 540, marginTop: 12 }}>
          <input
            value={name}
            onChange={event => setName(event.target.value)}
            placeholder="Rename collection..."
            aria-label="Collection name"
          />
          <button className="primary" disabled={busy || !name.trim()}>
            {busy ? "Saving..." : "Rename"}
          </button>
        </form>
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      {loading ? (
        <LoadingSpinner message="Loading collection papers..." />
      ) : (
        <div className="collection-detail-grid">
          <section>
            <div className="section-head">
              <h2>Included in Collection</h2>
              <span className="count">{collection?.papers.length ?? 0} papers</span>
            </div>

            <div style={{ display: "grid", gap: 10 }}>
              {collection?.papers.length ? (
                collection.papers.map(paper => (
                  <article className="collection-row" key={paper.id}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                        <strong>{paper.title}</strong>
                        <StatusBadge state={paper.evidence_state} />
                      </div>
                      <small style={{ color: "var(--muted)" }}>{paper.authors_raw || "Unknown authors"}</small>
                    </div>
                    <button
                      type="button"
                      className="link-button"
                      style={{ color: "var(--error)", fontSize: 13 }}
                      onClick={() => remove(paper.id)}
                      disabled={busy}
                    >
                      Remove
                    </button>
                  </article>
                ))
              ) : (
                <EmptyState
                  title="Empty collection"
                  description="Add papers from your library on the right to populate this collection."
                  icon="📄"
                />
              )}
            </div>
          </section>

          <section>
            <div className="section-head">
              <h2>Add from Library</h2>
              <span className="count">{availablePapers.length} available</span>
            </div>

            <div style={{ display: "grid", gap: 10 }}>
              {availablePapers.length ? (
                availablePapers.map(paper => (
                  <article className="collection-row" key={paper.id}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                        <strong>{paper.title}</strong>
                        <StatusBadge state={paper.evidence_state} />
                      </div>
                      <small style={{ color: "var(--muted)" }}>{paper.authors_raw || "Unknown authors"}</small>
                    </div>
                    <button
                      type="button"
                      className="primary small-button"
                      onClick={() => add(paper.id)}
                      disabled={busy}
                    >
                      Add to set
                    </button>
                  </article>
                ))
              ) : (
                <EmptyState
                  title="All library papers included"
                  description="Upload new papers or search literature to expand your collection."
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
