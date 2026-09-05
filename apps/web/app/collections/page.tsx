"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api } from "../../lib/api";
import AppHeader from "../../components/AppHeader";
import EmptyState from "../../components/EmptyState";
import LoadingSpinner from "../../components/LoadingSpinner";

type Collection = { id: string; name: string };

export default function Collections() {
  const router = useRouter();
  const [items, setItems] = useState<Collection[]>([]);
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);

  async function load() {
    try {
      setItems(await api<Collection[]>("/api/collections"));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load collections");
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
  }, [router]);

  async function create(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setBusy(true);
    setError("");
    try {
      await api("/api/collections", { method: "POST", body: JSON.stringify({ name: name.trim() }) });
      setName("");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to create collection");
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string, colName: string) {
    if (!window.confirm(`Delete collection "${colName}"?`)) return;
    try {
      await api(`/api/collections/${id}`, { method: "DELETE" });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to delete collection");
    }
  }

  return (
    <main className="shell">
      <AppHeader />

      <section className="tool-hero">
        <div className="eyebrow">Workspace Collections</div>
        <h1>Research Collections</h1>
        <p className="panel-copy">
          Curate and organize related papers into focused project collections for synthesis and comparison.
        </p>
      </section>

      <form className="search" onSubmit={create} style={{ maxWidth: 640 }}>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="New collection name (e.g. Graph Transformer Benchmarks)..."
          aria-label="Collection name"
        />
        <button className="primary" disabled={busy || !name.trim()}>
          {busy ? "Creating..." : "Create collection"}
        </button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}

      <div className="collection-list" style={{ marginTop: 28 }}>
        {loading ? (
          <LoadingSpinner message="Loading your collections..." />
        ) : items.length ? (
          items.map(item => (
            <article className="collection-row" key={item.id}>
              <div>
                <strong style={{ fontSize: 16 }}>{item.name}</strong>
                <div className="panel-copy" style={{ margin: "2px 0 0" }}>Project working set</div>
              </div>
              <span>
                <Link href={`/collections/${item.id}`} className="primary small-button" style={{ textDecoration: "none" }}>
                  Open Collection
                </Link>
                <button type="button" className="link-button" style={{ color: "var(--error)", fontSize: 13 }} onClick={() => remove(item.id, item.name)}>
                  Delete
                </button>
              </span>
            </article>
          ))
        ) : (
          <EmptyState
            title="No collections yet"
            description="Create your first collection above to organize your literature reviews."
            icon="📁"
          />
        )}
      </div>
    </main>
  );
}
