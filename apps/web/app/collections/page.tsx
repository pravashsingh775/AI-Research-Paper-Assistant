"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api } from "../../lib/api";
import AppHeader from "../../components/AppHeader";
import EmptyState from "../../components/EmptyState";
import LoadingSpinner from "../../components/LoadingSpinner";

type Collection = { id: string; name: string; papers?: any[] };

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
    if (!window.confirm(`Delete collection "${colName}"? Papers in your library will not be deleted.`)) return;
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

      {/* Hero Section */}
      <section className="tool-hero" style={{ paddingBottom: 16 }}>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>📁</span> PROJECT WORKSPACES
        </div>
        <h1 style={{ fontSize: "clamp(2rem, 3.5vw, 2.75rem)", marginTop: 8 }}>Research Collections</h1>
        <p className="panel-copy" style={{ maxWidth: 640 }}>
          Group related literature into curated topic collections. Perfect for literature reviews,
          specialized domain benchmarks, and comparative evidence synthesis.
        </p>
      </section>

      {/* Create New Collection Card */}
      <div
        style={{
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: "var(--radius-md)",
          padding: "20px 24px",
          boxShadow: "var(--shadow-sm)",
          marginTop: 16,
          marginBottom: 32,
        }}
      >
        <h3 style={{ margin: "0 0 12px", fontSize: 15, fontWeight: 700, color: "var(--ink)" }}>
          Create New Collection
        </h3>
        <form onSubmit={create} style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="e.g. Graph Transformer Benchmarks, Diffusion for Drug Discovery..."
            aria-label="Collection name"
            style={{
              flex: 1,
              minWidth: 280,
              padding: "10px 16px",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-sm)",
              background: "var(--surface)",
              color: "var(--ink)",
              fontSize: 14,
            }}
          />
          <button
            type="submit"
            className="primary"
            disabled={busy || !name.trim()}
            style={{
              padding: "10px 20px",
              fontSize: 14,
              fontWeight: 600,
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span>{busy ? "Creating..." : "+ Create Collection"}</span>
          </button>
        </form>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {/* Collections Grid */}
      <section aria-live="polite">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ margin: 0, fontSize: "1.15rem", fontWeight: 700, color: "var(--ink)" }}>
            Your Collections ({items.length})
          </h2>
        </div>

        {loading ? (
          <div style={{ padding: "60px 0" }}>
            <LoadingSpinner message="Loading your collections..." />
          </div>
        ) : items.length ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
              gap: 16,
            }}
          >
            {items.map(item => (
              <article
                key={item.id}
                style={{
                  background: "var(--surface)",
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-md)",
                  padding: "20px",
                  boxShadow: "var(--shadow-sm)",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  gap: 16,
                  transition: "transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease",
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
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                    <span
                      style={{
                        width: 36,
                        height: 36,
                        borderRadius: "var(--radius-sm)",
                        background: "var(--primary-50)",
                        color: "var(--primary)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 18,
                      }}
                    >
                      📁
                    </span>
                    <div>
                      <h3 style={{ margin: 0, fontSize: "1.05rem", fontWeight: 700, color: "var(--ink)" }}>
                        <Link href={`/collections/${item.id}`} style={{ color: "inherit", textDecoration: "none" }}>
                          {item.name}
                        </Link>
                      </h3>
                      <span style={{ fontSize: 12, color: "var(--ink-muted)" }}>
                        Project working set
                      </span>
                    </div>
                  </div>
                </div>

                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    borderTop: "1px solid var(--line)",
                    paddingTop: 12,
                  }}
                >
                  <Link
                    href={`/collections/${item.id}`}
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
                    <span>Open Workspace</span>
                    <span>&rarr;</span>
                  </Link>

                  <button
                    type="button"
                    onClick={() => remove(item.id, item.name)}
                    style={{
                      background: "none",
                      border: "none",
                      color: "var(--error)",
                      fontSize: 13,
                      fontWeight: 500,
                      cursor: "pointer",
                      padding: "4px 8px",
                      borderRadius: "var(--radius-sm)",
                    }}
                  >
                    Delete
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No collections created yet"
            description="Create your first collection above to organize papers for systematic reviews or comparative synthesis."
            icon="📁"
          />
        )}
      </section>
    </main>
  );
}
