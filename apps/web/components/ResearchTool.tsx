"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, loadPapers, WorkspacePaper } from "../lib/api";
import AppHeader from "./AppHeader";
import StatusBadge from "./StatusBadge";
import LoadingSpinner from "./LoadingSpinner";
import EmptyState from "./EmptyState";

type Tool = "compare" | "trends" | "gaps" | "ideas" | "proposal" | "similarity";

const labels: Record<Tool, string> = {
  compare: "Paper Comparison Matrix",
  trends: "Research Trends & Trajectory",
  gaps: "Research Gaps & Evidence Coverage",
  ideas: "Research Directions & Ideas",
  proposal: "Proposal Generator",
  similarity: "Paper Similarity Network",
};

const descriptions: Record<Tool, string> = {
  compare: "Synthesize structured differences, evaluation protocols, and dataset usage across selected papers.",
  trends: "Inspect chronological publication trends, emerging keywords, and domain evolution.",
  gaps: "Detect methodology gaps, missing evidence coverage, and evaluation discrepancies.",
  ideas: "Generate grounded, hypothesis-driven research ideas based on collective findings.",
  proposal: "Draft a formal academic research proposal with problem statement, methodology, and citations.",
  similarity: "Compute semantic pairwise similarities across paper representations.",
};

export default function ResearchTool({ tool }: { tool: Tool }) {
  const router = useRouter();
  const [papers, setPapers] = useState<WorkspacePaper[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [topic, setTopic] = useState("");
  const [filterQuery, setFilterQuery] = useState("");
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!localStorage.getItem("research_token")) {
      router.replace("/login");
      return;
    }
    loadPapers()
      .then(setPapers)
      .catch(reason => setError(reason instanceof Error ? reason.message : "Unable to load papers"))
      .finally(() => setLoading(false));
  }, [router]);

  function toggle(id: string) {
    setSelected(current =>
      current.includes(id) ? current.filter(item => item !== id) : current.length < 15 ? [...current, id] : current
    );
  }

  function selectAll() {
    const filteredIds = filteredPapers.map(p => p.id);
    setSelected(Array.from(new Set([...selected, ...filteredIds])).slice(0, 15));
  }

  function clearSelected() {
    setSelected([]);
  }

  const filteredPapers = papers.filter(p =>
    !filterQuery.trim() ||
    p.title.toLowerCase().includes(filterQuery.toLowerCase()) ||
    (p.authors_raw && p.authors_raw.toLowerCase().includes(filterQuery.toLowerCase()))
  );

  async function submit(event: FormEvent) {
    event.preventDefault();
    const minRequired = tool === "trends" ? 1 : 2;
    if (selected.length < minRequired) {
      setError(`Please select at least ${minRequired} papers to perform this analysis.`);
      return;
    }
    setBusy(true);
    setError("");

    const chosen = papers
      .filter(paper => selected.includes(paper.id))
      .map(paper => ({
        id: paper.id,
        title: paper.title,
        summary: paper.summary || "",
        authors: paper.authors_raw || "",
        year: paper.year,
        venue: paper.venue,
        citation_count: paper.citation_count || 0,
      }));

    const path =
      tool === "compare"
        ? "/api/compare"
        : tool === "trends"
        ? "/api/trends"
        : tool === "gaps"
        ? "/api/research-gaps"
        : tool === "ideas"
        ? "/api/research-ideas"
        : tool === "proposal"
        ? "/api/proposals"
        : "/api/similarity-map";

    try {
      const data = await api<Record<string, any>>(path, {
        method: "POST",
        body: JSON.stringify({
          topic: topic.trim() || "Selected Research Collection",
          papers: chosen,
        }),
      });
      setResult(data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <AppHeader />

      <section className="tool-hero">
        <div className="eyebrow">Research Intelligence</div>
        <h1>{labels[tool]}</h1>
        <p>{descriptions[tool]}</p>
      </section>

      <form className="tool-layout" onSubmit={submit}>
        <section>
          <label className="field-label">
            <span>Topic Focus (optional)</span>
            <input
              value={topic}
              onChange={event => setTopic(event.target.value)}
              placeholder="e.g. Graph representation learning, RAG benchmarks..."
            />
          </label>

          <div style={{ marginTop: 20, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>
              Select Papers ({selected.length} selected)
            </span>
            <div style={{ display: "flex", gap: 10 }}>
              <button type="button" className="link-button" style={{ fontSize: 12 }} onClick={selectAll}>
                Select all
              </button>
              <button type="button" className="link-button" style={{ fontSize: 12 }} onClick={clearSelected}>
                Clear
              </button>
            </div>
          </div>

          <div style={{ marginTop: 8 }}>
            <input
              type="text"
              value={filterQuery}
              onChange={e => setFilterQuery(e.target.value)}
              placeholder="Filter library papers..."
              style={{
                width: "100%",
                padding: "8px 12px",
                fontSize: 13,
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                background: "var(--surface)",
              }}
            />
          </div>

          <div className="paper-picker">
            {loading ? (
              <LoadingSpinner message="Loading your papers..." />
            ) : filteredPapers.length ? (
              filteredPapers.map(paper => {
                const isChecked = selected.includes(paper.id);
                return (
                  <label key={paper.id} className={`picker-row ${isChecked ? "chosen" : ""}`}>
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => toggle(paper.id)}
                      style={{ marginTop: 3 }}
                    />
                    <span style={{ flex: 1 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                        <strong>{paper.title}</strong>
                        <StatusBadge state={paper.evidence_state} />
                      </div>
                      <small>
                        {paper.authors_raw || "Unknown authors"} · {paper.year || "n/a"} · {paper.venue || paper.source || "Paper"}
                      </small>
                      {paper.summary && (
                        <small style={{ display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                          {paper.summary}
                        </small>
                      )}
                    </span>
                  </label>
                );
              })
            ) : (
              <EmptyState
                title="No matching papers"
                description="Upload a paper or discover papers from search to add to your library."
                actionText="Go to Discovery"
                onAction={() => router.push("/")}
              />
            )}
          </div>

          <button className="primary tool-submit" disabled={busy || loading}>
            {busy ? "Synthesizing Evidence..." : `Run ${labels[tool]}`}
          </button>

          {error && <p className="error">{error}</p>}
        </section>

        <aside className="result-panel">
          {busy ? (
            <LoadingSpinner message="Computing evidence synthesis..." size={36} />
          ) : result ? (
            <ResultViewer tool={tool} data={result} />
          ) : (
            <EmptyState
              title="Awaiting analysis"
              description="Select papers on the left and run the tool to generate evidence-backed research insights."
              icon="📊"
            />
          )}
        </aside>
      </form>
    </main>
  );
}

function ResultViewer({ tool, data }: { tool: Tool; data: Record<string, any> }) {
  if (tool === "compare" && data.papers) {
    return (
      <div>
        <div className="eyebrow">Comparison Matrix</div>
        <h3 style={{ margin: "0 0 14px", font: "700 18px var(--font-serif)" }}>{data.topic}</h3>

        {data.key_similarities && data.key_similarities.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <h5 style={{ margin: "0 0 6px", fontSize: 11, textTransform: "uppercase", color: "var(--teal)" }}>
              Shared Key Concepts & Topics
            </h5>
            <div className="tag-cloud">
              {data.key_similarities.map((item: string) => (
                <span key={item} className="tag-pill">
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="matrix-table-wrap">
          <table className="matrix-table">
            <thead>
              <tr>
                <th>Paper</th>
                <th>Year</th>
                <th>Venue</th>
                <th>Citations</th>
                <th>Key Evidence</th>
              </tr>
            </thead>
            <tbody>
              {data.papers.map((p: any, idx: number) => (
                <tr key={idx}>
                  <td><strong>{p.paper}</strong><br /><small style={{ color: "var(--muted)" }}>{p.authors}</small></td>
                  <td>{p.year || "—"}</td>
                  <td>{p.venue || "—"}</td>
                  <td>{p.citations ?? "—"}</td>
                  <td style={{ fontSize: 12, color: "var(--ink-secondary)", maxWidth: 300 }}>{p.evidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data.key_differences && (
          <div className="detail-block" style={{ marginTop: 14 }}>
            <h4>Key Differences & Divergent Findings</h4>
            <p>{data.key_differences}</p>
          </div>
        )}

        {data.potential_research_opportunity && (
          <div className="notice" style={{ marginTop: 14 }}>
            <strong>Research Opportunity:</strong> {data.potential_research_opportunity}
          </div>
        )}
      </div>
    );
  }

  if (tool === "trends" && data.publication_trend) {
    const years = Object.entries(data.publication_trend).sort(([a], [b]) => a.localeCompare(b));
    return (
      <div>
        <div className="eyebrow">Research Trajectory</div>
        <h3 style={{ margin: "0 0 14px", font: "700 18px var(--font-serif)" }}>{data.topic}</h3>
        <p style={{ fontSize: 13, color: "var(--muted)", margin: "0 0 12px" }}>
          Analyzed across {data.paper_count} papers in collection.
        </p>

        <h5 style={{ margin: "16px 0 8px", fontSize: 11, textTransform: "uppercase", color: "var(--teal)" }}>
          Publication Timeline
        </h5>
        <div className="trend-grid">
          {years.map(([year, count]) => (
            <div key={year} className="trend-card">
              <div className="trend-year">{year}</div>
              <div className="trend-count">{String(count)}</div>
            </div>
          ))}
        </div>

        {data.emerging_keywords && (
          <div style={{ marginTop: 20 }}>
            <h5 style={{ margin: "0 0 8px", fontSize: 11, textTransform: "uppercase", color: "var(--teal)" }}>
              High-Frequency Keywords & Methods
            </h5>
            <div className="tag-cloud">
              {data.emerging_keywords.map((kw: string) => (
                <span key={kw} className="tag-pill">
                  #{kw}
                </span>
              ))}
            </div>
          </div>
        )}

        {data.notice && (
          <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 18, fontStyle: "italic" }}>
            {data.notice}
          </p>
        )}
      </div>
    );
  }

  if (tool === "gaps") {
    return (
      <div>
        <div className="eyebrow">Gap Analysis</div>
        <h3 style={{ margin: "0 0 14px", font: "700 18px var(--font-serif)" }}>{data.topic}</h3>

        {data.evidence_based_findings?.map((f: any, idx: number) => (
          <div key={idx} className="detail-block" style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
              <h4>{f.category || "Evidence Coverage Gap"}</h4>
              <span className="badge badge-failed">Confidence {Math.round((f.confidence ?? 1) * 100)}%</span>
            </div>
            <p>{f.finding}</p>
            {f.affected_papers?.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 11, textTransform: "uppercase", color: "var(--muted)" }}>Affected Papers:</strong>
                <ul style={{ margin: "4px 0 0", paddingLeft: 16, fontSize: 12 }}>
                  {f.affected_papers.map((p: string, i: number) => (
                    <li key={i}>{p}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        {data.ai_generated_hypotheses?.map((h: any, idx: number) => (
          <div key={idx} className="notice" style={{ marginTop: 12 }}>
            <strong>Hypothesis ({h.category}):</strong> {h.finding}
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>Evidence: {h.evidence}</div>
          </div>
        ))}
      </div>
    );
  }

  if (tool === "ideas") {
    return (
      <div>
        <div className="eyebrow">Novel Research Directions</div>
        <h3 style={{ margin: "0 0 6px", font: "700 18px var(--font-serif)" }}>{data.topic}</h3>
        <p style={{ fontSize: 12, color: "var(--teal)", fontStyle: "italic", margin: "0 0 16px" }}>{data.label}</p>

        <div style={{ display: "grid", gap: 14 }}>
          {data.ideas?.map((idea: any, idx: number) => (
            <div key={idx} className="detail-block">
              <h4 style={{ color: "var(--ink)", fontSize: 15, textTransform: "none", letterSpacing: "normal" }}>
                {idx + 1}. {idea.title}
              </h4>
              <div style={{ margin: "8px 0" }}>
                <strong style={{ fontSize: 12, color: "var(--teal)", textTransform: "uppercase" }}>Unresolved Problem:</strong>
                <p style={{ margin: "2px 0 8px" }}>{idea.problem}</p>
              </div>
              <div style={{ margin: "8px 0" }}>
                <strong style={{ fontSize: 12, color: "var(--success)", textTransform: "uppercase" }}>Proposed Contribution:</strong>
                <p style={{ margin: "2px 0 8px" }}>{idea.proposed_contribution}</p>
              </div>
              {idea.evidence?.length > 0 && (
                <div style={{ borderTop: "1px solid var(--line-subtle)", paddingTop: 8, marginTop: 8 }}>
                  <span style={{ fontSize: 11, color: "var(--muted)", textTransform: "uppercase", fontWeight: 700 }}>
                    Grounding Papers:
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                    {idea.evidence.map((title: string, i: number) => (
                      <span key={i} className="citation-chip" style={{ fontSize: 10 }}>
                        {title}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (tool === "proposal") {
    return (
      <div className="proposal-sheet">
        <div className="eyebrow">{data.label || "Draft Research Proposal"}</div>
        <h2>{data.title}</h2>

        <div className="proposal-section">
          <h4>1. Problem Statement</h4>
          <p>{data.problem_statement}</p>
        </div>

        <div className="proposal-section">
          <h4>2. Proposed Methodology</h4>
          <p>{data.methodology}</p>
        </div>

        <div className="proposal-section">
          <h4>3. Evaluation Protocol</h4>
          <p>{data.evaluation}</p>
        </div>

        {data.references?.length > 0 && (
          <div className="proposal-section" style={{ borderTop: "1px solid var(--line)", paddingTop: 14 }}>
            <h4>4. References & Evidence Base</h4>
            <ol style={{ paddingLeft: 18, fontSize: 13, color: "var(--ink-secondary)", lineHeight: 1.6 }}>
              {data.references.map((r: any, idx: number) => (
                <li key={idx}>
                  <strong>{r.title}</strong> {r.authors ? `— ${r.authors}` : ""} ({r.year || "n.d."})
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  }

  if (tool === "similarity") {
    return (
      <div>
        <div className="eyebrow">Similarity Graph</div>
        <h3 style={{ margin: "0 0 14px", font: "700 18px var(--font-serif)" }}>Pairwise Semantic Map</h3>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
          <div className="trend-card">
            <div className="trend-year">Indexed Nodes</div>
            <div className="trend-count">{data.nodes?.length ?? 0}</div>
          </div>
          <div className="trend-card">
            <div className="trend-year">Semantic Edges</div>
            <div className="trend-count">{data.edges?.length ?? 0}</div>
          </div>
        </div>

        <h5 style={{ margin: "16px 0 8px", fontSize: 11, textTransform: "uppercase", color: "var(--teal)" }}>
          Strongest Relationships (Cosine Similarity &ge; 0.25)
        </h5>

        {data.edges?.length ? (
          <div style={{ display: "grid", gap: 8 }}>
            {data.edges.map((e: any, idx: number) => {
              const srcPaper = data.nodes?.find((n: any) => n.id === e.source);
              const tgtPaper = data.nodes?.find((n: any) => n.id === e.target);
              const percent = Math.round((e.similarity ?? 0) * 100);
              return (
                <div key={idx} className="detail-block" style={{ padding: 12 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 700 }}>
                      {srcPaper?.title?.slice(0, 32)}... &harr; {tgtPaper?.title?.slice(0, 32)}...
                    </span>
                    <span className="badge badge-full-text">{percent}% similarity</span>
                  </div>
                  <div style={{ background: "var(--cream)", height: 6, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${percent}%`, height: "100%", background: "var(--teal)" }} />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty">No pairwise edges met the &ge; 0.25 similarity threshold.</div>
        )}

        {data.notice && (
          <p style={{ fontSize: 12, color: "var(--muted)", marginTop: 14, fontStyle: "italic" }}>
            {data.notice}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      <div className="eyebrow">Computed Output</div>
      <pre style={{ fontSize: 12, background: "var(--paper)", padding: 14, borderRadius: "var(--radius-sm)", overflowX: "auto" }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
