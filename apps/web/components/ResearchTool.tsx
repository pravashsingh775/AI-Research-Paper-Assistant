"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, loadPapers, WorkspacePaper } from "../lib/api";
import AppHeader from "./AppHeader";
import StatusBadge from "./StatusBadge";
import LoadingSpinner from "./LoadingSpinner";
import EmptyState from "./EmptyState";

type Tool = "compare" | "trends" | "gaps" | "ideas" | "proposal" | "similarity";

const toolIcons: Record<Tool, string> = {
  compare: "⚖️",
  trends: "📈",
  gaps: "🧩",
  ideas: "💡",
  proposal: "📝",
  similarity: "🕸️",
};

const labels: Record<Tool, string> = {
  compare: "Paper Comparison Matrix",
  trends: "Research Trends & Trajectory",
  gaps: "Research Gaps & Evidence Coverage",
  ideas: "Novel Research Directions & Ideas",
  proposal: "Academic Proposal Generator",
  similarity: "Paper Similarity Network",
};

const descriptions: Record<Tool, string> = {
  compare: "Synthesize structured differences, evaluation protocols, and dataset usage across selected papers.",
  trends: "Inspect chronological publication trends, emerging keywords, and domain evolution.",
  gaps: "Detect methodology gaps, missing evidence coverage, and evaluation discrepancies across literature.",
  ideas: "Generate grounded, hypothesis-driven research ideas based on collective findings.",
  proposal: "Draft a formal academic research proposal with problem statement, methodology, and citations.",
  similarity: "Compute semantic pairwise similarities and graph relationships across paper representations.",
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
      setError(`Please select at least ${minRequired} paper${minRequired > 1 ? "s" : ""} to perform this analysis.`);
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

  const minRequired = tool === "trends" ? 1 : 2;
  const isReady = selected.length >= minRequired;

  return (
    <main className="shell">
      <AppHeader />

      {/* Tool Hero Banner */}
      <section className="tool-hero">
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>{toolIcons[tool]}</span>
          <span>RESEARCH INTELLIGENCE</span>
        </div>
        <h1 style={{ fontSize: "clamp(2rem, 3.5vw, 2.75rem)", marginTop: 8 }}>{labels[tool]}</h1>
        <p className="panel-copy" style={{ maxWidth: 680 }}>
          {descriptions[tool]}
        </p>
      </section>

      {/* Main 2-Column Tool Workspace */}
      <form className="tool-layout" onSubmit={submit} style={{ marginTop: 24 }}>
        {/* Left Config / Selector Column */}
        <section
          style={{
            background: "var(--surface)",
            border: "1px solid var(--line)",
            borderRadius: "var(--radius-md)",
            padding: "20px 24px",
            boxShadow: "var(--shadow-sm)",
            display: "flex",
            flexDirection: "column",
            gap: 18,
          }}
        >
          {/* Topic Focus */}
          <div>
            <label
              style={{
                display: "block",
                fontSize: 13,
                fontWeight: 700,
                color: "var(--ink)",
                marginBottom: 6,
                letterSpacing: "0.02em",
              }}
            >
              Topic Focus (Optional)
            </label>
            <input
              type="text"
              value={topic}
              onChange={event => setTopic(event.target.value)}
              placeholder="e.g. Graph Neural Networks, LLM Evaluation..."
              style={{
                width: "100%",
                padding: "10px 14px",
                fontSize: 14,
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                background: "var(--surface)",
                color: "var(--ink)",
                outline: "none",
              }}
            />
          </div>

          {/* Paper Picker Header */}
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>
                  Select Papers
                </span>
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    padding: "2px 8px",
                    borderRadius: "9999px",
                    background: selected.length >= minRequired ? "var(--primary-50)" : "var(--slate-100)",
                    color: selected.length >= minRequired ? "var(--primary-700)" : "var(--ink-muted)",
                  }}
                >
                  {selected.length} / 15 selected
                </span>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  type="button"
                  onClick={selectAll}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--primary)",
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: "pointer",
                    padding: "2px 6px",
                  }}
                >
                  Select all
                </button>
                <span style={{ color: "var(--line)" }}>|</span>
                <button
                  type="button"
                  onClick={clearSelected}
                  style={{
                    background: "none",
                    border: "none",
                    color: "var(--ink-muted)",
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: "pointer",
                    padding: "2px 6px",
                  }}
                >
                  Clear
                </button>
              </div>
            </div>

            {/* Quick Filter */}
            <div style={{ position: "relative", marginBottom: 12 }}>
              <span
                style={{
                  position: "absolute",
                  left: 10,
                  top: "50%",
                  transform: "translateY(-50%)",
                  color: "var(--ink-muted)",
                  fontSize: 13,
                  pointerEvents: "none",
                }}
              >
                🔍
              </span>
              <input
                type="text"
                value={filterQuery}
                onChange={e => setFilterQuery(e.target.value)}
                placeholder="Filter library papers..."
                style={{
                  width: "100%",
                  padding: "8px 12px 8px 30px",
                  fontSize: 13,
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--slate-50)",
                  color: "var(--ink)",
                }}
              />
            </div>

            {/* Paper Selection List */}
            <div
              className="paper-picker"
              style={{
                maxHeight: 380,
                overflowY: "auto",
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                background: "var(--slate-50)",
                padding: 6,
                display: "grid",
                gap: 6,
              }}
            >
              {loading ? (
                <div style={{ padding: "30px 0" }}>
                  <LoadingSpinner message="Loading your papers..." />
                </div>
              ) : filteredPapers.length ? (
                filteredPapers.map(paper => {
                  const isChecked = selected.includes(paper.id);
                  return (
                    <label
                      key={paper.id}
                      style={{
                        display: "flex",
                        gap: 12,
                        alignItems: "flex-start",
                        padding: "10px 12px",
                        borderRadius: "var(--radius-sm)",
                        background: isChecked ? "var(--primary-50)" : "var(--surface)",
                        border: isChecked ? "1px solid var(--primary-300)" : "1px solid var(--line)",
                        boxShadow: isChecked ? "var(--shadow-sm)" : "none",
                        cursor: "pointer",
                        transition: "all var(--speed-fast)",
                      }}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => toggle(paper.id)}
                        style={{
                          marginTop: 3,
                          accentColor: "var(--primary)",
                          width: 16,
                          height: 16,
                          cursor: "pointer",
                        }}
                      />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center", marginBottom: 2 }}>
                          <strong
                            style={{
                              fontSize: 13,
                              color: isChecked ? "var(--primary-900)" : "var(--ink)",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {paper.title}
                          </strong>
                          <StatusBadge state={paper.evidence_state} />
                        </div>
                        <div style={{ fontSize: 11, color: "var(--ink-muted)", marginBottom: 2 }}>
                          {paper.authors_raw || "Unknown"} · {paper.year || "n/a"} · {paper.venue || "Paper"}
                        </div>
                        {paper.summary && (
                          <div
                            style={{
                              fontSize: 11,
                              color: "var(--ink-secondary)",
                              display: "-webkit-box",
                              WebkitLineClamp: 1,
                              WebkitBoxOrient: "vertical",
                              overflow: "hidden",
                            }}
                          >
                            {paper.summary}
                          </div>
                        )}
                      </div>
                    </label>
                  );
                })
              ) : (
                <div style={{ padding: "24px 12px", textAlign: "center" }}>
                  <EmptyState
                    title="No matching papers"
                    description="Upload a paper or discover papers from search to add to your library."
                    actionText="Go to Discovery"
                    onAction={() => router.push("/")}
                  />
                </div>
              )}
            </div>
          </div>

          {/* Submit Action */}
          <button
            type="submit"
            className="primary tool-submit"
            disabled={busy || loading || !isReady}
            style={{
              width: "100%",
              padding: "12px 20px",
              fontSize: 14,
              fontWeight: 700,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              opacity: isReady ? 1 : 0.6,
              cursor: isReady ? "pointer" : "not-allowed",
            }}
          >
            <span>{busy ? "⏳" : toolIcons[tool]}</span>
            <span>{busy ? "Synthesizing Evidence..." : `Run ${labels[tool]}`}</span>
          </button>

          {!isReady && papers.length > 0 && (
            <p style={{ fontSize: 12, color: "var(--ink-muted)", margin: 0, textAlign: "center" }}>
              Select at least {minRequired} paper{minRequired > 1 ? "s" : ""} to begin analysis.
            </p>
          )}

          {error && <p className="error" role="alert">{error}</p>}
        </section>

        {/* Right Output Results Column */}
        <aside
          className="result-panel"
          style={{
            background: "var(--surface)",
            border: "1px solid var(--line)",
            borderRadius: "var(--radius-md)",
            padding: "24px 28px",
            boxShadow: "var(--shadow-sm)",
            minHeight: 480,
          }}
        >
          {busy ? (
            <div style={{ padding: "80px 0", textAlign: "center" }}>
              <LoadingSpinner message="Synthesizing multi-paper evidence..." size={42} />
            </div>
          ) : result ? (
            <ResultViewer tool={tool} data={result} />
          ) : (
            <div style={{ padding: "80px 0" }}>
              <EmptyState
                title="Awaiting Analysis"
                description={`Select ${minRequired}+ papers on the left and run ${labels[tool]} to generate evidence-backed research insights.`}
                icon={toolIcons[tool]}
              />
            </div>
          )}
        </aside>
      </form>
    </main>
  );
}

function ResultViewer({ tool, data }: { tool: Tool; data: Record<string, any> }) {
  const [copied, setCopied] = useState(false);

  // Compare Tool Output
  if (tool === "compare" && data.papers) {
    return (
      <div>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>⚖️</span> COMPARISON MATRIX
        </div>
        <h2 style={{ margin: "4px 0 16px", fontSize: "1.4rem", fontWeight: 800, color: "var(--ink)" }}>
          {data.topic}
        </h2>

        {data.key_similarities && data.key_similarities.length > 0 && (
          <div style={{ marginBottom: 20 }}>
            <h5 style={{ margin: "0 0 8px", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--primary-600)", fontWeight: 700 }}>
              Shared Core Concepts &amp; Methodologies
            </h5>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {data.key_similarities.map((item: string) => (
                <span
                  key={item}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "9999px",
                    background: "var(--primary-50)",
                    color: "var(--primary-700)",
                    fontSize: 12,
                    fontWeight: 600,
                    border: "1px solid var(--primary-100)",
                  }}
                >
                  {item}
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="matrix-table-wrap" style={{ borderRadius: "var(--radius-md)", border: "1px solid var(--line)", overflow: "hidden", marginBottom: 20 }}>
          <table className="matrix-table" style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ background: "var(--slate-50)", borderBottom: "1px solid var(--line)" }}>
                <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 700, color: "var(--ink-secondary)" }}>Paper</th>
                <th style={{ padding: "12px 12px", textAlign: "center", fontWeight: 700, color: "var(--ink-secondary)" }}>Year</th>
                <th style={{ padding: "12px 12px", textAlign: "left", fontWeight: 700, color: "var(--ink-secondary)" }}>Venue</th>
                <th style={{ padding: "12px 12px", textAlign: "center", fontWeight: 700, color: "var(--ink-secondary)" }}>Citations</th>
                <th style={{ padding: "12px 16px", textAlign: "left", fontWeight: 700, color: "var(--ink-secondary)" }}>Key Findings &amp; Evidence</th>
              </tr>
            </thead>
            <tbody>
              {data.papers.map((p: any, idx: number) => (
                <tr key={idx} style={{ borderBottom: "1px solid var(--line)", background: idx % 2 === 0 ? "var(--surface)" : "var(--slate-50)" }}>
                  <td style={{ padding: "12px 16px", verticalAlign: "top" }}>
                    <strong style={{ color: "var(--ink)", display: "block", marginBottom: 2 }}>{p.paper}</strong>
                    <small style={{ color: "var(--ink-muted)" }}>{p.authors}</small>
                  </td>
                  <td style={{ padding: "12px 12px", textAlign: "center", verticalAlign: "top", color: "var(--ink-secondary)" }}>
                    {p.year || "—"}
                  </td>
                  <td style={{ padding: "12px 12px", verticalAlign: "top", color: "var(--ink-secondary)" }}>
                    {p.venue || "—"}
                  </td>
                  <td style={{ padding: "12px 12px", textAlign: "center", verticalAlign: "top", fontWeight: 600, color: "var(--primary-600)" }}>
                    {p.citations ?? "—"}
                  </td>
                  <td style={{ padding: "12px 16px", verticalAlign: "top", color: "var(--ink-secondary)", fontSize: 12, lineHeight: 1.5 }}>
                    {p.evidence}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {data.key_differences && (
          <div
            style={{
              background: "#fffbeb",
              border: "1px solid #fde68a",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              marginBottom: 16,
            }}
          >
            <h4 style={{ margin: "0 0 6px", fontSize: 13, fontWeight: 700, color: "#92400e", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              ⚡ Key Differences &amp; Divergent Findings
            </h4>
            <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#78350f" }}>{data.key_differences}</p>
          </div>
        )}

        {data.potential_research_opportunity && (
          <div
            style={{
              background: "var(--success-bg)",
              border: "1px solid #a7f3d0",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
            }}
          >
            <h4 style={{ margin: "0 0 6px", fontSize: 13, fontWeight: 700, color: "var(--success)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              💡 Identified Research Opportunity
            </h4>
            <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "#065f46" }}>{data.potential_research_opportunity}</p>
          </div>
        )}
      </div>
    );
  }

  // Trends Tool Output
  if (tool === "trends" && data.publication_trend) {
    const years = Object.entries(data.publication_trend).sort(([a], [b]) => a.localeCompare(b));
    const maxCount = Math.max(...years.map(([_, count]) => Number(count) || 1), 1);

    return (
      <div>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>📈</span> RESEARCH TRAJECTORY
        </div>
        <h2 style={{ margin: "4px 0 6px", fontSize: "1.4rem", fontWeight: 800, color: "var(--ink)" }}>{data.topic}</h2>
        <p style={{ fontSize: 13, color: "var(--ink-muted)", margin: "0 0 20px" }}>
          Analyzed across {data.paper_count} papers in collection.
        </p>

        <h5 style={{ margin: "16px 0 12px", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--primary-600)", fontWeight: 700 }}>
          Chronological Publication Activity
        </h5>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 12, marginBottom: 24 }}>
          {years.map(([year, count]) => {
            const countNum = Number(count) || 0;
            const barPercent = Math.round((countNum / maxCount) * 100);
            return (
              <div
                key={year}
                style={{
                  background: "var(--slate-50)",
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-sm)",
                  padding: "12px",
                  textAlign: "center",
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", marginBottom: 4 }}>{year}</div>
                <div style={{ fontSize: 22, fontWeight: 800, color: "var(--primary-700)", lineHeight: 1 }}>{countNum}</div>
                <div style={{ height: 4, background: "var(--line)", borderRadius: 2, marginTop: 8, overflow: "hidden" }}>
                  <div style={{ height: "100%", background: "var(--primary)", width: `${barPercent}%` }} />
                </div>
              </div>
            );
          })}
        </div>

        {data.emerging_keywords && (
          <div>
            <h5 style={{ margin: "0 0 10px", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--primary-600)", fontWeight: 700 }}>
              High-Frequency Keywords &amp; Methodologies
            </h5>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {data.emerging_keywords.map((kw: string) => (
                <span
                  key={kw}
                  style={{
                    padding: "6px 12px",
                    borderRadius: "9999px",
                    background: "var(--surface)",
                    border: "1px solid var(--line)",
                    fontSize: 13,
                    fontWeight: 600,
                    color: "var(--ink)",
                    boxShadow: "var(--shadow-sm)",
                  }}
                >
                  #{kw}
                </span>
              ))}
            </div>
          </div>
        )}

        {data.notice && (
          <p style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 24, fontStyle: "italic", borderTop: "1px solid var(--line)", paddingTop: 12 }}>
            {data.notice}
          </p>
        )}
      </div>
    );
  }

  // Gaps Tool Output
  if (tool === "gaps") {
    return (
      <div>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>🧩</span> GAP ANALYSIS
        </div>
        <h2 style={{ margin: "4px 0 16px", fontSize: "1.4rem", fontWeight: 800, color: "var(--ink)" }}>{data.topic}</h2>

        {data.evidence_based_findings?.map((f: any, idx: number) => {
          const conf = Math.round((f.confidence ?? 1) * 100);
          return (
            <div
              key={idx}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                borderLeft: "4px solid var(--warning)",
                borderRadius: "var(--radius-sm)",
                padding: "16px 20px",
                marginBottom: 16,
                boxShadow: "var(--shadow-sm)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <h4 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "var(--ink)" }}>
                  {f.category || "Evidence Coverage Gap"}
                </h4>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "9999px",
                    fontSize: 11,
                    fontWeight: 700,
                    background: "#fef3c7",
                    color: "#b45309",
                  }}
                >
                  Confidence {conf}%
                </span>
              </div>
              <p style={{ margin: "0 0 10px", fontSize: 13, lineHeight: 1.6, color: "var(--ink-secondary)" }}>{f.finding}</p>
              {f.affected_papers?.length > 0 && (
                <div style={{ background: "var(--slate-50)", padding: "10px 12px", borderRadius: "var(--radius-sm)" }}>
                  <strong style={{ fontSize: 11, textTransform: "uppercase", color: "var(--ink-muted)", display: "block", marginBottom: 4 }}>
                    Affected Literature:
                  </strong>
                  <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "var(--ink-secondary)" }}>
                    {f.affected_papers.map((p: string, i: number) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}

        {data.ai_generated_hypotheses?.map((h: any, idx: number) => (
          <div
            key={idx}
            style={{
              background: "var(--primary-50)",
              border: "1px solid var(--primary-100)",
              borderRadius: "var(--radius-sm)",
              padding: "16px 20px",
              marginTop: 14,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 700, color: "var(--primary-800)", marginBottom: 4 }}>
              Hypothesis ({h.category}):
            </div>
            <p style={{ margin: "0 0 6px", fontSize: 13, color: "var(--primary-900)", lineHeight: 1.5 }}>{h.finding}</p>
            <div style={{ fontSize: 12, color: "var(--primary-600)" }}>Evidence: {h.evidence}</div>
          </div>
        ))}
      </div>
    );
  }

  // Ideas Tool Output
  if (tool === "ideas") {
    return (
      <div>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>💡</span> NOVEL RESEARCH DIRECTIONS
        </div>
        <h2 style={{ margin: "4px 0 6px", fontSize: "1.4rem", fontWeight: 800, color: "var(--ink)" }}>{data.topic}</h2>
        <p style={{ fontSize: 12, color: "var(--primary-600)", fontStyle: "italic", margin: "0 0 20px" }}>{data.label}</p>

        <div style={{ display: "grid", gap: 16 }}>
          {data.ideas?.map((idea: any, idx: number) => (
            <div
              key={idx}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-md)",
                padding: "20px 24px",
                boxShadow: "var(--shadow-sm)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                <span
                  style={{
                    width: 24,
                    height: 24,
                    borderRadius: "50%",
                    background: "var(--primary)",
                    color: "#ffffff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontSize: 12,
                    fontWeight: 700,
                  }}
                >
                  {idx + 1}
                </span>
                <h4 style={{ margin: 0, color: "var(--ink)", fontSize: 15, fontWeight: 700 }}>
                  {idea.title}
                </h4>
              </div>

              <div
                style={{
                  background: "#fff1f2",
                  borderLeft: "3px solid #f43f5e",
                  padding: "10px 14px",
                  borderRadius: 4,
                  marginBottom: 10,
                }}
              >
                <strong style={{ fontSize: 11, color: "#9f1239", textTransform: "uppercase", display: "block", marginBottom: 2 }}>
                  Unresolved Problem:
                </strong>
                <p style={{ margin: 0, fontSize: 13, color: "#881337", lineHeight: 1.5 }}>{idea.problem}</p>
              </div>

              <div
                style={{
                  background: "var(--success-bg)",
                  borderLeft: "3px solid var(--success)",
                  padding: "10px 14px",
                  borderRadius: 4,
                  marginBottom: 12,
                }}
              >
                <strong style={{ fontSize: 11, color: "var(--success)", textTransform: "uppercase", display: "block", marginBottom: 2 }}>
                  Proposed Contribution:
                </strong>
                <p style={{ margin: 0, fontSize: 13, color: "#065f46", lineHeight: 1.5 }}>{idea.proposed_contribution}</p>
              </div>

              {idea.evidence?.length > 0 && (
                <div style={{ borderTop: "1px solid var(--line)", paddingTop: 10 }}>
                  <span style={{ fontSize: 11, color: "var(--ink-muted)", textTransform: "uppercase", fontWeight: 700, display: "block", marginBottom: 6 }}>
                    Grounding Papers:
                  </span>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {idea.evidence.map((title: string, i: number) => (
                      <span
                        key={i}
                        style={{
                          fontSize: 11,
                          padding: "3px 8px",
                          borderRadius: 4,
                          background: "var(--slate-100)",
                          color: "var(--ink-secondary)",
                          fontWeight: 500,
                        }}
                      >
                        📄 {title}
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

  // Proposal Tool Output
  if (tool === "proposal") {
    function copyProposal() {
      const markdown = `# ${data.title}\n\n## 1. Problem Statement\n${data.problem_statement}\n\n## 2. Proposed Methodology\n${data.methodology}\n\n## 3. Evaluation Protocol\n${data.evaluation}\n\n## 4. References\n${data.references?.map((r: any) => `- ${r.title} (${r.year || "n.d."})`).join("\n")}`;
      navigator.clipboard.writeText(markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    }

    return (
      <div className="proposal-sheet" style={{ maxWidth: 740, marginInline: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span>📝</span> {data.label || "FORMAL RESEARCH PROPOSAL"}
          </div>
          <button
            type="button"
            onClick={copyProposal}
            style={{
              padding: "6px 14px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--line)",
              background: copied ? "var(--success-bg)" : "var(--surface)",
              color: copied ? "var(--success)" : "var(--ink)",
              fontSize: 12,
              fontWeight: 600,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 6,
              boxShadow: "var(--shadow-sm)",
            }}
          >
            <span>{copied ? "✓ Copied" : "📋 Copy Markdown"}</span>
          </button>
        </div>

        <h2 style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--ink)", lineHeight: 1.3, marginBottom: 24 }}>
          {data.title}
        </h2>

        <div className="proposal-section" style={{ marginBottom: 20 }}>
          <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--primary-700)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 8 }}>
            1. Problem Statement
          </h4>
          <p style={{ fontSize: 14, lineHeight: 1.7, color: "var(--ink-secondary)" }}>{data.problem_statement}</p>
        </div>

        <div className="proposal-section" style={{ marginBottom: 20 }}>
          <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--primary-700)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 8 }}>
            2. Proposed Methodology
          </h4>
          <p style={{ fontSize: 14, lineHeight: 1.7, color: "var(--ink-secondary)" }}>{data.methodology}</p>
        </div>

        <div className="proposal-section" style={{ marginBottom: 24 }}>
          <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--primary-700)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 8 }}>
            3. Evaluation Protocol
          </h4>
          <p style={{ fontSize: 14, lineHeight: 1.7, color: "var(--ink-secondary)" }}>{data.evaluation}</p>
        </div>

        {data.references?.length > 0 && (
          <div className="proposal-section" style={{ borderTop: "1px solid var(--line)", paddingTop: 18 }}>
            <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--primary-700)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 12 }}>
              4. Grounding References &amp; Evidence Base
            </h4>
            <ol style={{ paddingLeft: 20, fontSize: 13, color: "var(--ink-secondary)", lineHeight: 1.8 }}>
              {data.references.map((r: any, idx: number) => (
                <li key={idx}>
                  <strong style={{ color: "var(--ink)" }}>{r.title}</strong> {r.authors ? `— ${r.authors}` : ""} ({r.year || "n.d."})
                </li>
              ))}
            </ol>
          </div>
        )}
      </div>
    );
  }

  // Similarity Tool Output
  if (tool === "similarity") {
    return (
      <div>
        <div className="eyebrow" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span>🕸️</span> SIMILARITY GRAPH
        </div>
        <h2 style={{ margin: "4px 0 16px", fontSize: "1.4rem", fontWeight: 800, color: "var(--ink)" }}>
          Pairwise Semantic Map
        </h2>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 24 }}>
          <div
            style={{
              background: "var(--slate-50)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", textTransform: "uppercase" }}>
              Indexed Nodes
            </div>
            <div style={{ fontSize: 26, fontWeight: 800, color: "var(--ink)", marginTop: 4 }}>
              {data.nodes?.length ?? 0}
            </div>
          </div>
          <div
            style={{
              background: "var(--slate-50)",
              border: "1px solid var(--line)",
              borderRadius: "var(--radius-md)",
              padding: "16px 20px",
              textAlign: "center",
            }}
          >
            <div style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-muted)", textTransform: "uppercase" }}>
              Semantic Edges
            </div>
            <div style={{ fontSize: 26, fontWeight: 800, color: "var(--primary-700)", marginTop: 4 }}>
              {data.edges?.length ?? 0}
            </div>
          </div>
        </div>

        <h5 style={{ margin: "16px 0 12px", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.04em", color: "var(--primary-600)", fontWeight: 700 }}>
          Strongest Relationships (Cosine Similarity &ge; 0.25)
        </h5>

        {data.edges?.length ? (
          <div style={{ display: "grid", gap: 10 }}>
            {data.edges.map((e: any, idx: number) => {
              const srcPaper = data.nodes?.find((n: any) => n.id === e.source);
              const tgtPaper = data.nodes?.find((n: any) => n.id === e.target);
              const percent = Math.round((e.similarity ?? 0) * 100);
              return (
                <div
                  key={idx}
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--line)",
                    borderRadius: "var(--radius-sm)",
                    padding: "14px 18px",
                    boxShadow: "var(--shadow-sm)",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8, gap: 12 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "var(--ink)" }}>
                      {srcPaper?.title?.slice(0, 36)}... <span style={{ color: "var(--primary)" }}>&harr;</span> {tgtPaper?.title?.slice(0, 36)}...
                    </span>
                    <span
                      style={{
                        padding: "3px 8px",
                        borderRadius: "9999px",
                        fontSize: 11,
                        fontWeight: 700,
                        background: "var(--success-bg)",
                        color: "var(--success)",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {percent}% similarity
                    </span>
                  </div>
                  <div style={{ background: "var(--slate-100)", height: 6, borderRadius: 3, overflow: "hidden" }}>
                    <div style={{ width: `${percent}%`, height: "100%", background: "var(--primary)" }} />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <EmptyState
            title="No Strong Edges Found"
            description="No pairwise edges met the &ge; 0.25 similarity threshold. Select more closely related papers to map relationships."
            icon="🕸️"
          />
        )}

        {data.notice && (
          <p style={{ fontSize: 12, color: "var(--ink-muted)", marginTop: 20, fontStyle: "italic", borderTop: "1px solid var(--line)", paddingTop: 12 }}>
            {data.notice}
          </p>
        )}
      </div>
    );
  }

  return (
    <div>
      <div className="eyebrow">COMPUTED OUTPUT</div>
      <pre style={{ fontSize: 12, background: "var(--slate-50)", padding: 16, borderRadius: "var(--radius-sm)", overflowX: "auto", border: "1px solid var(--line)" }}>
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}
