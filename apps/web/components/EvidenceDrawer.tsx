"use client";

import { useState } from "react";
import CitationCard, { CitationItem } from "./CitationCard";

interface EvidenceDrawerProps {
  evidence: CitationItem[];
  title?: string;
}

export default function EvidenceDrawer({ evidence, title = "Retrieved Evidence" }: EvidenceDrawerProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!evidence || evidence.length === 0) {
    return null;
  }

  return (
    <div style={{ marginTop: 14, borderTop: "1px solid var(--line-subtle)", paddingTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h5 style={{ margin: 0, fontSize: 12, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--teal)" }}>
          {title} ({evidence.length} {evidence.length === 1 ? "source" : "sources"})
        </h5>
        <button
          type="button"
          className="link-button"
          style={{ fontSize: 12 }}
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? "Collapse evidence" : "Inspect excerpts"}
        </button>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
        {evidence.map((item, idx) => (
          <CitationCard key={`${item.chunk_index ?? idx}-${item.page ?? "na"}`} citation={item} index={idx} />
        ))}
      </div>

      {isOpen && (
        <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
          {evidence.map((item, idx) => (
            <div key={`expanded-${item.chunk_index ?? idx}`} className="evidence-quote">
              <header>
                <span>{item.section || "Evidence Chunk"}</span>
                <span>{item.page ? `Page ${item.page}` : `Index ${item.chunk_index ?? idx}`}</span>
              </header>
              <p style={{ margin: "4px 0 0" }}>{item.text || "Direct text citation verified from full-text index."}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

