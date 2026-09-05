"use client";

import { useState } from "react";

export interface CitationItem {
  page?: number | null;
  section?: string | null;
  chunk_index?: number;
  text?: string;
  document_id?: string | null;
}

interface CitationCardProps {
  citation: CitationItem;
  index: number;
}

export default function CitationCard({ citation, index }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false);
  const locationLabel = citation.page ? `p. ${citation.page}` : (citation.section || `Chunk ${index + 1}`);

  return (
    <div style={{ margin: "4px 0" }}>
      <button
        type="button"
        className="citation-chip"
        onClick={() => setExpanded(!expanded)}
        title={citation.text ? "Click to view citation text" : undefined}
      >
        <span>[{index + 1}]</span>
        <span>{citation.section || "Body"}</span>
        <span>·</span>
        <span>{locationLabel}</span>
      </button>

      {expanded && citation.text && (
        <blockquote className="evidence-quote" tabIndex={0}>
          <header>
            <span>{citation.section || "Section Excerpt"}</span>
            <span>{locationLabel}</span>
          </header>
          <p style={{ margin: 0 }}>"{citation.text}"</p>
        </blockquote>
      )}
    </div>
  );
}

