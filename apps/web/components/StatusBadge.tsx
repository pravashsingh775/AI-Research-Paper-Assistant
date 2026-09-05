"use client";

export type EvidenceState = "full-text" | "processing" | "metadata-only" | "failed" | "private-upload";

interface StatusBadgeProps {
  state?: EvidenceState | string;
  className?: string;
}

export default function StatusBadge({ state = "metadata-only", className = "" }: StatusBadgeProps) {
  const normalized = state.toLowerCase();

  if (normalized === "full-text") {
    return (
      <span className={`badge badge-full-text ${className}`} title="Full-text indexed and verified for Q&A">
        <span className="badge-dot" /> Full text
      </span>
    );
  }

  if (normalized === "processing") {
    return (
      <span className={`badge badge-processing ${className}`} title="PDF is currently being extracted and indexed">
        <span className="badge-dot" /> Processing
      </span>
    );
  }

  if (normalized === "failed") {
    return (
      <span className={`badge badge-failed ${className}`} title="Full text extraction failed">
        <span className="badge-dot" /> Failed
      </span>
    );
  }

  return (
    <span className={`badge badge-metadata ${className}`} title="Discovery metadata only. Full-text not indexed.">
      <span className="badge-dot" /> Metadata only
    </span>
  );
}

