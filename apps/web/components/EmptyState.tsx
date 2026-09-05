"use client";

interface EmptyStateProps {
  title: string;
  description?: string;
  actionText?: string;
  onAction?: () => void;
  icon?: string;
}

export default function EmptyState({
  title,
  description,
  actionText,
  onAction,
  icon = "📄",
}: EmptyStateProps) {
  return (
    <div className="empty">
      <div style={{ fontSize: 32, marginBottom: 12 }}>{icon}</div>
      <strong style={{ display: "block", fontSize: 16, color: "var(--ink)", marginBottom: 6 }}>
        {title}
      </strong>
      {description && <p style={{ margin: "0 0 14px", color: "var(--muted)", maxWidth: 420, marginInline: "auto" }}>{description}</p>}
      {actionText && onAction && (
        <button type="button" className="secondary-btn" onClick={onAction} style={{ marginTop: 8 }}>
          {actionText}
        </button>
      )}
    </div>
  );
}

