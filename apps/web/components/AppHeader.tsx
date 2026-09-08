"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import AISettingsModal from "./AISettingsModal";

const links = [
  ["Discover", "/"],
  ["My Papers", "/papers"],
  ["Collections", "/collections"],
  ["Compare", "/compare"],
  ["Trends", "/trends"],
  ["Gaps", "/research-gaps"],
  ["Ideas", "/research-ideas"],
  ["Proposal", "/proposal"],
  ["Similarity", "/similarity-map"],
] as const;

export default function AppHeader() {
  const [email, setEmail] = useState("");
  const [isAIModalOpen, setIsAIModalOpen] = useState(false);
  const [aiStatus, setAiStatus] = useState({ hasKey: false, provider: "gemini", model: "" });
  const pathname = usePathname();

  useEffect(() => {
    function updateStatus() {
      const key = localStorage.getItem("research_llm_key");
      const prov = localStorage.getItem("research_llm_provider") || "gemini";
      const mod = localStorage.getItem("research_llm_model") || "gemini-1.5-flash";
      setAiStatus({ hasKey: !!key, provider: prov, model: mod });
    }
    updateStatus();
    window.addEventListener("llm-settings-updated", updateStatus);
    return () => window.removeEventListener("llm-settings-updated", updateStatus);
  }, []);

  useEffect(() => {
    const token = localStorage.getItem("research_token");
    if (!token) return;
    fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/api/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(response => (response.ok ? response.json() : null))
      .then(data => setEmail(data?.email || ""))
      .catch(() => undefined);
  }, []);

  function logout() {
    localStorage.removeItem("research_token");
    localStorage.removeItem("research_chat_session");
    window.location.href = "/";
  }

  return (
    <header className="app-header">
      <Link className="brand" href="/" aria-label="Lumen Research Home">
        <span className="brand-icon">⚛</span>
        <span className="brand-title">LUMEN</span>
        <span className="brand-badge">RESEARCH</span>
      </Link>

      <nav className="app-nav" aria-label="Research workspace">
        {links.map(([label, href]) => {
          const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
          return (
            <Link
              key={label}
              href={href}
              className={`nav-link ${isActive ? "active" : ""}`}
            >
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="header-actions">
        <button
          type="button"
          onClick={() => setIsAIModalOpen(true)}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            padding: "5px 12px",
            borderRadius: "var(--radius-pill)",
            fontSize: "12px",
            fontWeight: 600,
            cursor: "pointer",
            border: aiStatus.hasKey ? "1px solid var(--emerald-border)" : "1px solid var(--primary-border)",
            backgroundColor: aiStatus.hasKey ? "var(--emerald-light)" : "var(--primary-light)",
            color: aiStatus.hasKey ? "var(--emerald)" : "var(--primary)",
            transition: "all 0.15s ease",
          }}
          title="Configure Google Gemini / Cloud AI key"
        >
          <span style={{ fontSize: "13px" }}>⚡</span>
          <span>
            {aiStatus.hasKey
              ? `${aiStatus.provider === "gemini" ? "Gemini" : aiStatus.provider.toUpperCase()} Active`
              : "AI Key: Setup Gemini"}
          </span>
        </button>

        <div className="status-indicator" title="All backend, Redis and pgvector services active">
          <span className="status-pulse" />
          <span>Live Systems</span>
        </div>

        <div className="auth-actions">
          {email ? (
            <>
              <div className="user-chip" title={email}>
                <span className="user-avatar">{email.slice(0, 1).toUpperCase()}</span>
                <span className="user-email-text">{email}</span>
              </div>
              <button
                type="button"
                className="secondary small-button"
                onClick={logout}
                title="Sign out of your session"
              >
                Sign out
              </button>
            </>
          ) : (
            <>
              <Link href="/login" className="secondary small-button">
                Sign in
              </Link>
              <Link href="/register" className="primary small-button">
                Get Started
              </Link>
            </>
          )}
        </div>
      </div>
      <AISettingsModal isOpen={isAIModalOpen} onClose={() => setIsAIModalOpen(false)} />
    </header>
  );
}
