"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

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
  const pathname = usePathname();

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
    </header>
  );
}
