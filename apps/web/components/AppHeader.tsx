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
      <Link className="brand" href="/">
        LUMEN <span>RESEARCH</span>
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
      <div className="auth-actions">
        {email ? (
          <>
            <span className="user-chip" title={email}>
              {email.slice(0, 1).toUpperCase()}
            </span>
            <button type="button" className="link-button" onClick={logout}>
              Sign out
            </button>
          </>
        ) : (
          <>
            <Link href="/login" className="link-button">
              Sign in
            </Link>
            <Link className="primary small-button" href="/register">
              Create account
            </Link>
          </>
        )}
      </div>
    </header>
  );
}
