"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API}/api/auth/token`, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ username: email, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to sign in");
      localStorage.setItem("research_token", data.access_token);
      router.push("/");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to sign in");
    } finally {
      setBusy(false);
    }
  }

  function fillDemo() {
    setEmail("researcher@lumen.ai");
    setPassword("research123");
  }

  return (
    <main
      className="auth-shell"
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "24px",
        background: "radial-gradient(ellipse at top, #eef2ff 0%, #f8fafc 60%, #f1f5f9 100%)",
      }}
    >
      <div
        className="auth-card"
        style={{
          width: "100%",
          maxWidth: 440,
          background: "var(--surface)",
          border: "1px solid var(--line)",
          borderRadius: "var(--radius-lg)",
          padding: "36px 32px",
          boxShadow: "0 20px 40px -15px rgba(79, 70, 229, 0.08), 0 0 1px 1px rgba(0, 0, 0, 0.04)",
        }}
      >
        {/* Brand Header */}
        <div style={{ textAlign: "center", marginBottom: 28 }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 48,
              height: 48,
              borderRadius: "var(--radius-md)",
              background: "linear-gradient(135deg, var(--primary) 0%, var(--primary-800) 100%)",
              color: "#ffffff",
              fontSize: 24,
              boxShadow: "0 8px 16px -4px rgba(79, 70, 229, 0.35)",
              marginBottom: 16,
            }}
          >
            ⚛
          </div>
          <div className="eyebrow" style={{ letterSpacing: "0.08em", marginBottom: 6 }}>
            LUMEN RESEARCH
          </div>
          <h1 style={{ fontSize: "1.75rem", fontWeight: 800, color: "var(--ink)", margin: "0 0 8px" }}>
            Welcome Back
          </h1>
          <p className="panel-copy" style={{ margin: 0, fontSize: 13, color: "var(--ink-secondary)" }}>
            Access your private research library, saved collections, and evidence-grounded synthesis tools.
          </p>
        </div>

        {/* Sign In Form */}
        <form onSubmit={submit} className="auth-form" style={{ display: "grid", gap: 18 }}>
          <label style={{ display: "grid", gap: 6, fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
            <span>Email Address</span>
            <input
              type="email"
              value={email}
              onChange={event => setEmail(event.target.value)}
              placeholder="researcher@university.edu"
              autoComplete="email"
              required
              style={{
                padding: "10px 14px",
                fontSize: 14,
                border: "1px solid var(--line)",
                borderRadius: "var(--radius-sm)",
                background: "var(--surface)",
                color: "var(--ink)",
              }}
            />
          </label>

          <label style={{ display: "grid", gap: 6, fontSize: 13, fontWeight: 600, color: "var(--ink)" }}>
            <span>Password</span>
            <div style={{ position: "relative" }}>
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={event => setPassword(event.target.value)}
                placeholder="Enter password"
                autoComplete="current-password"
                minLength={8}
                required
                style={{
                  width: "100%",
                  padding: "10px 60px 10px 14px",
                  fontSize: 14,
                  border: "1px solid var(--line)",
                  borderRadius: "var(--radius-sm)",
                  background: "var(--surface)",
                  color: "var(--ink)",
                }}
              />
              <button
                type="button"
                onClick={() => setShowPassword(v => !v)}
                style={{
                  position: "absolute",
                  right: 12,
                  top: "50%",
                  transform: "translateY(-50%)",
                  background: "none",
                  border: "none",
                  color: "var(--ink-muted)",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                }}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>

          <button
            type="submit"
            className="primary"
            disabled={busy}
            style={{
              padding: "12px",
              fontSize: 14,
              fontWeight: 700,
              width: "100%",
              marginTop: 6,
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            {busy ? "Signing in..." : "Sign In to Workspace"}
          </button>
        </form>

        {error && (
          <p
            className="error"
            role="alert"
            style={{
              marginTop: 16,
              padding: "10px 14px",
              borderRadius: "var(--radius-sm)",
              fontSize: 13,
            }}
          >
            {error}
          </p>
        )}

        {/* Quick Fill / Demo Link */}
        <div style={{ marginTop: 20, textAlign: "center", borderTop: "1px solid var(--line)", paddingTop: 18 }}>
          <p style={{ margin: "0 0 12px", fontSize: 13, color: "var(--ink-secondary)" }}>
            New researcher?{" "}
            <Link href="/register" style={{ color: "var(--primary)", fontWeight: 600, textDecoration: "none" }}>
              Create an account
            </Link>
          </p>
          <button
            type="button"
            onClick={fillDemo}
            style={{
              background: "none",
              border: "none",
              fontSize: 12,
              color: "var(--ink-muted)",
              cursor: "pointer",
              textDecoration: "underline",
            }}
          >
            Fill Demo Account
          </button>
        </div>
      </div>
    </main>
  );
}
