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

  return (
    <main className="auth-shell">
      <div className="auth-card">
        <div className="eyebrow">LUMEN RESEARCH</div>
        <h1>Sign In</h1>
        <p className="panel-copy">
          Access your private research library, saved collections, and evidence-grounded chat sessions.
        </p>

        <form onSubmit={submit} className="auth-form">
          <label>
            Email Address
            <input
              type="email"
              value={email}
              onChange={event => setEmail(event.target.value)}
              placeholder="researcher@university.edu"
              autoComplete="email"
              required
            />
          </label>

          <label>
            Password
            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={event => setPassword(event.target.value)}
                placeholder="Enter password"
                autoComplete="current-password"
                minLength={8}
                required
              />
              <button
                type="button"
                className="link-button"
                style={{ fontSize: 13 }}
                onClick={() => setShowPassword(v => !v)}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>

          <button className="primary" disabled={busy}>
            {busy ? "Signing in..." : "Sign in to workspace"}
          </button>
        </form>

        {error && <p className="error" role="alert">{error}</p>}

        <p className="panel-copy" style={{ marginTop: 20 }}>
          New researcher? <Link href="/register" className="tool-link">Create an account</Link>
        </p>
      </div>
    </main>
  );
}
