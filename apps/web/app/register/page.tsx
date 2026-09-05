"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    setError("");

    try {
      const response = await fetch(`${API}/api/auth/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to create account");
      localStorage.setItem("research_token", data.access_token);
      router.push("/");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Unable to create account");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <div className="auth-card">
        <div className="eyebrow">LUMEN RESEARCH</div>
        <h1>Create Account</h1>
        <p className="panel-copy">
          Establish your private research workspace for isolated PDF vector indexing and synthesis.
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
                placeholder="Minimum 8 characters"
                autoComplete="new-password"
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

          <label>
            Confirm Password
            <input
              type={showPassword ? "text" : "password"}
              value={confirmPassword}
              onChange={event => setConfirmPassword(event.target.value)}
              placeholder="Re-enter password"
              autoComplete="new-password"
              minLength={8}
              required
            />
          </label>

          <button className="primary" disabled={busy}>
            {busy ? "Setting up workspace..." : "Create workspace account"}
          </button>
        </form>

        {error && <p className="error" role="alert">{error}</p>}

        <p className="panel-copy" style={{ marginTop: 20 }}>
          Already have an account? <Link href="/login" className="tool-link">Sign in</Link>
        </p>
      </div>
    </main>
  );
}
