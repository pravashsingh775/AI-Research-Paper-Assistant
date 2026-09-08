"use client";

import { useEffect, useState } from "react";
import { api } from "../lib/api";

interface AISettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const PROVIDER_MODELS: Record<string, { id: string; name: string; badge?: string }[]> = {
  gemini: [
    { id: "gemini-1.5-flash", name: "Gemini 1.5 Flash (Recommended)", badge: "Fast & Free" },
    { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", badge: "Deep Reasoning" },
    { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", badge: "Next Gen" },
  ],
  openai: [
    { id: "gpt-4o-mini", name: "GPT-4o Mini", badge: "Fast" },
    { id: "gpt-4o", name: "GPT-4o", badge: "Capable" },
  ],
  anthropic: [
    { id: "claude-sonnet-4-5-20250929", name: "Claude Sonnet 4.5", badge: "Flagship" },
    { id: "claude-3-5-haiku-20241022", name: "Claude 3.5 Haiku", badge: "Fast" },
  ],
};

export default function AISettingsModal({ isOpen, onClose }: AISettingsModalProps) {
  const [provider, setProvider] = useState("gemini");
  const [model, setModel] = useState("gemini-1.5-flash");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [testing, setTesting] = useState(false);
  const [statusMessage, setStatusMessage] = useState<{ type: "success" | "error" | "info"; text: string } | null>(null);
  const [serverStatus, setServerStatus] = useState<{ server_key_configured: boolean; default_provider?: string } | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    const storedKey = localStorage.getItem("research_llm_key") || "";
    const storedProvider = localStorage.getItem("research_llm_provider") || "gemini";
    const storedModel = localStorage.getItem("research_llm_model") || (PROVIDER_MODELS[storedProvider]?.[0]?.id || "gemini-1.5-flash");

    setApiKey(storedKey);
    setProvider(storedProvider);
    setModel(storedModel);
    setStatusMessage(null);

    // Fetch server status
    api<{ server_key_configured: boolean; default_provider?: string }>("/api/settings/ai-status")
      .then(setServerStatus)
      .catch(() => undefined);
  }, [isOpen]);

  if (!isOpen) return null;

  function handleProviderChange(newProvider: string) {
    setProvider(newProvider);
    const defaultM = PROVIDER_MODELS[newProvider]?.[0]?.id || "";
    setModel(defaultM);
    setStatusMessage(null);
  }

  async function handleVerifyAndSave() {
    if (!apiKey.trim()) {
      setStatusMessage({ type: "error", text: "Please enter a valid API key to test and activate." });
      return;
    }
    setTesting(true);
    setStatusMessage(null);
    try {
      const res = await api<{ valid: boolean; message: string }>("/api/settings/verify-key", {
        method: "POST",
        body: JSON.stringify({
          api_key: apiKey.trim(),
          provider,
          model,
        }),
      });

      if (res.valid) {
        localStorage.setItem("research_llm_key", apiKey.trim());
        localStorage.setItem("research_llm_provider", provider);
        localStorage.setItem("research_llm_model", model);
        window.dispatchEvent(new Event("llm-settings-updated"));
        setStatusMessage({
          type: "success",
          text: `✓ Successfully verified! ${res.message}`,
        });
      } else {
        setStatusMessage({
          type: "error",
          text: `Verification failed: ${res.message}`,
        });
      }
    } catch (err: any) {
      setStatusMessage({
        type: "error",
        text: err?.message || "Failed to verify API key with backend service.",
      });
    } finally {
      setTesting(false);
    }
  }

  function handleClearKey() {
    localStorage.removeItem("research_llm_key");
    localStorage.removeItem("research_llm_provider");
    localStorage.removeItem("research_llm_model");
    setApiKey("");
    window.dispatchEvent(new Event("llm-settings-updated"));
    setStatusMessage({
      type: "info",
      text: "API key cleared. Lumen will operate in Offline Heuristic Mode (TF-IDF & Lexical Synthesis).",
    });
  }

  const hasActiveKey = typeof window !== "undefined" && !!localStorage.getItem("research_llm_key");

  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "rgba(15, 23, 42, 0.65)",
        backdropFilter: "blur(6px)",
        padding: "16px",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        style={{
          backgroundColor: "var(--surface)",
          borderRadius: "var(--radius-xl)",
          boxShadow: "var(--shadow-xl)",
          border: "1px solid var(--border)",
          width: "100%",
          maxWidth: "540px",
          padding: "28px",
          display: "flex",
          flexDirection: "column",
          gap: "20px",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "12px" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "22px" }}>⚡</span>
              <h3 style={{ margin: 0, fontSize: "19px", fontWeight: 700, color: "var(--ink)" }}>
                AI Engine & Reasoning Settings
              </h3>
            </div>
            <p style={{ margin: "4px 0 0", fontSize: "13px", color: "var(--muted)" }}>
              Power multi-paper synthesis, deep RAG chat, and idea generation with state-of-the-art LLMs.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            style={{
              background: "transparent",
              border: "none",
              fontSize: "20px",
              cursor: "pointer",
              color: "var(--muted)",
              padding: "4px 8px",
              borderRadius: "var(--radius-sm)",
            }}
            title="Close modal"
          >
            ✕
          </button>
        </div>

        {/* Current Active Mode Banner */}
        <div
          style={{
            padding: "12px 14px",
            borderRadius: "var(--radius-md)",
            border: hasActiveKey
              ? "1px solid var(--emerald-border)"
              : serverStatus?.server_key_configured
              ? "1px solid var(--cyan-border)"
              : "1px solid var(--border)",
            backgroundColor: hasActiveKey
              ? "var(--emerald-light)"
              : serverStatus?.server_key_configured
              ? "var(--cyan-light)"
              : "var(--surface-muted)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span
              style={{
                width: "10px",
                height: "10px",
                borderRadius: "50%",
                backgroundColor: hasActiveKey
                  ? "var(--emerald)"
                  : serverStatus?.server_key_configured
                  ? "var(--cyan)"
                  : "var(--muted)",
                display: "inline-block",
              }}
            />
            <div>
              <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--ink)" }}>
                {hasActiveKey
                  ? `Custom ${provider.toUpperCase()} Key Active (${model})`
                  : serverStatus?.server_key_configured
                  ? `Server Default AI Active (${serverStatus.default_provider || "Gemini"})`
                  : "Offline Deterministic Mode"}
              </div>
              <div style={{ fontSize: "11px", color: "var(--muted)" }}>
                {hasActiveKey || serverStatus?.server_key_configured
                  ? "High-precision generative synthesis & evidence citation enabled"
                  : "No API key configured. Using local academic heuristics."}
              </div>
            </div>
          </div>
          {hasActiveKey && (
            <button
              type="button"
              onClick={handleClearKey}
              style={{
                fontSize: "11px",
                padding: "4px 8px",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-xs)",
                cursor: "pointer",
                color: "var(--rose)",
              }}
            >
              Disconnect
            </button>
          )}
        </div>

        {/* Google Gemini Free Key Callout */}
        <div
          style={{
            padding: "12px 14px",
            borderRadius: "var(--radius-md)",
            background: "linear-gradient(135deg, rgba(79, 70, 229, 0.08) 0%, rgba(124, 58, 237, 0.08) 100%)",
            border: "1px solid var(--primary-border)",
            display: "flex",
            flexDirection: "column",
            gap: "6px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--primary)" }}>
              💎 Recommended: Google Gemini (Free Tier)
            </span>
            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: "12px",
                fontWeight: 600,
                color: "var(--primary)",
                textDecoration: "underline",
                display: "flex",
                alignItems: "center",
                gap: "2px",
              }}
            >
              Get Free Gemini Key ↗
            </a>
          </div>
          <p style={{ margin: 0, fontSize: "11.5px", color: "var(--ink-secondary)", lineHeight: 1.4 }}>
            Google AI Studio offers free API keys with generous rate limits. No credit card required. Paste your key below to unlock full academic synthesis!
          </p>
        </div>

        {/* Provider and Model selectors */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
          <div>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "var(--ink)", marginBottom: "6px" }}>
              AI Provider
            </label>
            <select
              value={provider}
              onChange={(e) => handleProviderChange(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                fontSize: "13px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
                backgroundColor: "var(--surface)",
                color: "var(--ink)",
              }}
            >
              <option value="gemini">Google Gemini (Recommended)</option>
              <option value="openai">OpenAI</option>
              <option value="anthropic">Anthropic Claude</option>
            </select>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "12px", fontWeight: 600, color: "var(--ink)", marginBottom: "6px" }}>
              Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                width: "100%",
                padding: "8px 12px",
                fontSize: "13px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border)",
                backgroundColor: "var(--surface)",
                color: "var(--ink)",
              }}
            >
              {(PROVIDER_MODELS[provider] || []).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} {m.badge ? `[${m.badge}]` : ""}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* API Key Input */}
        <div>
          <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "12px", fontWeight: 600, color: "var(--ink)", marginBottom: "6px" }}>
            <span>{provider === "gemini" ? "Google Gemini API Key" : provider === "openai" ? "OpenAI API Key" : "Anthropic API Key"}</span>
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              style={{
                background: "transparent",
                border: "none",
                fontSize: "11px",
                color: "var(--primary)",
                cursor: "pointer",
                padding: 0,
              }}
            >
              {showKey ? "Hide" : "Show"}
            </button>
          </label>
          <input
            type={showKey ? "text" : "password"}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={
              provider === "gemini"
                ? "AIzaSy..."
                : provider === "openai"
                ? "sk-proj-..."
                : "sk-ant-..."
            }
            style={{
              width: "100%",
              padding: "10px 12px",
              fontSize: "13px",
              fontFamily: "var(--font-mono)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)",
              backgroundColor: "var(--surface)",
              color: "var(--ink)",
            }}
          />
          <div style={{ display: "flex", alignItems: "center", gap: "4px", marginTop: "6px" }}>
            <span style={{ fontSize: "12px" }}>🔒</span>
            <span style={{ fontSize: "11px", color: "var(--muted)" }}>
              Stored securely in your browser. Transmitted directly to backend via header.
            </span>
          </div>
        </div>

        {/* Status Message */}
        {statusMessage && (
          <div
            style={{
              padding: "10px 14px",
              borderRadius: "var(--radius-sm)",
              fontSize: "12px",
              backgroundColor:
                statusMessage.type === "success"
                  ? "var(--emerald-light)"
                  : statusMessage.type === "error"
                  ? "var(--rose-light)"
                  : "var(--surface-muted)",
              color:
                statusMessage.type === "success"
                  ? "var(--emerald)"
                  : statusMessage.type === "error"
                  ? "var(--rose)"
                  : "var(--ink)",
              border: `1px solid ${
                statusMessage.type === "success"
                  ? "var(--emerald-border)"
                  : statusMessage.type === "error"
                  ? "var(--rose-border)"
                  : "var(--border)"
              }`,
            }}
          >
            {statusMessage.text}
          </div>
        )}

        {/* Footer Actions */}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px", marginTop: "4px" }}>
          <button
            type="button"
            className="secondary"
            onClick={onClose}
            style={{ fontSize: "13px", padding: "8px 16px" }}
          >
            Close
          </button>
          <button
            type="button"
            className="primary"
            onClick={handleVerifyAndSave}
            disabled={testing}
            style={{
              fontSize: "13px",
              padding: "8px 18px",
              display: "flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            {testing ? "Verifying..." : "Verify & Connect"}
          </button>
        </div>
      </div>
    </div>
  );
}

