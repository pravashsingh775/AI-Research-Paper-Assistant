const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) { super(message); }
}

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = typeof window === "undefined" ? "" : localStorage.getItem("research_token") || "";
  const headers = new Headers(options.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  // The browser must set FormData's multipart boundary itself.
  if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  let response: Response;
  try {
    response = await fetch(`${API}${path}`, { ...options, headers });
  } catch {
    throw new ApiError(0, "Unable to reach the service. Check your connection and try again.");
  }
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) {
    if (typeof window !== "undefined") localStorage.removeItem("research_token");
    throw new ApiError(401, "Your session has expired. Please sign in again.");
  }
  if (!response.ok) {
    const fallback = response.status === 403 ? "You do not have permission to perform this action."
      : response.status === 404 ? "The requested resource was not found."
      : response.status === 409 ? "This action conflicts with the current resource state."
      : response.status === 422 ? "Please check the submitted information and try again."
      : response.status >= 500 ? "The service encountered an error. Please try again shortly."
      : "Request failed.";
    throw new ApiError(response.status, typeof data.detail === "string" ? data.detail : fallback);
  }
  return data as T;
}

export type WorkspacePaper = { id: string; title: string; summary?: string; authors_raw?: string; year?: number; venue?: string; citation_count?: number; source?: string; evidence_state?: string; analysis?: Record<string, unknown> };
export async function loadPapers() { return api<WorkspacePaper[]>("/api/papers"); }
