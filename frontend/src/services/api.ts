// Thin fetch client for the CogniTrack API. The API key (if auth is enabled) comes from
// VITE_API_KEY at build time or from the key the user enters in the header (sessionStorage).

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";
const KEY_STORAGE = "cognitrack.apiKey";

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

export function getApiKey(): string | null {
  try {
    return sessionStorage.getItem(KEY_STORAGE) ?? (import.meta.env.VITE_API_KEY as string | undefined) ?? null;
  } catch {
    return (import.meta.env.VITE_API_KEY as string | undefined) ?? null;
  }
}

export function setApiKey(key: string | null): void {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable: key lasts for this page only */
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const key = getApiKey();
  if (key) headers.set("X-API-Key", key);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "network", "Cannot reach the CogniTrack API. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let code = "http_error";
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
      headers: body instanceof FormData || body === undefined ? undefined : { "Content-Type": "application/json" },
    }),
  url: (path: string) => `${BASE}${path}`,
};
