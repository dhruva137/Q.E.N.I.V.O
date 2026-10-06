let token = "";
let user = "";

export function setAuth(next: { token?: string; user?: string }): void {
  if (next.token !== undefined) token = next.token;
  if (next.user !== undefined) user = next.user;
}

export async function request(path: string, body?: unknown): Promise<unknown> {
  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  if (user) headers["X-QENIVO-User"] = user;
  let payload: string | undefined;
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const response = await fetch(path, { method: body !== undefined ? "POST" : "GET", headers, body: payload });
  const text = await response.text();
  let data: unknown = null;
  if (text) {
    try {
      data = JSON.parse(text) as unknown;
    } catch {
      throw new Error(text.slice(0, 400) || response.statusText);
    }
  }
  const record = data != null && typeof data === "object" && !Array.isArray(data)
    ? data as Record<string, unknown>
    : null;
  if (!response.ok || (record && typeof record.error === "string")) {
    const message = record && typeof record.error === "string" ? record.error : response.statusText;
    throw new Error(message);
  }
  return data;
}
