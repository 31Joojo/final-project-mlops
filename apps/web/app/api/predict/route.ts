import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const apiBase = process.env.API_URL;
  if (!apiBase) {
    return NextResponse.json({ error: "API_URL is not set" }, { status: 500 });
  }

  const payload = await req.json();

  const upstream = await fetch(`${apiBase}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const text = await upstream.text();
  const contentType = upstream.headers.get("content-type") ?? "application/json";

  return new NextResponse(text, {
    status: upstream.status,
    headers: { "content-type": contentType }
  });
}
