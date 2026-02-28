import { NextResponse } from "next/server";

export async function GET() {
  const apiBase = process.env.API_URL;
  if (!apiBase) {
    return NextResponse.json({ error: "API_URL is not set" }, { status: 500 });
  }

  const upstream = await fetch(`${apiBase}/options`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });

  const text = await upstream.text();
  const contentType = upstream.headers.get("content-type") ?? "application/json";

  return new NextResponse(text, {
    status: upstream.status,
    headers: { "content-type": contentType },
  });
}
