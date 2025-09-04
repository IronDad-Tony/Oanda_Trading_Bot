import { NextRequest } from "next/server";

const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export async function GET(req: NextRequest, { params }: { params: { path: string[] } }) {
  const url = new URL(req.url);
  const target = `${BASE}/${params.path.join("/")}${url.search}`;
  const res = await fetch(target, { method: "GET", headers: { "Content-Type": "application/json" } });
  return new Response(await res.text(), { status: res.status, headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" } });
}

export async function POST(req: NextRequest, { params }: { params: { path: string[] } }) {
  const url = new URL(req.url);
  const target = `${BASE}/${params.path.join("/")}${url.search}`;
  const body = await req.text();
  const res = await fetch(target, { method: "POST", headers: { "Content-Type": req.headers.get("Content-Type") || "application/json" }, body });
  return new Response(await res.text(), { status: res.status, headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" } });
}

export async function PUT(req: NextRequest, { params }: { params: { path: string[] } }) {
  const url = new URL(req.url);
  const target = `${BASE}/${params.path.join("/")}${url.search}`;
  const body = await req.text();
  const res = await fetch(target, { method: "PUT", headers: { "Content-Type": req.headers.get("Content-Type") || "application/json" }, body });
  return new Response(await res.text(), { status: res.status, headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" } });
}

export async function DELETE(req: NextRequest, { params }: { params: { path: string[] } }) {
  const url = new URL(req.url);
  const target = `${BASE}/${params.path.join("/")}${url.search}`;
  const res = await fetch(target, { method: "DELETE" });
  return new Response(await res.text(), { status: res.status, headers: { "Content-Type": res.headers.get("Content-Type") || "application/json" } });
}

