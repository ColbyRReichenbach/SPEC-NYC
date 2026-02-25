import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

import { BRAND_HEADER, type BrandId } from "@/src/lib/brand";

const ACCESS_QUERY_KEY = "access";
const ACCESS_COOKIE_KEY = "spec_brand_access";

type BrandTokenPayload = {
  brand: "azuli";
  exp: number;
};

function parseBrandEnv(value: string | undefined): BrandId {
  return value === "azuli" ? "azuli" : "default";
}

function parseBoolEnv(value: string | undefined, fallback: boolean): boolean {
  if (!value) {
    return fallback;
  }
  return value.toLowerCase() === "true";
}

function parseHostList(raw: string | undefined): string[] {
  if (!raw) {
    return [];
  }
  return raw
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);
}

function hostFromRequest(req: NextRequest) {
  return (req.headers.get("x-forwarded-host") ?? req.headers.get("host") ?? "").split(":")[0].toLowerCase();
}

function base64UrlToBytes(value: string): Uint8Array | null {
  try {
    const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  } catch {
    return null;
  }
}

function bytesToBase64Url(bytes: Uint8Array) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function timingSafeEqual(a: string, b: string) {
  if (a.length !== b.length) {
    return false;
  }
  let diff = 0;
  for (let index = 0; index < a.length; index += 1) {
    diff |= a.charCodeAt(index) ^ b.charCodeAt(index);
  }
  return diff === 0;
}

async function signPayload(payloadBase64: string, secret: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(payloadBase64));
  return bytesToBase64Url(new Uint8Array(signature));
}

async function verifyToken(rawToken: string | undefined, secret: string): Promise<BrandTokenPayload | null> {
  if (!rawToken) {
    return null;
  }

  const [payloadBase64, signature] = rawToken.split(".");
  if (!payloadBase64 || !signature) {
    return null;
  }

  const expectedSignature = await signPayload(payloadBase64, secret);
  if (!timingSafeEqual(signature, expectedSignature)) {
    return null;
  }

  const payloadBytes = base64UrlToBytes(payloadBase64);
  if (!payloadBytes) {
    return null;
  }

  try {
    const payload = JSON.parse(new TextDecoder().decode(payloadBytes)) as Partial<BrandTokenPayload>;
    if (payload.brand !== "azuli") {
      return null;
    }
    if (typeof payload.exp !== "number" || !Number.isFinite(payload.exp)) {
      return null;
    }
    if (payload.exp < Math.floor(Date.now() / 1000)) {
      return null;
    }
    return { brand: "azuli", exp: payload.exp };
  } catch {
    return null;
  }
}

function withBrandHeader(req: NextRequest, brand: BrandId) {
  const requestHeaders = new Headers(req.headers);
  requestHeaders.set(BRAND_HEADER, brand);
  return requestHeaders;
}

function denyAccess() {
  return new NextResponse("Brand access denied.", { status: 403 });
}

export async function middleware(req: NextRequest) {
  const forcedBrand = parseBrandEnv(process.env.APP_BRAND);
  const requireToken = parseBoolEnv(process.env.BRAND_REQUIRE_TOKEN_FOR_AZULI, true);
  const brandHosts = parseHostList(process.env.AZULI_BRAND_HOSTS);
  const secret = process.env.BRAND_ACCESS_SECRET;
  const host = hostFromRequest(req);
  const isAzuliHost = brandHosts.includes(host);
  const wantsAzuli = forcedBrand === "azuli" || isAzuliHost;

  if (!wantsAzuli) {
    return NextResponse.next({
      request: { headers: withBrandHeader(req, "default") }
    });
  }

  if (!requireToken) {
    return NextResponse.next({
      request: { headers: withBrandHeader(req, "azuli") }
    });
  }

  if (!secret) {
    return denyAccess();
  }

  const queryToken = req.nextUrl.searchParams.get(ACCESS_QUERY_KEY) ?? undefined;
  const cookieToken = req.cookies.get(ACCESS_COOKIE_KEY)?.value;
  const verifiedFromQuery = await verifyToken(queryToken, secret);
  const verifiedFromCookie = await verifyToken(cookieToken, secret);
  const verified = verifiedFromQuery ?? verifiedFromCookie;
  if (!verified) {
    return denyAccess();
  }

  if (verifiedFromQuery) {
    const redirectUrl = req.nextUrl.clone();
    redirectUrl.searchParams.delete(ACCESS_QUERY_KEY);
    const response = NextResponse.redirect(redirectUrl);
    response.cookies.set(ACCESS_COOKIE_KEY, queryToken as string, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      path: "/",
      expires: new Date(verified.exp * 1000)
    });
    return response;
  }

  return NextResponse.next({
    request: { headers: withBrandHeader(req, "azuli") }
  });
}

export const config = {
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico).*)"]
};
