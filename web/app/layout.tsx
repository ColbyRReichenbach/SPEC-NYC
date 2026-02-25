import type { Metadata } from "next";
import { headers } from "next/headers";
import { Instrument_Sans, Inter } from "next/font/google";

import { BrandProvider } from "@/src/components/providers/BrandProvider";
import { BRAND_HEADER, getBrandProfile, parseBrandId } from "@/src/lib/brand";

import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  variable: "--font-display",
  subsets: ["latin"]
});

const inter = Inter({
  variable: "--font-body",
  subsets: ["latin"]
});

export const metadata: Metadata = {
  title: "AVM Intelligence Dashboard",
  description: "Datasource-agnostic AVM dashboard with governance and monitoring controls"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const brandId = parseBrandId(headers().get(BRAND_HEADER));
  const profile = getBrandProfile(brandId);

  return (
    <html lang="en" className={`${instrumentSans.variable} ${inter.variable}`}>
      <body data-brand={brandId}>
        <BrandProvider brandId={brandId} profile={profile}>
          {children}
        </BrandProvider>
      </body>
    </html>
  );
}
