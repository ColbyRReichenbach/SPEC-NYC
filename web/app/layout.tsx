import type { Metadata } from "next";
import { Instrument_Sans, Inter } from "next/font/google";

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
  title: "Azuli AVM Intelligence",
  description: "Datasource-agnostic AVM dashboard with governance and monitoring controls"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${instrumentSans.variable} ${inter.variable}`}>
      <body>{children}</body>
    </html>
  );
}
