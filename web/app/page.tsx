import Link from "next/link";
import { headers } from "next/headers";

import { BRAND_HEADER, getBrandProfile, parseBrandId } from "@/src/lib/brand";

export default function HomePage() {
  const brandId = parseBrandId(headers().get(BRAND_HEADER));
  const profile = getBrandProfile(brandId);

  return (
    <main className="landing-shell">
      <section className="card fade-in-up">
        <h1>{profile.appName}</h1>
        <p className="muted">Canonical, datasource-agnostic valuation workflows for production decisioning.</p>
        <div className="button-row">
          <Link href="/valuation/single" className="primary-btn">
            Open Dashboard
          </Link>
        </div>
      </section>
    </main>
  );
}
