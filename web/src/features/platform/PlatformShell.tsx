import { headers } from "next/headers";

import { BRAND_HEADER, getBrandProfile, parseBrandId } from "@/src/lib/brand";
import type { PlatformData } from "@/src/features/platform/data";
import { PlatformFrame } from "@/src/features/platform/PlatformFrame";

export function PlatformShell({
  data,
  children
}: {
  data: PlatformData;
  children: React.ReactNode;
}) {
  const brandId = parseBrandId(headers().get(BRAND_HEADER));
  const profile = getBrandProfile(brandId);

  return (
    <PlatformFrame appName={profile.appName} data={data}>
      {children}
    </PlatformFrame>
  );
}
