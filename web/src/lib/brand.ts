export type BrandId = "default" | "azuli";

export type BrandProfile = {
  id: BrandId;
  appName: string;
  navSubtitle: string;
  logoPath?: string;
  logoAlt?: string;
};

export const BRAND_HEADER = "x-spec-brand";

export const DEFAULT_BRAND: BrandProfile = {
  id: "default",
  appName: "AVM Intelligence Console",
  navSubtitle: "Valuation Intelligence"
};

export const AZULI_BRAND: BrandProfile = {
  id: "azuli",
  appName: "Azuli AVM Dashboard",
  navSubtitle: "Valuation Intelligence",
  logoPath: "/brand/azuli-logo-white.png",
  logoAlt: "Azuli"
};

const BRAND_PROFILES: Record<BrandId, BrandProfile> = {
  default: DEFAULT_BRAND,
  azuli: AZULI_BRAND
};

export function parseBrandId(raw: string | null | undefined): BrandId {
  if (raw === "azuli") {
    return "azuli";
  }
  return "default";
}

export function getBrandProfile(brandId: BrandId): BrandProfile {
  return BRAND_PROFILES[brandId];
}
