"use client";

import { createContext, useContext } from "react";

import type { BrandId, BrandProfile } from "@/src/lib/brand";
import { DEFAULT_BRAND } from "@/src/lib/brand";

type BrandContextValue = {
  brandId: BrandId;
  profile: BrandProfile;
};

const BrandContext = createContext<BrandContextValue>({
  brandId: "default",
  profile: DEFAULT_BRAND
});

export function BrandProvider({
  brandId,
  profile,
  children
}: {
  brandId: BrandId;
  profile: BrandProfile;
  children: React.ReactNode;
}) {
  return <BrandContext.Provider value={{ brandId, profile }}>{children}</BrandContext.Provider>;
}

export function useBrandProfile() {
  return useContext(BrandContext);
}
