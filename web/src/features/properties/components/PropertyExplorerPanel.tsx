"use client";

import { useEffect, useMemo, useState } from "react";

import PropertyMap from "@/src/features/properties/components/PropertyMap";
import PropertySearchResults from "@/src/features/properties/components/PropertySearchResults";
import {
  fetchNearbyProperties,
  fetchPropertyDetail
} from "@/src/features/properties/services/propertiesApi";
import type {
  CanonicalPropertyDetailResponse,
  CanonicalPropertyPreview
} from "@/src/features/properties/schemas/propertySchemas";

type PropertyExplorerPanelProps = {
  open: boolean;
  onClose: () => void;
  onUseProperty: (property: {
    property_id: string;
    address: string;
    borough: string;
    property_segment: string;
    gross_square_feet: number | null;
    year_built: number | null;
    residential_units: number | null;
    total_units: number | null;
    building_class: string | null;
    sale_date: string | null;
  }) => void;
};

const NYC_BBOX = {
  minLng: -74.3,
  minLat: 40.48,
  maxLng: -73.68,
  maxLat: 40.95
};

type BrowseMode = "viewport" | "all";
type ExplorerFilters = {
  borough: string;
  segment: string;
  zipCode: string;
  tier: string;
};

export default function PropertyExplorerPanel({
  open,
  onClose,
  onUseProperty
}: PropertyExplorerPanelProps) {
  const [browseMode, setBrowseMode] = useState<BrowseMode>("viewport");
  const [filters, setFilters] = useState<ExplorerFilters>({
    borough: "",
    segment: "",
    zipCode: "",
    tier: ""
  });
  const [nearbyItems, setNearbyItems] = useState<CanonicalPropertyPreview[]>([]);
  const [nearbyAvailable, setNearbyAvailable] = useState(0);
  const [catalogTotal, setCatalogTotal] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<CanonicalPropertyDetailResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeItems = nearbyItems;

  const boroughOptions = useMemo(
    () =>
      Array.from(new Set(nearbyItems.map((item) => item.borough).filter(Boolean)))
        .sort(),
    [nearbyItems]
  );
  const segmentOptions = useMemo(
    () =>
      Array.from(new Set(nearbyItems.map((item) => item.property_segment).filter(Boolean)))
        .sort(),
    [nearbyItems]
  );
  const zipOptions = useMemo(
    () =>
      Array.from(new Set(nearbyItems.map((item) => item.zip_code).filter((value): value is string => Boolean(value))))
        .sort(),
    [nearbyItems]
  );
  const tierOptions = useMemo(
    () =>
      Array.from(
        new Set(nearbyItems.map((item) => item.price_tier_proxy).filter((value): value is string => Boolean(value)))
      ).sort(),
    [nearbyItems]
  );

  const shownCount = activeItems.length;
  const availableCount = nearbyAvailable;
  const counterLabel = `Showing ${shownCount} of ${availableCount} properties (catalog ${catalogTotal})`;

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setPending(true);
    fetchNearbyProperties({
      bbox: NYC_BBOX,
      limit: browseMode === "all" ? 2_000 : 450,
      scope: browseMode,
      borough: filters.borough || undefined,
      segment: filters.segment || undefined,
      zipCode: filters.zipCode || undefined,
      tier: filters.tier || undefined
    })
      .then((response) => {
        if (cancelled) return;
        setNearbyItems(response.items);
        setNearbyAvailable(response.total_available);
        setCatalogTotal(response.total_catalog);
        setError(null);
        if (response.items[0]) {
          setSelectedId(response.items[0].property_id);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Unable to load properties");
      })
      .finally(() => {
        if (!cancelled) setPending(false);
      });

    return () => {
      cancelled = true;
    };
  }, [open, browseMode, filters]);

  useEffect(() => {
    if (!selectedId || !open) return;
    let cancelled = false;
    fetchPropertyDetail(selectedId)
      .then((detail) => {
        if (!cancelled) {
          setSelectedDetail(detail);
          setError(null);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Unable to load property detail");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId, open]);

  useEffect(() => {
    if (!open) return;
    const originalOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.overflow = originalOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  async function handleViewportBounds(bbox: {
    minLng: number;
    minLat: number;
    maxLng: number;
    maxLat: number;
  }) {
    if (browseMode === "all") return;
    try {
      const response = await fetchNearbyProperties({
        bbox,
        limit: 450,
        scope: "viewport",
        borough: filters.borough || undefined,
        segment: filters.segment || undefined,
        zipCode: filters.zipCode || undefined,
        tier: filters.tier || undefined
      });
      setNearbyItems(response.items);
      setNearbyAvailable(response.total_available);
      setCatalogTotal(response.total_catalog);
    } catch {
      // Nearby refresh is best-effort to keep map interaction responsive.
    }
  }

  function handleUseProperty() {
    if (!selectedDetail) return;
    onUseProperty({
      property_id: selectedDetail.property_id,
      address: selectedDetail.address,
      borough: selectedDetail.borough,
      property_segment: selectedDetail.property_segment,
      gross_square_feet: selectedDetail.features.gross_square_feet,
      year_built: selectedDetail.features.year_built,
      residential_units: selectedDetail.features.residential_units,
      total_units: selectedDetail.features.total_units,
      building_class: selectedDetail.features.building_class,
      sale_date: selectedDetail.features.sale_date
    });
  }

  if (!open) return null;

  return (
    <div className="property-explorer-overlay" role="dialog" aria-modal="true" aria-label="Property explorer">
      <div className="property-explorer-panel">
        <header className="property-explorer-header">
          <div>
            <h2>Property Explorer</h2>
            <p className="muted">
              Search and map-filter canonical properties, then prefill valuation inputs in one click.
            </p>
          </div>
          <button type="button" className="secondary-btn" onClick={onClose}>
            Close
          </button>
        </header>

        <section className="property-explorer-body">
          <aside className="property-explorer-column search">
            <div className="property-search-form">
              <label>
                Borough
                <select
                  value={filters.borough}
                  onChange={(event) =>
                    setFilters((prev) => ({
                      ...prev,
                      borough: event.target.value
                    }))
                  }
                >
                  <option value="">All boroughs</option>
                  {boroughOptions.map((borough) => (
                    <option key={borough} value={borough}>
                      {borough}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Property type
                <select
                  value={filters.segment}
                  onChange={(event) =>
                    setFilters((prev) => ({
                      ...prev,
                      segment: event.target.value
                    }))
                  }
                >
                  <option value="">All segments</option>
                  {segmentOptions.map((segment) => (
                    <option key={segment} value={segment}>
                      {segment}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Price tier
                <select
                  value={filters.tier}
                  onChange={(event) =>
                    setFilters((prev) => ({
                      ...prev,
                      tier: event.target.value
                    }))
                  }
                >
                  <option value="">All tiers</option>
                  {tierOptions.map((tier) => (
                    <option key={tier} value={tier}>
                      {tier}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                ZIP code
                <select
                  value={filters.zipCode}
                  onChange={(event) =>
                    setFilters((prev) => ({
                      ...prev,
                      zipCode: event.target.value
                    }))
                  }
                >
                  <option value="">All ZIPs</option>
                  {zipOptions.map((zipCode) => (
                    <option key={zipCode} value={zipCode}>
                      {zipCode}
                    </option>
                  ))}
                </select>
              </label>
              <div className="button-row">
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() => {
                    setFilters({
                      borough: "",
                      segment: "",
                      zipCode: "",
                      tier: ""
                    });
                  }}
                >
                  Reset filters
                </button>
              </div>
            </div>

            <div className="property-mode-toggle" role="group" aria-label="Property loading mode">
              <button
                type="button"
                className={browseMode === "viewport" ? "primary-btn" : "secondary-btn"}
                onClick={() => setBrowseMode("viewport")}
                disabled={pending}
              >
                Viewport mode
              </button>
              <button
                type="button"
                className={browseMode === "all" ? "primary-btn" : "secondary-btn"}
                onClick={() => setBrowseMode("all")}
                disabled={pending}
              >
                Show all sample
              </button>
            </div>
            <p className="property-counter muted">{counterLabel}</p>

            <PropertySearchResults
              items={activeItems}
              selectedId={selectedId}
              onSelect={(propertyId) => setSelectedId(propertyId)}
            />
          </aside>

          <div className="property-explorer-column map">
            <PropertyMap
              points={activeItems}
              selectedId={selectedId}
              onSelect={(propertyId) => setSelectedId(propertyId)}
              onViewportBounds={handleViewportBounds}
            />
          </div>

          <aside className="property-explorer-column detail">
            <div className="card property-detail-card">
              <h3>Selected Property</h3>
              {selectedDetail ? (
                <>
                  <p className="property-detail-address">{selectedDetail.address}</p>
                  <p className="muted">
                    {selectedDetail.borough} · {selectedDetail.property_segment}
                  </p>
                  <ul className="property-detail-metrics">
                    <li>
                      <span>Gross sqft</span>
                      <strong>{selectedDetail.features.gross_square_feet ?? "-"}</strong>
                    </li>
                    <li>
                      <span>Year built</span>
                      <strong>{selectedDetail.features.year_built ?? "-"}</strong>
                    </li>
                    <li>
                      <span>Total units</span>
                      <strong>{selectedDetail.features.total_units ?? "-"}</strong>
                    </li>
                    <li>
                      <span>Class</span>
                      <strong>{selectedDetail.features.building_class ?? "-"}</strong>
                    </li>
                    <li>
                      <span>Completeness</span>
                      <strong>{Math.round(selectedDetail.feature_completeness * 100)}%</strong>
                    </li>
                  </ul>
                  {!selectedDetail.availability.inference_ready ? (
                    <p className="warn-text">
                      Missing features: {selectedDetail.availability.missing_required_features.join(", ")}
                    </p>
                  ) : null}
                  <button
                    type="button"
                    className="primary-btn"
                    onClick={handleUseProperty}
                    disabled={!selectedDetail.availability.inference_ready}
                  >
                    Use This Property
                  </button>
                </>
              ) : (
                <p className="muted">Select a property from the list or map to continue.</p>
              )}
              {error ? <p className="error-text">{error}</p> : null}
            </div>
          </aside>
        </section>
      </div>
    </div>
  );
}
