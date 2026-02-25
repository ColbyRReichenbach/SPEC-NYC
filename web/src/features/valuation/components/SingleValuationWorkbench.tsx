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
import ShapGlobalSummary from "@/src/features/valuation/components/ShapGlobalSummary";
import ShapWaterfall from "@/src/features/valuation/components/ShapWaterfall";
import { requestSingleValuation } from "@/src/features/valuation/services/valuationApi";
import type { SingleValuationResponse } from "@/src/features/valuation/schemas/valuationSchemas";

const NYC_BBOX = {
  minLng: -74.3,
  minLat: 40.48,
  maxLng: -73.68,
  maxLat: 40.95
};
const LIST_MAX_ITEMS = 80;

type BrowseMode = "viewport" | "all";
type Filters = {
  borough: string;
  segment: string;
  zipCode: string;
  tier: string;
};

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value);
}

export default function SingleValuationWorkbench() {
  const [browseMode, setBrowseMode] = useState<BrowseMode>("viewport");
  const [filters, setFilters] = useState<Filters>({
    borough: "",
    segment: "",
    zipCode: "",
    tier: ""
  });
  const [items, setItems] = useState<CanonicalPropertyPreview[]>([]);
  const [totalAvailable, setTotalAvailable] = useState(0);
  const [catalogTotal, setCatalogTotal] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<CanonicalPropertyDetailResponse | null>(null);
  const [loadingCatalog, setLoadingCatalog] = useState(false);

  const [pendingValuation, setPendingValuation] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SingleValuationResponse | null>(null);
  const [expertOpen, setExpertOpen] = useState(false);

  const confidenceClass = useMemo(() => {
    if (!result) return "neutral";
    return result.confidence.band;
  }, [result]);
  const listItems = useMemo(() => items.slice(0, LIST_MAX_ITEMS), [items]);

  const boroughOptions = useMemo(
    () => Array.from(new Set(items.map((item) => item.borough).filter(Boolean))).sort(),
    [items]
  );
  const segmentOptions = useMemo(
    () => Array.from(new Set(items.map((item) => item.property_segment).filter(Boolean))).sort(),
    [items]
  );
  const zipOptions = useMemo(
    () => Array.from(new Set(items.map((item) => item.zip_code).filter((value): value is string => Boolean(value)))).sort(),
    [items]
  );
  const tierOptions = useMemo(
    () =>
      Array.from(
        new Set(items.map((item) => item.price_tier_proxy).filter((value): value is string => Boolean(value)))
      ).sort(),
    [items]
  );

  useEffect(() => {
    let cancelled = false;
    setLoadingCatalog(true);
    setError(null);
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
        setItems(response.items);
        setTotalAvailable(response.total_available);
        setCatalogTotal(response.total_catalog);
        setSelectedId((current) => current ?? response.items[0]?.property_id ?? null);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load map properties");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingCatalog(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [browseMode, filters]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedDetail(null);
      return;
    }
    let cancelled = false;
    fetchPropertyDetail(selectedId)
      .then((detail) => {
        if (!cancelled) {
          setSelectedDetail(detail);
        }
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Unable to load selected property details");
        }
      });

    return () => {
      cancelled = true;
    };
  }, [selectedId]);

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
      setItems(response.items);
      setTotalAvailable(response.total_available);
      setCatalogTotal(response.total_catalog);
    } catch {
      // Keep map responsive if a refresh fails.
    }
  }

  async function handleRunValuation() {
    if (!selectedDetail) {
      setError("Select a property on the map before running valuation.");
      return;
    }
    if (!selectedDetail.availability.inference_ready) {
      setError("Selected property is missing required inference features.");
      return;
    }
    if (
      selectedDetail.features.gross_square_feet === null ||
      selectedDetail.features.year_built === null ||
      selectedDetail.features.residential_units === null ||
      selectedDetail.features.total_units === null ||
      selectedDetail.features.building_class === null ||
      selectedDetail.features.sale_date === null
    ) {
      setError("Selected property has incomplete valuation inputs.");
      return;
    }

    setPendingValuation(true);
    setError(null);
    try {
      const response = await requestSingleValuation({
        property: {
          address: selectedDetail.address,
          borough: selectedDetail.borough,
          gross_square_feet: selectedDetail.features.gross_square_feet,
          year_built: selectedDetail.features.year_built,
          residential_units: selectedDetail.features.residential_units,
          total_units: selectedDetail.features.total_units,
          building_class: selectedDetail.features.building_class,
          property_segment: selectedDetail.property_segment,
          sale_date: selectedDetail.features.sale_date
        },
        context: {
          dataset_version: `property_dataset:${selectedDetail.property_id}`,
          model_alias: "champion",
          property_id: selectedDetail.property_id
        }
      });
      setResult(response);
    } catch (valuationError) {
      setResult(null);
      setError(valuationError instanceof Error ? valuationError.message : "Unexpected valuation failure");
    } finally {
      setPendingValuation(false);
    }
  }

  return (
    <section className="stack-lg fade-in-up">
      <div className="card">
        <h1>Map Valuation</h1>
        <p className="muted">Single valuation is now map-first. Select a property and run without manual typing.</p>
      </div>

      <div className="property-explorer-inline">
        <section className="property-explorer-body map-first">
          <aside className="property-explorer-column search">
            <div className="property-search-form">
              <label>
                Borough
                <select
                  value={filters.borough}
                  onChange={(event) => setFilters((prev) => ({ ...prev, borough: event.target.value }))}
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
                  onChange={(event) => setFilters((prev) => ({ ...prev, segment: event.target.value }))}
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
                  onChange={(event) => setFilters((prev) => ({ ...prev, tier: event.target.value }))}
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
                  onChange={(event) => setFilters((prev) => ({ ...prev, zipCode: event.target.value }))}
                >
                  <option value="">All ZIPs</option>
                  {zipOptions.map((zipCode) => (
                    <option key={zipCode} value={zipCode}>
                      {zipCode}
                    </option>
                  ))}
                </select>
              </label>
              <div className="property-mode-toggle" role="group" aria-label="Map loading mode">
                <button
                  type="button"
                  className={browseMode === "viewport" ? "primary-btn" : "secondary-btn"}
                  onClick={() => setBrowseMode("viewport")}
                  disabled={loadingCatalog}
                >
                  Viewport mode
                </button>
                <button
                  type="button"
                  className={browseMode === "all" ? "primary-btn" : "secondary-btn"}
                  onClick={() => setBrowseMode("all")}
                  disabled={loadingCatalog}
                >
                  Show all sample
                </button>
              </div>
              <div className="button-row">
                <button
                  type="button"
                  className="secondary-btn"
                  onClick={() =>
                    setFilters({
                      borough: "",
                      segment: "",
                      zipCode: "",
                      tier: ""
                    })
                  }
                >
                  Reset filters
                </button>
              </div>
            </div>
            <p className="property-counter muted">
              Showing {listItems.length} of {items.length} loaded ({totalAvailable} in view, catalog {catalogTotal})
            </p>
            <details className="property-selected-dropdown" open>
              <summary>Selected property</summary>
              <div className="card property-detail-card">
                {selectedDetail ? (
                  <>
                    <p className="property-detail-address">{selectedDetail.address}</p>
                    <p className="muted">
                      {selectedDetail.borough}
                      {selectedDetail.zip_code ? ` · ${selectedDetail.zip_code}` : ""} · {selectedDetail.property_segment}
                      {selectedDetail.price_tier_proxy ? ` · ${selectedDetail.price_tier_proxy}` : ""}
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
                      onClick={handleRunValuation}
                      disabled={!selectedDetail.availability.inference_ready || pendingValuation}
                    >
                      {pendingValuation ? "Calculating..." : "Run valuation"}
                    </button>
                  </>
                ) : (
                  <p className="muted">Select a property from map or list.</p>
                )}
                {error ? <p className="error-text">{error}</p> : null}
              </div>
            </details>
            <PropertySearchResults
              items={listItems}
              selectedId={selectedId}
              onSelect={(propertyId) => setSelectedId(propertyId)}
            />
          </aside>

          <div className="property-explorer-column map">
            <PropertyMap
              points={items}
              selectedId={selectedId}
              onSelect={(propertyId) => setSelectedId(propertyId)}
              onViewportBounds={handleViewportBounds}
            />
          </div>

        </section>
      </div>

      <div className="valuation-results-stack">
        <div className="card kpi-card">
          <h2>Estimate</h2>
          <p className="kpi-value">{result ? formatCurrency(result.predicted_price) : "-"}</p>
          <p className="muted">
            Interval:{" "}
            {result
              ? `${formatCurrency(result.prediction_interval.low)} – ${formatCurrency(result.prediction_interval.high)}`
              : "-"}
          </p>
        </div>

        <div className={`card confidence-card ${confidenceClass}`}>
          <h2>Confidence</h2>
          <p className="kpi-value">{result ? `${Math.round(result.confidence.score * 100)}%` : "-"}</p>
          <p className="muted">Band: {result?.confidence.band ?? "-"}</p>
          <ul>
            {(result?.confidence.caveats ?? ["Estimate is probabilistic, not an appraisal."]).map((caveat) => (
              <li key={caveat}>{caveat}</li>
            ))}
          </ul>
        </div>

        <div className="card drivers-card">
          <h2>Top Drivers</h2>
          <div className="driver-columns">
            <div>
              <h3>Positive</h3>
              <ul>
                {(result?.explanation.drivers_positive ?? []).map((driver) => (
                  <li key={`${driver.feature}-pos`}>
                    <span>{driver.display}</span>
                    <strong>{formatCurrency(driver.impact)}</strong>
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h3>Negative</h3>
              <ul>
                {(result?.explanation.drivers_negative ?? []).map((driver) => (
                  <li key={`${driver.feature}-neg`}>
                    <span>{driver.display}</span>
                    <strong>{formatCurrency(driver.impact)}</strong>
                  </li>
                ))}
              </ul>
            </div>
          </div>
          <p className="muted">Explainer: {result?.explanation.explainer_type ?? "-"}</p>
        </div>

        <ShapWaterfall valuation={result} />
        <ShapGlobalSummary segment={result?.model.route ?? selectedDetail?.property_segment ?? "SMALL_MULTI"} window="180d" />

        <div className="card expert-card">
          <button className="secondary-btn" onClick={() => setExpertOpen((prev) => !prev)}>
            {expertOpen ? "Hide expert details" : "Show expert details"}
          </button>
          {expertOpen ? (
            <div className="expert-grid">
              <div>
                <span className="context-label">Model Route</span>
                <p>
                  <code>{result?.model.route ?? "-"}</code>
                </p>
              </div>
              <div>
                <span className="context-label">Run ID</span>
                <p>
                  <code>{result?.model.run_id ?? "-"}</code>
                </p>
              </div>
              <div>
                <span className="context-label">Source Context</span>
                <p>
                  <code>{result?.source_context?.source_id ?? "-"}</code>
                </p>
              </div>
              <div>
                <span className="context-label">Evidence</span>
                <ul>
                  <li>
                    <code>{result?.evidence.metrics_path ?? "-"}</code>
                  </li>
                  <li>
                    <code>{result?.evidence.run_card_path ?? "-"}</code>
                  </li>
                </ul>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}
