"use client";

import type { CanonicalPropertyPreview } from "@/src/features/properties/schemas/propertySchemas";

type PropertySearchResultsProps = {
  items: CanonicalPropertyPreview[];
  selectedId: string | null;
  onSelect: (propertyId: string) => void;
};

export default function PropertySearchResults({
  items,
  selectedId,
  onSelect
}: PropertySearchResultsProps) {
  if (!items.length) {
    return (
      <div className="property-results-empty">
        <p className="muted">No matching properties found in the current canonical sample.</p>
      </div>
    );
  }

  return (
    <ul className="property-results-list">
      {items.map((item) => (
        <li key={item.property_id}>
          <button
            type="button"
            onClick={() => onSelect(item.property_id)}
            className={`property-result-item ${item.property_id === selectedId ? "selected" : ""}`}
          >
            <span className="property-result-address">{item.address}</span>
            <span className="property-result-meta">
              {item.borough}
              {item.zip_code ? ` · ${item.zip_code}` : ""} · {item.property_segment}
              {item.price_tier_proxy ? ` · ${item.price_tier_proxy}` : ""}
            </span>
            <span className={`property-result-quality ${item.data_quality_status}`}>
              {item.data_quality_status.toUpperCase()}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}
