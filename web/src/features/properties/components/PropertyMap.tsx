"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Map, { type MapRef, Marker } from "react-map-gl/maplibre";
import maplibregl from "maplibre-gl";

import type { CanonicalPropertyPreview } from "@/src/features/properties/schemas/propertySchemas";

type BBox = {
  minLng: number;
  minLat: number;
  maxLng: number;
  maxLat: number;
};

type PropertyMapProps = {
  points: CanonicalPropertyPreview[];
  selectedId: string | null;
  onSelect: (propertyId: string) => void;
  onViewportBounds?: (bbox: BBox) => void;
};

type DisplayPoint =
  | { type: "property"; item: CanonicalPropertyPreview }
  | { type: "cluster"; key: string; lat: number; lng: number; count: number };

const DEFAULT_VIEW = {
  longitude: -73.94,
  latitude: 40.72,
  zoom: 10.8
};

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

export default function PropertyMap({
  points,
  selectedId,
  onSelect,
  onViewportBounds
}: PropertyMapProps) {
  const mapRef = useRef<MapRef | null>(null);
  const lastFocusedIdRef = useRef<string | null>(null);
  const [zoom, setZoom] = useState(DEFAULT_VIEW.zoom);

  const initialView = useMemo(() => DEFAULT_VIEW, []);

  function emitBounds() {
    if (!onViewportBounds) return;
    const bounds = mapRef.current?.getBounds();
    if (!bounds) return;
    setZoom(mapRef.current?.getZoom() ?? DEFAULT_VIEW.zoom);
    onViewportBounds({
      minLng: bounds.getWest(),
      minLat: bounds.getSouth(),
      maxLng: bounds.getEast(),
      maxLat: bounds.getNorth()
    });
  }

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // Ensure canvas dimensions match container after layout changes.
    const resize = () => map.resize();
    resize();
    const timer = window.setTimeout(resize, 120);
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    if (lastFocusedIdRef.current === selectedId) return;
    const selected = points.find((point) => point.property_id === selectedId);
    if (!selected) return;
    const map = mapRef.current;
    if (!map) return;

    lastFocusedIdRef.current = selectedId;
    map.flyTo({
      center: [selected.lng, selected.lat],
      zoom: Math.max(map.getZoom(), 12),
      duration: 450
    });
  }, [selectedId, points]);

  const displayPoints = useMemo<DisplayPoint[]>(() => {
    if (zoom >= 11.5) {
      return points.map((item) => ({ type: "property", item }));
    }

    const cellSize = zoom < 9 ? 0.06 : 0.03;
    const buckets = new globalThis.Map<
      string,
      { count: number; latSum: number; lngSum: number; items: CanonicalPropertyPreview[] }
    >();

    for (const item of points) {
      const cellLng = Math.floor(item.lng / cellSize);
      const cellLat = Math.floor(item.lat / cellSize);
      const key = `${cellLng}:${cellLat}`;
      const bucket = buckets.get(key);
      if (!bucket) {
        buckets.set(key, {
          count: 1,
          latSum: item.lat,
          lngSum: item.lng,
          items: [item]
        });
        continue;
      }
      bucket.count += 1;
      bucket.latSum += item.lat;
      bucket.lngSum += item.lng;
      bucket.items.push(item);
    }

    const clustered: DisplayPoint[] = [];
    for (const [key, bucket] of buckets) {
      if (bucket.count === 1) {
        clustered.push({ type: "property", item: bucket.items[0] });
        continue;
      }

      clustered.push({
        type: "cluster",
        key,
        count: bucket.count,
        lat: bucket.latSum / bucket.count,
        lng: bucket.lngSum / bucket.count
      });
    }

    return clustered;
  }, [points, zoom]);

  return (
    <div className="property-map-wrap">
      <Map
        ref={mapRef}
        initialViewState={initialView}
        mapLib={maplibregl}
        mapStyle={MAP_STYLE}
        onLoad={emitBounds}
        onMoveEnd={emitBounds}
      >
        {displayPoints.map((point) => {
          if (point.type === "property") {
            return (
              <Marker key={point.item.property_id} longitude={point.item.lng} latitude={point.item.lat} anchor="bottom">
                <button
                  type="button"
                  className={`map-marker ${point.item.property_id === selectedId ? "selected" : ""}`}
                  onClick={() => onSelect(point.item.property_id)}
                  aria-label={`Select ${point.item.address}`}
                />
              </Marker>
            );
          }

          return (
            <Marker key={point.key} longitude={point.lng} latitude={point.lat} anchor="center">
              <button
                type="button"
                className="map-cluster-marker"
                onClick={() => {
                  const map = mapRef.current;
                  if (!map) return;
                  map.flyTo({
                    center: [point.lng, point.lat],
                    zoom: Math.min(map.getZoom() + 1.8, 13),
                    duration: 320
                  });
                }}
                aria-label={`Zoom into ${point.count} properties`}
              >
                {point.count}
              </button>
            </Marker>
          );
        })}
      </Map>
    </div>
  );
}
