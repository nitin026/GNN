import { useEffect, useRef } from "react";
import maplibregl, { type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { MapboxOverlay } from "@deck.gl/mapbox";
import type { Layer } from "@deck.gl/core";

// Free OpenStreetMap raster basemap (no API key; attribution shown). Carto's free raster tiles now
// return "API key required", so they are not used. Dark theme = the same tiles with inverted
// brightness. If tiles cannot load (offline), the background plus the India outline / district
// layers the views add still work.
function style(theme: "dark" | "light"): StyleSpecification {
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster", tileSize: 256, maxzoom: 19,
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": theme === "dark" ? "#0b1117" : "#eef2f5" } },
      theme === "dark"
        ? { id: "osm", type: "raster", source: "osm", paint: { "raster-brightness-min": 0.92, "raster-brightness-max": 0.08, "raster-saturation": -0.7, "raster-hue-rotate": 180, "raster-opacity": 0.85 } }
        : { id: "osm", type: "raster", source: "osm", paint: { "raster-saturation": -0.4, "raster-opacity": 0.9 } },
    ],
  };
}

interface Props {
  layers: Layer[];
  theme: "dark" | "light";
  pitch?: number;
  bounds?: [number, number, number, number];
  onClick?: (lat: number, lon: number) => void;
  getTooltip?: (info: any) => any;
  label: string;
}

export default function MapView({ layers, theme, pitch = 0, bounds, onClick, getTooltip, label }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const overlay = useRef<MapboxOverlay | null>(null);
  const click = useRef(onClick);
  click.current = onClick;

  useEffect(() => {
    if (!el.current) return;
    const m = new maplibregl.Map({
      container: el.current, style: style(theme), center: [80, 20], zoom: 3.6, pitch,
      maxPitch: 70, attributionControl: { compact: true }, keyboard: true,
    });
    m.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
    m.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    const o = new MapboxOverlay({ interleaved: false, layers, getTooltip });
    m.addControl(o as unknown as maplibregl.IControl);
    m.on("click", (e) => click.current?.(e.lngLat.lat, e.lngLat.lng));
    map.current = m;
    overlay.current = o;
    return () => { m.remove(); map.current = null; overlay.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);

  useEffect(() => { overlay.current?.setProps({ layers, getTooltip }); }, [layers, getTooltip]);
  useEffect(() => { map.current?.easeTo({ pitch, duration: 600 }); }, [pitch]);
  useEffect(() => {
    if (bounds && map.current) map.current.fitBounds([[bounds[0], bounds[1]], [bounds[2], bounds[3]]], { padding: 20, duration: 0 });
  }, [bounds?.join(",")]);  // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={el} className="map" role="application" aria-label={label} tabIndex={0} />;
}
