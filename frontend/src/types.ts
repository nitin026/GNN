export type Hazard = "tropical_cyclone" | "heat_dome" | "cold_wave";
export type Category = "low" | "moderate" | "severe";

export interface CaseSummary {
  case: string;
  hazard: Hazard;
  synthetic: boolean;
  badge: string;
  init_time: string;
  start: string;
  end: string;
  source?: string;
}

export interface LegendStop { value: number; color: string }
export interface Legend { cmap: string; vmin: number; vmax: number; units: string; stops: LegendStop[] }

export interface Box4D {
  lat_min: number; lat_max: number; lon_min: number; lon_max: number;
  level_hPa_bottom: number; level_hPa_top: number;
  t_start: string; t_end: string; lead_start_h: number; lead_end_h: number;
  member_count_min: number; n_members: number; hazard: Hazard; source: string;
}

export interface TrackingMetrics {
  iou: number; csi: number; pod: number; far: number; spurious_tracks: number;
  centroid_err_km: number | null; detect_lag_h: number | null;
}

export interface Meta {
  case: string; hazard: Hazard; synthetic: boolean; badge: string; source: string;
  init_time: string; valid_times: string[]; leads_h: number[]; leads_5km_h: number[];
  n_members: number; bounds: [number, number, number, number];
  fields_member: string; downscaler: string; tracker: string;
  legends: Record<string, Legend>; bbox4d: Box4D[];
  metrics: {
    tracking?: Record<string, TrackingMetrics>;
    split?: string;
    amphan_track_error_vs_ibtracs?: Record<string, { n_matched: number; err_mean_km: number; err_median_km: number }>;
  };
}

export interface Alert {
  id: string; case: string; hazard: Hazard; kind: "wind" | "rain" | "heat" | "cold";
  lead_h: number; valid_time: string; category: Category; colour: string;
  probability: number; probability_type: string; probability_cell: number; reason: string;
  pinpoint: { lat: number; lon: number; grid: string };
  impact_polygon: { type: "Polygon"; coordinates: number[][][] };
  impact_radius_km: number; region_bbox: [number, number, number, number];
  region_cells_12km: number; in_india: boolean; n_members: number; synthetic: boolean;
  distance_to_pinpoint_km?: number; within_impact_radius?: boolean;
}

export interface DistrictRow {
  district: string; state: string; gid: string; kind: string; category: Category; colour: string;
  probability: number; reason: string; fraction_of_district: number;
  first_lead_h: number; last_lead_h: number; centroid: { lat: number; lon: number };
}

export interface Feature {
  type: "Feature";
  geometry: { type: string; coordinates: any };
  properties: Record<string, any>;
}
export interface FeatureCollection { type: "FeatureCollection"; features: Feature[] }
