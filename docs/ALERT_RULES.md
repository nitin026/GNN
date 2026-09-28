# Alert rules

`backend/alert_rules.py` implements these rules. `scripts/export_products.py` and the API use them.

## 1. Physical thresholds (IMD)

| kind | variable (from the ensemble) | low | moderate | severe | source of the thresholds |
|---|---|---|---|---|---|
| wind | 10 m wind speed (km/h) | >= 62 (cyclonic storm) | >= 89 (severe cyclonic storm) | >= 118 (very severe cyclonic storm) | IMD cyclone classification |
| rain | 24 h accumulation (mm), four complete 6-h steps | >= 64.5 (heavy) | >= 115.6 (very heavy) | >= 204.5 (extremely heavy) | IMD rainfall categories |
| heat | Tmax departure from normal (C), plus the regional absolute criterion (see below) | >= 4.5 (heat wave) | >= 5.5 * | >= 6.5 (severe heat wave) | IMD heat-wave criteria |
| cold | Tmin departure from normal (C), plus the regional absolute criterion (see below) | <= -4.5 (cold wave) | <= -5.5 * | <= -6.5 (severe cold wave) | IMD cold-wave criteria |

\* IMD has two heat-wave levels and two cold-wave levels. The "moderate" level at 5.5 C is **our** midpoint,
added to fit the three-colour scheme.

**Regional absolute criteria (IMD-style, as in `pipeline/tracker2.py`)**
- Heat: Tmax >= 40 C in the plains, >= 37 C in coastal areas, >= 30 C in the hills, or Tmax >= 45 C anywhere.
- Cold: Tmin <= 10 C in the plains (or <= 4 C), <= 15 C in coastal areas, <= 0 C in the hills.

**How the inputs are derived**
- Tmax/Tmin come from a centred 24 h window of 6-hourly T2m.
- Normals come from the ERA5 1990-2019 climatology, not from IMD station normals.
- The 10 m wind is a 10-minute-style model wind, while IMD's classes use 3-minute sustained wind;
  no conversion factor is applied.

## 2. From exceedance probability to category (project design choice, not IMD)

For each lead and 12 km cell, P(threshold) is the **neighbourhood** exceedance probability: the
fraction of the 20 members whose field exceeds the threshold anywhere within **50 km** of the cell.
This allows for ensemble position spread: a strong vortex displaced by 30 km still counts.
Every alert also carries the plain per-cell probability (`probability_cell`).

| category | rule | IMD colour |
|---|---|---|
| severe | P(severe threshold) >= 0.5 | red `#D7191C` |
| moderate | P(severe threshold) >= 0.2, or P(moderate threshold) >= 0.5 | orange `#FF8C00` |
| low | P(moderate threshold) >= 0.2, or P(low threshold) >= 0.5 | yellow `#FFD400` |

The 0.2 and 0.5 cut-offs and the 50 km radius were set once by hand. They were not fitted to any case.

**Honest note.** The per-cell rule was tried first. On `amphan_replay` (a TEST case), its peak P(wind >= 118 km/h) is 0.45 over all leads, below the 0.5 severe cut-off, so a super cyclone could never get a severe wind alert. That observation led to the neighbourhood rule; with it, the exported peak is 0.65 (`backend/products/amphan_replay/alerts.json`). The decision was made after looking at a test case; both probabilities are therefore shown on every alert.

## 3. From cells to alerts

1. Connected regions of category >= low, at least 3 cells at 12 km, one alert per region, per lead,
   per kind.
2. **Category** = the highest category in the region. **Probability** = the largest probability, within the
   region, of the rule that set the category. **Reason** is that probability written out, e.g.
   `P(wind >= 118 km/h) = 0.65 within 50 km`.
3. **Pinpoint (core coordinate)**: the 5 km cell with the most extreme value inside the region, taken
   from the representative member (the member whose GNN main track is closest to the consensus track;
   `fields_member` in meta.json) downscaled to 5 km by the U-Net. Most extreme means maximum wind, 24 h rain or T2m,
   or minimum T2m for cold.
4. **Impact polygon**: a 5 km-radius circle (32 vertices) around the pinpoint. `region_bbox` gives the
   12 km region's extent.
5. `in_india`: whether the region touches the Natural Earth India mask. Marine alerts over the Bay of
   Bengal or the Arabian Sea are kept and flagged.
6. The real case, `amphan_era5_real`, is a single ERA5 run, so its "probabilities" are 0 or 1. They are
   deterministic exceedances, labelled as such in the UI.
