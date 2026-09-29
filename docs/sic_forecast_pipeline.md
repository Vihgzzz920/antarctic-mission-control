# SIC forecast artifacts, and why the demo route does not use them yet

This describes the pipeline added for the sea-ice forecast artifact stage, and
states plainly the temporal gap between what the model produces and what the
router can currently consume. The gap is intentional and unresolved; nothing in
this pipeline hides it, and no field is relabelled to close it.

## What now exists

```
data/raw/sic_YYYYMMDD.tif            observed SIC, already on the routing grid
data/processed/currents/…            daily currents, same grid
        │
        ├── src/models/build_forecast_dataset.py   features(t) -> sic(t+1)
        ├── src/models/train_forecast_model.py     HistGradientBoostingRegressor
        │
        ▼
data/processed/forecast_model_hgb.joblib          (already existed)
        │
        ├── src/models/forecast_sic_raster.py      ← NEW: model -> field
        ▼
data/processed/sic_forecast/
    sic_forecast_hgb_origin<YYYYMMDD>_plus24h_valid<YYYYMMDD>_3976.tif
    …_3976.json                                    provenance sidecar
    …_hindcast_metrics.json                        HGB vs persistence vs observed
        │
        ├── src/api/environment_forecast.py        ← NEW: the routing door
        ▼
    resolve_environment_forecast(origin_time, valid_time, policy)
```

Generate one:

```
python -m src.models.forecast_sic_raster --origin 2025-12-15 --validate
```

Every raster is written on the routing grid (EPSG:3976, 1328 × 1264, 6250 m,
the same transform `RoutingGrid` loads), float32, nodata NaN, values clipped to
the training script's own [0, 100] bounds, with the forecast origin, valid
time, lead, model artifact hash, feature names, training period, input files
and the prediction and nodata policies written into the GeoTIFF's own tags.

## The temporal gap, stated

| | |
|---|---|
| HGB SIC model lead | **24 h**, the only lead it was fitted for |
| SIC / current archive cadence | **daily**, one field per day |
| Router time buckets | **6 h** (`bucket_hours` in `src/api/config.py`) |
| Exposure fields the router prices with | **6 h and 12 h** iceberg rasters |
| Demonstration route duration | **under 10 h**, inside a 12 h horizon |

So a +24 h sea-ice field has no routing bucket to occupy. The demonstration
route finishes twelve hours before the forecast becomes valid.

**Therefore the default demo does not consume the HGB SIC forecast, and still
prices every bucket from the departure-day analysis.** That is recorded per
bucket in each route's `temporal_provenance`
(`environment_policies: ["persistence_from_departure_analysis"]`,
`environment_is_time_varying: false`) rather than left implicit.

This is deliberate. Making the demo consume the 24 h field would require one of:
slowing the vessel, lengthening the route, moving the endpoints, widening the
buckets, or relabelling the 24 h prediction as a 6 h or 12 h one. The last is a
fabrication and the others distort the demonstration to flatter the model.
`resolve_environment_forecast` refuses every lead except +24 h precisely so that
this cannot happen by accident.

One correction to `TEAM_HANDOFF.md` §1, which says "the routing system then uses
the forecast at the vessel's actual arrival time": that is true of the **iceberg
exposure** term, which really is a different raster per bucket (6 h and 12 h).
It is not true of the **sea-ice environment**, which is one analysis for the
whole route.

## What the numbers say

Hindcasts, scored only on cells where the prediction, persistence and the
observation are all valid, using `persistence.score()` and the training
script's own band definitions:

| origin | window | cells | HGB MAE | persistence MAE | HGB RMSE | persistence RMSE |
|---|---|---|---|---|---|---|
| 2025-01-08 | inside the **training** period | 132,930 | 14.988 | 15.436 | 22.113 | 23.661 |
| 2025-12-15 | the model's **held-out test** window | 223,516 | 11.431 | **10.647** | **17.969** | 18.141 |

On the held-out date persistence has the lower MAE; HGB is better on RMSE and in
the marginal ice zone. That matches the project's own published test figures and
is a second reason not to rush this field into route cost.

## A note on the valid area

The observed GeoTIFFs declare `nodata = 0`, so `preprocess.load_sic` masks every
zero cell. The valid area is therefore the ice-bearing part of the grid
(~144k cells on 2025-01-08), not the whole ocean. The router already sees
exactly this, and the forecast pipeline inherits the same rule rather than
introducing a second masking convention.

## Lead support, from the archive

Both sea-ice sources in this repository are **daily**: the Bremen AMSR2 product
is a day-grid swath composite with no time of day, and the independent
OSI-430-a record declares `time_coverage_resolution: P1D` centred at 12:00 UTC
(3-day sample, on its own 432 × 432 grid). There is **no sub-daily sea-ice
observation anywhere in the data**.

| lead | training target | leakage-safe | model | routable today |
|---|---|---|---|---|
| **0 h** | none needed — it *is* the observation | yes | — | yes, as `observed_analysis` |
| **6 h** | **none exists** | n/a | none | **no** |
| **12 h** | **none exists** | n/a | none | **no** |
| **24 h** | `sic(t) → sic(t+1d)`, **362** usable pairs | yes (origin-only features) | HGB | yes, at its validity |
| **48 h** | `sic(t) → sic(t+2d)`, **361** usable pairs | yes, if built origin-only | **none trained** | **no** |

6 h and 12 h are refused because **no target exists**. 48 h is refused because
**no model exists** — a different refusal, and the registry
(`environment_forecast.LEAD_SUPPORT`) keeps the two apart rather than collapsing
them into "unavailable".

### What 6 h / 12 h would require

A sub-daily SIC product — a swath-level or 6-hourly gridded concentration
(e.g. an hourly/3-hourly passive-microwave or SAR-derived product on, or
reprojectable to, EPSG:3976 6.25 km). Nothing less will do: interpolating
between two daily composites produces a field no model was trained on and no
observation supports, and relabelling the 24 h field is a fabrication. Until
such a product is in `data/`, no 6 h or 12 h model may be trained or claimed.

### What 48 h would require

Only work, not new data:

```
python -m src.models.build_forecast_dataset --lead-days 2 --out data/processed/forecast_dataset_48h.npz
python -m src.models.train_forecast_model   # pointed at that dataset, its own artifact
```

`build_forecast_dataset.py` now takes `--lead-days` (whole days only — the
archive is daily, so a sub-daily lead cannot even be expressed). The default is
1 and the shipped dataset path is unchanged. Each lead then needs its own
artifact, its own provenance, its own leakage checks and its own metrics before
`LEAD_SUPPORT[48]` may claim a model.

## The validity rule: what a forecast field's claim covers

A forecast raster carries one valid time; a routing bucket spans six hours.
Whether the field may price that bucket is a question about **what the model
was trained to predict**, and it is answered by an explicit, named rule. Two
exist, and both remain selectable:

| rule | interval | status |
|---|---|---|
| `exact_valid_time` | the instant `origin + lead` | **available, strict.** The conservative reading, kept for validation: nothing is claimed about any other time, so no six-hour bucket is ever covered. |
| `target_composite_day` | `[midnight(valid_time), midnight(valid_time) + 1 day)` | **the selected rule** for the opt-in `model_forecast_when_supported` routing policy. |

The pairing is not left to the caller. `environment_timeline.POLICY_VALIDITY_RULE`
binds `model_forecast_when_supported` to `target_composite_day`, the timeline
**refuses** a resolver that answers under any other rule, and
`environment_forecast.resolver_for_policy()` hands back a resolver already
carrying it.

### What `target_composite_day` means

The SIC forecast target is the **daily composite associated with the target
calendar day**. The model was fitted on `sic(t) → sic(t + N days)`, where both
sides are daily composite fields; the prediction is therefore a statement about
that whole calendar day, and routing may use it only inside that day's
interval.

It is the same interval convention this project already applies to an *observed*
daily analysis, which prices every bucket inside its own day.

### What it does not mean

- **It is not sub-daily accuracy.** The field does not become more precise by
  being read at 03:00 rather than 21:00. It is one daily quantity, used across
  the day it describes.
- **It does not claim SIC is observed at every six-hour instant.** No
  sub-daily sea-ice observation exists anywhere in this project's data. Nothing
  is interpolated between daily fields and no instantaneous state is implied.
- **It creates no new forecast leads.** 6 h and 12 h remain unsupported MODEL
  LEADS: there is no sub-daily training target, so no model exists for them, and
  widening a *validity interval* cannot manufacture one. A 6 h or 12 h request
  is still answered `unavailable`, never with the +24 h or +48 h field.

### Coverage, measured

A bucket may use a field only if the bucket lies **entirely** inside the
interval. Half-open at the end, so a bucket ending exactly at midnight is
inside the day it ran through. For a midnight departure on 2025-01-08, with the
+24 h field valid on 2025-01-09 and the +48 h field valid on 2025-01-10:

| bucket (h after departure) | clock | `exact_valid_time` | `target_composite_day` |
|---|---|---|---|
| 0–6, 6–12, 12–18 | 08 Jan | unavailable | unavailable |
| **18–24** | 08 Jan 18:00 → 09 Jan 00:00 | unavailable | **unavailable** — straddles midnight |
| 24–30 … 42–48 | 09 Jan | unavailable | **+24 h model_forecast** |
| **48–54 … 66–72** | 10 Jan | unavailable | **+48 h model_forecast** |
| 72–78 | 11 Jan | unavailable | unavailable — past every target day |

The interval decides, never proximity to the valid time: a bucket half an hour
either side of a valid instant is refused, not matched to the nearest field.

## What routing can request today

`resolve_environment(window_from, window_to, origin_time=…, policy=…,
validity_rule=…)` answers with exactly one of four source types, always with
provenance (`source_type`, `forecast_origin`, `valid_time`, `lead_hours`,
`model`, `fallback`, `data_available`, `validity_rule`, `validity_from/to`,
`reason`):

- `observed_analysis` — the analysis at its **own** instant;
- `model_forecast` — a real artifact whose validity covers the **whole** window;
- `explicit_persistence` — only under the policy that names it
  (`model_forecast_else_explicit_persistence`);
- `unavailable` — with the reason, and nothing attached.

`resolve_environment_forecast(origin_time, valid_time, policy)` remains the
narrow artifact lookup (`exact_forecast_valid_time`, `observed_analysis`).

The routing timeline consumes this through the opt-in policy
`model_forecast_when_supported` in `src/api/environment_timeline.py`, which
takes a `resolver` callable so that module stays free of file access. A bucket
nothing covers is **refused** unless the documented persistence fallback was
requested, and every bucket reports its `source_type`, `model` and `valid_time`
in the route's existing `temporal_provenance`.

Nothing in `src/routing` or `src/api/world.py` uses that policy. The shipped
demo still resolves every bucket to the departure-day analysis, and its route is
byte-identical.
