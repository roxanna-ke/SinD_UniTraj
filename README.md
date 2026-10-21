# SinD → UniTraj: traffic-signal-aware trajectory prediction

This project converts the four-city, drone-collected [SinD dataset](SinD/README.md) into ScenarioNet-style scenarios and UniTraj caches. It also associates SinD traffic-light channels with map lanes, encodes observed signal histories, and supplies them to **MTR** and **Wayformer** for vehicle trajectory prediction.

The pipeline is split into reusable stages:

```text
SinD tracks + Lanelet2 maps + traffic-light CSV
    → four-city OSM tag audit
    → canonical ScenarioNet scenarios (shared windows, multiple targets)
    → record-level or city-holdout splits
    → UniTraj HDF5 caches
    → MTR / Wayformer baseline and signal-aware training
```

## Project contributions

- Four-city record discovery handles different directory layouts, file names, missing metadata, and map fallbacks.
- Scenario conversion uses 81-frame windows (21 observed + 60 future frames) with configurable stride. Cars, trucks, and buses with complete histories and futures are prediction targets; other road users remain context.
- A [four-city map audit](osm_audit_four_cities/osm_audit_report.md) determines which OSM features are admitted to the [versioned mapping table](osm_audit_four_cities/osm_training_mapping_table.csv). Missing optional map semantics degrade gracefully.
- [Explicit channel and lane bindings](sind_converter/lights/config/) replace lane-order guesses. Signal states enter ScenarioNet `dynamic_map_states`; UniTraj forms tokens from **observed frames only**. Baseline caches have zero-valued signal map channels and no light tokens; signal-aware caches enable both inputs while retaining the same map feature width.

## Xi'an city-holdout results

The available experiment records train on the other cities and evaluate on held-out Xi'an. Values below are from saved validation summaries. Both Wayformer runs completed full post-training validation; the available MTR records do not establish that a full validation run completed.

| Model | Input | minADE6 | minFDE6 | Miss rate |
| --- | --- | ---: | ---: | ---: |
| MTR | Baseline | 0.3913 | 1.0669 | 0.1525 |
| MTR | + traffic signals | **0.3666** | **0.9033** | **0.1488** |
| Wayformer | Baseline | 0.3117 | 0.7927 | **0.1208** |
| Wayformer | + traffic signals | **0.2993** | **0.7680** | 0.1282 |

Traffic signals improved minADE6 and minFDE6 in both available comparisons. Wayformer's miss rate increased. These are single-city results; they do not establish a gain for every held-out city.

## Repository layout

| Path | Role |
| --- | --- |
| [`sind_converter/`](sind_converter/) | Project-owned data discovery, maps, lights, scenarios, splits, cache, and command-line scripts |
| [`UniTraj/`](UniTraj/) | Local UniTraj source with dataset and MTR/Wayformer signal-token integrations |
| [`scenarionet/`](scenarionet/) | Local ScenarioNet source used for serialization |
| [`SinD/`](SinD/) | SinD documentation and visualization tools; raw data is **not** committed |
| [`osm_audit_four_cities/`](osm_audit_four_cities/) | Four-city audit report, inventories, and map admission table |

## Data and environment

Obtain the full SinD data from the [SinD project](SinD/README.md) and supply paths to its `Dataset/` and `Data/` directories. The local raw data under `SinD/` is excluded from Git. Full conversion and model training are intended for a Linux/CUDA environment; Python 3.10 and the [MTR/Wayformer requirements](UniTraj/requirements-mtr-wayformer.txt) are used by the [environment installer](scripts/install_mtr_wayformer_signal_env.sh). The installer also builds UniTraj's CUDA operations. It is designed for a CUDA development environment with `nvcc`.

All conversion paths are CLI arguments. The SLURM jobs under [`sind_converter/scripts/`](sind_converter/scripts/) provide cluster entry points and accept environment overrides for data, output, and environment roots.

## Run the pipeline

From the repository root, after installing the dependencies:

```bash
export DATA_ROOT=/path/to/SinD/Dataset
export MAP_ROOT=/path/to/SinD/Data
export WORK_ROOT=/path/to/workspace

COMMON=(
  --data-root "$DATA_ROOT"
  --map-fallback-root "$MAP_ROOT"
  --canonical-scenario-root "$WORK_ROOT/canonical"
  --split-root "$WORK_ROOT/splits"
  --cache-root "$WORK_ROOT/cache_signal"
)

# Audit the maps before conversion.
python -m sind_converter.scripts.sind_pipeline audit-maps \
  "${COMMON[@]}" --output-dir "$WORK_ROOT/map_audit"

python -m sind_converter.scripts.sind_pipeline convert-scenarios \
  "${COMMON[@]}" \
  --training-mapping-table "$WORK_ROOT/map_audit/osm_training_mapping_table.csv"

python -m sind_converter.scripts.sind_pipeline make-splits \
  "${COMMON[@]}" --canonical-root "$WORK_ROOT/canonical/sind" \
  --mode city-holdout --heldout-cities Xi_an

python -m sind_converter.scripts.build_sind_unitraj_caches \
  --train-dir "$WORK_ROOT/splits/city_holdout/train/sind" \
  --val-dir "$WORK_ROOT/splits/city_holdout/test/sind" \
  --cache-root "$WORK_ROOT/cache_signal" --methods MTR wayformer \
  --use-traffic-light-tokens --overwrite
```

The final command validates ScenarioNet inputs and performs a small UniTraj dataset dry run before writing `.h5` files and `file_list.pkl`. For baseline caches, omit `--use-traffic-light-tokens` and use a separate cache root such as `$WORK_ROOT/cache_baseline`. Rebuild older caches because map feature widths and signal schemas differ from earlier versions. `make-splits --mode record-level` provides the default reproducible 80/20 record split. `--stride` can be changed from its default of 40 without changing the split or cache code.

For a cluster run, see the [signal-aware conversion/cache job](sind_converter/scripts/build_sind_scenarionet_and_caches_signal.slurm), the [city-holdout cache job](sind_converter/scripts/build_sind_city_cache_signal.slurm), and the MTR/Wayformer baseline and signal training jobs in the same directory.

## Checks and attribution

Focused tests live in [`UniTraj/tests/`](UniTraj/tests/).

SinD, UniTraj, and ScenarioNet are upstream projects included here for integration. See their respective [SinD](SinD/LICENSE), [UniTraj](UniTraj/LICENSE), and [ScenarioNet](scenarionet/LICENSE) licenses and project READMEs for attribution and citation details.
