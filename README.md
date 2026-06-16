# SinD_UniTraj

This repository combines four related components:

- `SinD/`: SinD raw data, maps, and visualization utilities
- `scenarionet/`: local ScenarioNet source used for scenario serialization and dataset summaries
- `UniTraj/`: local UniTraj training code and model implementations
- `sind_converter/`: modular pipeline for converting SinD data into ScenarioNet-format datasets and UniTraj caches

## What is implemented

The current integration supports:

- discovering SinD records across the bundled dataset layout
- converting SinD trajectories, maps, pedestrians, and traffic lights into ScenarioNet scenarios
- writing train/validation ScenarioNet datasets for UniTraj
- building UniTraj caches for `MTR`, `autobot`, and `wayformer`
- basic regression tests for the modular converter

## Main entry points

Generate ScenarioNet-style train/val data from SinD:

```bash
python UniTraj/unitraj/utils/sind_converter.py \
  --sind-data-root /path/to/SinD/Dataset \
  --map-fallback-root /path/to/SinD/Data \
  --output-dir /path/to/scenarionet_output
```

Build UniTraj caches from existing ScenarioNet train/val splits:

```bash
python sind_converter/scripts/build_sind_unitraj_caches.py \
  --train-dir /path/to/scenarionet_output/train/sind \
  --val-dir /path/to/scenarionet_output/val/sind \
  --cache-root /path/to/cache_root
```

Run the full cluster workflow that writes both ScenarioNet output and caches to scratch:

```bash
sbatch sind_converter/scripts/build_sind_scenarionet_and_caches.slurm
```

## Repository notes

- `Summary/` contains local research and integration notes.
- `UniTraj/scenarionet_output/` contains local generated sample output in this workspace.
- The root `.gitignore` excludes local data products, caches, and training artifacts.

## Tests

Relevant tests are under `UniTraj/tests/`, including:

- `test_sind_converter.py`
- `test_modular_sind_converter.py`
