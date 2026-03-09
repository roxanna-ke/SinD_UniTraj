from pathlib import Path
import shutil

import pytest
from scenarionet.common_utils import read_dataset_summary, read_scenario


REPO_ROOT = Path(__file__).resolve().parents[1]
SIND_ROOT = REPO_ROOT.parent / "SinD"
SIND_RECORD_DIR = SIND_ROOT / "Data" / "Tianjin" / "8_2_1"


@pytest.mark.skipif(not SIND_RECORD_DIR.exists(), reason="SinD sample data is not available")
def test_convert_sind_record_writes_scenarionet_dataset(tmp_path):
    from unitraj.utils.sind_converter import convert_sind_record

    output_dir = tmp_path / "sind_dataset"
    result = convert_sind_record(
        sind_record_dir=SIND_RECORD_DIR,
        output_dir=output_dir,
        city="Tianjin",
        dataset_name="sind",
        dataset_version="v1",
        past_len=21,
        future_len=60,
        stride=20,
        max_scenarios=6,
        train_ratio=0.5,
    )

    assert (result["train"] / "dataset_summary.pkl").exists()
    assert (result["val"] / "dataset_summary.pkl").exists()

    summary, scenario_files, mapping = read_dataset_summary(result["train"])
    assert summary
    assert scenario_files

    scenario = read_scenario(result["train"], mapping, scenario_files[0])
    required_keys = {"id", "version", "length", "tracks", "dynamic_map_states", "map_features", "metadata"}
    assert required_keys.issubset(set(scenario.keys()))

    metadata = scenario["metadata"]
    assert metadata["dataset"] == "sind"
    assert metadata["scenario_id"] == scenario["id"]
    assert metadata["sdc_id"] in scenario["tracks"]
    assert metadata["tracks_to_predict"]
    assert len(metadata["ts"]) == scenario["length"] == 81

    target_id = next(iter(metadata["tracks_to_predict"]))
    assert target_id in scenario["tracks"]

    target_track = scenario["tracks"][target_id]
    assert target_track["type"] == "VEHICLE"
    assert target_track["state"]["position"].shape == (81, 3)
    assert target_track["state"]["velocity"].shape == (81, 2)
    assert target_track["state"]["valid"].shape == (81,)
    assert target_track["state"]["valid"][20] == 1.0

    from omegaconf import OmegaConf
    from unitraj.datasets.autobot_dataset import AutoBotDataset

    cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/config.yaml")
    method_cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/method/autobot.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, {"method": method_cfg})
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg.train_data_path = [str(result["train"])]
    cfg.max_data_num = [None]
    cfg.starting_frame = [0]
    cfg.debug = True

    dataset = AutoBotDataset(cfg, is_validation=False)
    assert len(dataset) > 0


def test_autobot_forward_stays_finite_with_empty_map_sind_batch():
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from unitraj.datasets.autobot_dataset import AutoBotDataset
    from unitraj.models.autobot.autobot import AutoBotEgo
    import torch

    converted_train = REPO_ROOT / "converted_data" / "sind_mvp" / "train" / "sind"
    if not converted_train.exists():
        pytest.skip("Converted SinD dataset is not available yet")

    cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/config.yaml")
    method_cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/method/autobot.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, {"method": method_cfg})
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg.train_data_path = [str(converted_train)]
    cfg.max_data_num = [1]
    cfg.starting_frame = [0]
    cfg.debug = True

    dataset = AutoBotDataset(cfg, is_validation=False)
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)))
    model = AutoBotEgo(config=cfg)

    assert batch["input_dict"]["map_polylines_mask"].sum() == 0

    with torch.no_grad():
        prediction, loss = model(batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction["predicted_probability"]).all()
    assert torch.isfinite(prediction["predicted_trajectory"]).all()


def test_autobot_forward_stays_finite_with_mixed_empty_and_nonempty_map_batch():
    from omegaconf import OmegaConf
    from unitraj.datasets.autobot_dataset import AutoBotDataset
    from unitraj.models.autobot.autobot import AutoBotEgo
    import torch

    converted_train = REPO_ROOT / "converted_data" / "sind_mvp" / "train" / "sind"
    nuscenes_train = REPO_ROOT / "unitraj/data_samples/nuscenes"
    if not converted_train.exists() or not nuscenes_train.exists():
        pytest.skip("Required local sample datasets are not available")

    def build_cfg(dataset_path: Path):
        cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/config.yaml")
        method_cfg = OmegaConf.load(REPO_ROOT / "unitraj/configs/method/autobot.yaml")
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, {"method": method_cfg})
        cfg = OmegaConf.merge(cfg, cfg.method)
        cfg.train_data_path = [str(dataset_path)]
        cfg.max_data_num = [1]
        cfg.starting_frame = [0]
        cfg.debug = True
        return cfg

    sind_cfg = build_cfg(converted_train)
    nuscenes_cfg = build_cfg(nuscenes_train)

    sind_dataset = AutoBotDataset(sind_cfg, is_validation=False)
    nuscenes_dataset = AutoBotDataset(nuscenes_cfg, is_validation=False)
    batch = sind_dataset.collate_fn([sind_dataset[0], nuscenes_dataset[0]])
    model = AutoBotEgo(config=sind_cfg)

    map_mask_sums = batch["input_dict"]["map_polylines_mask"].sum(dim=(1, 2))
    assert map_mask_sums[0] == 0
    assert map_mask_sums[1] > 0

    with torch.no_grad():
        prediction, loss = model(batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction["predicted_probability"]).all()
    assert torch.isfinite(prediction["predicted_trajectory"]).all()


@pytest.mark.skipif(not SIND_RECORD_DIR.exists(), reason="SinD sample data is not available")
def test_convert_sind_record_exports_map_pedestrians_and_traffic_lights(tmp_path):
    from unitraj.utils.sind_converter import convert_sind_record

    output_dir = tmp_path / "sind_enhanced"
    result = convert_sind_record(
        sind_record_dir=SIND_RECORD_DIR,
        output_dir=output_dir,
        city="Tianjin",
        dataset_name="sind",
        dataset_version="v1",
        past_len=21,
        future_len=60,
        stride=40,
        max_scenarios=4,
        train_ratio=0.5,
    )

    summary, scenario_files, mapping = read_dataset_summary(result["train"])
    scenario = read_scenario(result["train"], mapping, scenario_files[0])

    assert scenario["map_features"]
    assert any(feature["type"] == "LANE_SURFACE_STREET" for feature in scenario["map_features"].values())
    assert any(feature["type"] == "CROSSWALK" for feature in scenario["map_features"].values())
    assert scenario["dynamic_map_states"]
    assert len(scenario["dynamic_map_states"]) == 8
    first_signal = next(iter(scenario["dynamic_map_states"].values()))
    assert len(first_signal["state"]["object_state"]) == scenario["length"]
    assert any(track["type"] == "PEDESTRIAN" for track in scenario["tracks"].values())


@pytest.mark.skipif(not (SIND_ROOT / "Data").exists(), reason="SinD dataset root is not available")
def test_convert_sind_dataset_splits_by_record(tmp_path):
    from unitraj.utils.sind_converter import convert_sind_dataset

    data_root = SIND_ROOT / "Data"
    available_records = []
    for record_dir in sorted([p for p in data_root.glob("*/*_*_*") if p.is_dir()]):
        required_files = [
            record_dir / "Veh_smoothed_tracks.csv",
            record_dir / "Veh_tracks_meta.csv",
            record_dir / "Ped_smoothed_tracks.csv",
            record_dir / "Ped_tracks_meta.csv",
            record_dir / f"TrafficLight_{record_dir.name}.csv",
            record_dir / "recording_metas.csv",
        ]
        if all(path.exists() for path in required_files):
            available_records.append(record_dir)
    if len(available_records) < 2:
        pytest.skip("Need at least two fully convertible SinD records for split testing")

    output_dir = tmp_path / "sind_split"
    result = convert_sind_dataset(
        sind_data_root=data_root,
        output_dir=output_dir,
        dataset_name="sind",
        dataset_version="v1",
        max_records=2,
        stride=80,
        max_scenarios_per_record=2,
        train_ratio=0.5,
    )

    train_summary, _, _ = read_dataset_summary(result["train"])
    val_summary, _, _ = read_dataset_summary(result["val"])

    train_records = {meta["source_file"] for meta in train_summary.values()}
    val_records = {meta["source_file"] for meta in val_summary.values()}
    assert train_records
    assert val_records
    assert train_records.isdisjoint(val_records)
