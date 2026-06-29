import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import matplotlib
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unitraj.datasets import build_dataset
from unitraj.models import build_model
from unitraj.utils import visualization
from unitraj.utils.utils import set_seed
from sind_converter.data.discovery import resolve_map_path
from sind_converter.maps.osm import parse_osm_map


ALL_SIND_CITIES = ("Xi_an", "Changchun", "Chongqing", "Tianjin")


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected checkpoint keys: {len(unexpected)}")


def _city_from_scenario_id(scenario_id):
    scenario_id = str(scenario_id).lower()
    if scenario_id.startswith("sind_xi_an") or "xi_an" in scenario_id or "xian" in scenario_id:
        return "Xi_an"
    if "changchun" in scenario_id:
        return "Changchun"
    if "chongqing" in scenario_id:
        return "Chongqing"
    if "tianjin" in scenario_id:
        return "Tianjin"
    return "unknown"


def _parse_city_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.replace(",", " ").split() if item]
    return [str(item) for item in value]


def _save_aggregate_visualizations(cfg, output_dir, aggregate_records):
    if not aggregate_records:
        return
    data_root = Path(cfg.get("visualization_data_root", "/scratch/izar/ke/sind_raw"))
    map_fallback_root = Path(cfg.get("visualization_map_fallback_root", str(data_root)))
    max_tracks = int(cfg.get("aggregate_max_tracks", 24))
    min_tracks = int(cfg.get("aggregate_min_tracks", min(3, max_tracks)))
    min_track_distance = float(cfg.get("aggregate_min_track_distance", 8.0))
    min_total_steps = int(cfg.get("aggregate_min_total_steps", 61))
    requested_cities = _parse_city_list(cfg.get("aggregate_cities", None))
    cities = requested_cities or sorted({_city_from_scenario_id(record["scenario_id"]) for record in aggregate_records})
    for city in cities:
        city_records = [record for record in aggregate_records if _city_from_scenario_id(record["scenario_id"]) == city]
        if city == "unknown" or not city_records:
            print(f"[warn] no aggregate candidates for city={city}")
            continue
        diagnostics = visualization.prediction_record_diagnostics(city_records, min_total_steps=min_total_steps)
        selected_records = visualization.select_prediction_records_for_osm_map(
            city_records,
            max_tracks=max_tracks,
            min_track_distance=min_track_distance,
            min_total_steps=min_total_steps,
        )
        print(
            "[info] aggregate city={city} candidates={candidates} target_track={target_track} "
            "past_21={past_21} gt_60={gt_60} pred_60={pred_60} "
            "drawable={drawable} selected={selected}/{min_tracks}-{max_tracks}".format(
                city=city,
                selected=len(selected_records),
                min_tracks=min_tracks,
                max_tracks=max_tracks,
                **diagnostics,
            )
        )
        if len(selected_records) < min_tracks:
            print(f"[warn] insufficient 81-step target aggregate tracks for city={city}")
            continue
        map_path = resolve_map_path(city, data_root, map_fallback_root)
        map_features, _ = parse_osm_map(map_path)
        output_path = output_dir / f"aggregate_{city}.png"
        plot = visualization.visualize_prediction_records_on_osm_map(
            map_features,
            selected_records,
            max_tracks=max_tracks,
            min_track_distance=min_track_distance,
            min_total_steps=min_total_steps,
            title=f"{cfg.exp_name} | {city}",
        )
        plot.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.05)
        plot.close()
        plt.close("all")
        print(f"[done] saved aggregate visualization to {output_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def visualize(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)

    if not cfg.get("ckpt_path"):
        raise ValueError("Set ckpt_path=/path/to/model.ckpt")

    output_dir = Path(cfg.get("visualization_output_dir", "prediction_visualizations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    num_images = int(cfg.get("num_prediction_visualizations", 32))
    aggregate_only = bool(cfg.get("aggregate_only", False))

    device_name = cfg.get("visualization_device", None)
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() and not cfg.debug else "cpu"
    device = torch.device(device_name)

    dataset = build_dataset(cfg, val=True)
    batch_size = int(cfg.get("visualization_batch_size", min(cfg.method.get("eval_batch_size", 16), 16)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    model = build_model(cfg)
    _load_checkpoint(model, cfg.ckpt_path)
    model.to(device)
    model.eval()

    saved = 0
    processed = 0
    aggregate_records = []
    aggregate_visualization = bool(cfg.get("aggregate_visualization", False))
    target_cities = _parse_city_list(cfg.get("aggregate_cities", None)) or list(ALL_SIND_CITIES)
    aggregate_max_tracks = int(cfg.get("aggregate_max_tracks", 24))
    aggregate_min_tracks = int(cfg.get("aggregate_min_tracks", min(3, aggregate_max_tracks)))
    aggregate_min_total_steps = int(cfg.get("aggregate_min_total_steps", 61))

    def enough_aggregate_records():
        if not aggregate_visualization:
            return saved >= num_images
        for city in target_cities:
            city_records = [
                record
                for record in aggregate_records
                if _city_from_scenario_id(record["scenario_id"]) == city
            ]
            selected_records = visualization.select_prediction_records_for_osm_map(
                city_records,
                max_tracks=aggregate_max_tracks,
                min_track_distance=float(cfg.get("aggregate_min_track_distance", 8.0)),
                min_total_steps=aggregate_min_total_steps,
            )
            if len(selected_records) < aggregate_max_tracks:
                return False
        return True

    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            if processed >= num_images or enough_aggregate_records():
                break
            batch = _move_to_device(batch, device)
            prediction, _ = model(batch)
            batch_size_actual = int(batch["batch_size"])

            for draw_index in range(batch_size_actual):
                if processed >= num_images or enough_aggregate_records():
                    break

                scenario_id = str(batch["input_dict"]["scenario_id"][draw_index])

                if aggregate_visualization:
                    city = _city_from_scenario_id(scenario_id)
                    if city in target_cities:
                        aggregate_records.append(
                            visualization.extract_prediction_visualization_record(batch, prediction, draw_index=draw_index)
                        )

                if not aggregate_only:
                    object_id = str(batch["input_dict"]["center_objects_id"][draw_index])
                    safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in scenario_id)
                    output_path = output_dir / f"{saved:04d}_{safe_id}_{object_id}.png"
                    plot = visualization.visualize_prediction(batch, prediction, draw_index=draw_index)
                    plot.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
                    plot.close()
                    plt.close("all")
                    saved += 1

                processed += 1

    print(f"[done] processed {processed} samples; saved {saved} single-sample figures to {output_dir}")
    if aggregate_visualization:
        _save_aggregate_visualizations(cfg, output_dir, aggregate_records)


if __name__ == "__main__":
    visualize()
