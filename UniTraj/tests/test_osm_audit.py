from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


def test_audit_osm_tags_writes_mapping_outputs(tmp_path: Path) -> None:
    module_path = Path(__file__).resolve().parents[2] / "sind_converter" / "maps" / "osm.py"
    spec = importlib.util.spec_from_file_location("sind_converter.maps.osm", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audit_osm_tags = module.audit_osm_tags
    load_training_mapping_table = module.load_training_mapping_table
    build_admission_rules = module.build_admission_rules
    parse_osm_map = module.parse_osm_map

    osm_path = tmp_path / "Tianjin" / "sample.osm"
    osm_path.parent.mkdir(parents=True, exist_ok=True)
    osm_path.write_text(
        """<osm version="0.6" generator="lanelet2">
  <node id="1" lat="0.0" lon="0.0" />
  <node id="2" lat="0.0" lon="0.001" />
  <node id="3" lat="0.001" lon="0.0" />
  <node id="4" lat="0.001" lon="0.001" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="type" v="line_thin" />
  </way>
  <way id="11">
    <nd ref="3" />
    <nd ref="4" />
    <tag k="type" v="stop_line" />
  </way>
  <relation id="20">
    <member type="way" ref="10" role="left" />
    <member type="way" ref="11" role="right" />
    <tag k="type" v="lanelet" />
  </relation>
  <relation id="21">
    <member type="way" ref="11" role="ref_line" />
    <member type="way" ref="10" role="refers" />
    <tag k="type" v="regulatory_element" />
    <tag k="subtype" v="traffic_light" />
  </relation>
</osm>
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "audit"
    table_path = audit_osm_tags([osm_path], output_dir)

    assert table_path == output_dir / "osm_training_mapping_table.csv"
    assert (output_dir / "osm_tag_inventory.csv").exists()
    assert (output_dir / "osm_relation_member_role_inventory.csv").exists()
    assert (output_dir / "osm_recovery_metrics.csv").exists()
    assert (output_dir / "osm_audit_summary.json").exists()
    assert (output_dir / "osm_audit_report.md").exists()

    with table_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    patterns = {row["osm_pattern"]: row for row in rows}
    assert patterns["relation:type=lanelet"]["train_data_admission"] == "include"
    assert patterns["relation:type=lanelet"]["recovery_rate"] == "1.0"
    assert patterns["way:type=stop_line"]["parser_action"] == "map_stop_line_way_to_road_line_polyline"
    assert patterns["way:type=stop_line"]["train_data_admission"] == "include"
    assert patterns["relation:regulatory_element:member:ref_line:way"]["train_data_admission"] == "exclude"

    summary = json.loads((output_dir / "osm_audit_summary.json").read_text(encoding="utf-8"))
    assert summary["observed_predecessor_or_successor_roles"] is False
    assert "relation:type=lanelet" in summary["included_patterns"]
    assert summary["recovery"]["relation:type=lanelet"]["recovered_count"] == 1

    table = load_training_mapping_table(table_path)
    rules = build_admission_rules(
        {
            **row,
            "train_data_admission": "exclude" if row["osm_pattern"] == "way:type=stop_line" else row["train_data_admission"],
        }
        for row in table
    )
    map_features, lane_centers = parse_osm_map(osm_path, admission_rules=rules)

    assert "boundary_10" in map_features
    assert "boundary_11" not in map_features
    assert "lane_20" in map_features
    assert "20" in lane_centers
