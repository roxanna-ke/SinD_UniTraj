from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MIN_RECOVERY_RATE_FOR_TRAINING = 0.95


WAY_TYPE_TO_SCENARIONET = {
    "curbstone": "BOUNDARY_LINE",
    "line_thin": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "line_thick": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "stop_line": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "virtual": "ROAD_LINE_BROKEN_SINGLE_WHITE",
}

TAG_RULES = {
    ("relation", "type", "lanelet"): {
        "scenarionet_feature_type": "LANE_SURFACE_STREET",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "skip malformed relation and continue",
        "parser_action": "construct_lane_from_lanelet_relation",
        "requires_explicit_osm_structure": "relation type=lanelet with left/right way members",
        "notes": "Lane centerline/polygon extraction comes from left/right member ways.",
    },
    ("relation", "subtype", "crosswalk"): {
        "scenarionet_feature_type": "CROSSWALK",
        "mapping_status": "checked",
        "admission_decision": "include_if_recovered",
        "degrade_strategy": "omit crosswalk polygon and continue",
        "parser_action": "construct_crosswalk_polygon_from_lanelet_relation",
        "requires_explicit_osm_structure": "relation type=lanelet subtype=crosswalk with left/right way members",
        "notes": "Crosswalk is only emitted when the lanelet relation has usable left/right boundaries.",
    },
    ("relation", "type", "regulatory_element"): {
        "scenarionet_feature_type": "",
        "mapping_status": "audited_only",
        "admission_decision": "exclude",
        "degrade_strategy": "ignore unsupported regulatory elements and continue",
        "parser_action": "inventory_regulatory_element_for_future_linking",
        "requires_explicit_osm_structure": "relation type=regulatory_element",
        "notes": "Used for traffic-light / stop-line association audit; not yet converted into static map features.",
    },
    ("relation", "subtype", "traffic_light"): {
        "scenarionet_feature_type": "",
        "mapping_status": "audited_only",
        "admission_decision": "exclude",
        "degrade_strategy": "keep dynamic light CSV pipeline if map relation linking is unavailable",
        "parser_action": "inventory_traffic_light_regulatory_relations",
        "requires_explicit_osm_structure": "relation type=regulatory_element subtype=traffic_light",
        "notes": "Traffic-light regulatory relations are present in some cities and should be preserved for later lane-light linking.",
    },
    ("way", "type", "curbstone"): {
        "scenarionet_feature_type": "BOUNDARY_LINE",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "omit feature and continue",
        "parser_action": "map_way_type_to_boundary_polyline",
        "requires_explicit_osm_structure": "way type=curbstone with two or more nodes",
        "notes": "Acts as robust road-edge / curb geometry.",
    },
    ("way", "type", "line_thin"): {
        "scenarionet_feature_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "fall back to generic road line and continue",
        "parser_action": "map_way_type_to_road_line_polyline",
        "requires_explicit_osm_structure": "way type=line_thin with two or more nodes",
        "notes": "Currently collapsed to a single white solid road-line class.",
    },
    ("way", "type", "line_thick"): {
        "scenarionet_feature_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "fall back to generic road line and continue",
        "parser_action": "map_way_type_to_road_line_polyline",
        "requires_explicit_osm_structure": "way type=line_thick with two or more nodes",
        "notes": "Currently collapsed to a single white solid road-line class.",
    },
    ("way", "type", "stop_line"): {
        "scenarionet_feature_type": "ROAD_LINE_SOLID_SINGLE_WHITE",
        "mapping_status": "stable_geometry",
        "admission_decision": "include",
        "degrade_strategy": "preserve stop-line geometry as a generic road line and continue",
        "parser_action": "map_stop_line_way_to_road_line_polyline",
        "requires_explicit_osm_structure": "way type=stop_line with two or more nodes",
        "notes": "Stop-line geometry exists, but lane association and dedicated stop-line semantics are still pending.",
    },
    ("way", "type", "virtual"): {
        "scenarionet_feature_type": "ROAD_LINE_BROKEN_SINGLE_WHITE",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "omit feature and continue",
        "parser_action": "map_virtual_way_to_broken_road_line_polyline",
        "requires_explicit_osm_structure": "way type=virtual with two or more nodes",
        "notes": "Used as lane-divider style geometry.",
    },
    ("way", "type", "traffic_light"): {
        "scenarionet_feature_type": "",
        "mapping_status": "audited_only",
        "admission_decision": "exclude",
        "degrade_strategy": "ignore unsupported traffic-light ways and continue",
        "parser_action": "inventory_traffic_light_way_for_future_linking",
        "requires_explicit_osm_structure": "way type=traffic_light",
        "notes": "Traffic-light physical geometry is audited but not yet converted into map_features.",
    },
}

ROLE_RULES = {
    ("lanelet", "left", "way"): {
        "scenarionet_feature_type": "LANE_SURFACE_STREET",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "skip malformed lanelet relation and continue",
        "parser_action": "use_way_as_left_lane_boundary",
        "requires_explicit_osm_structure": "lanelet relation member role=left type=way",
        "notes": "Stable lanelet boundary role required for lane polygon / centerline reconstruction.",
    },
    ("lanelet", "right", "way"): {
        "scenarionet_feature_type": "LANE_SURFACE_STREET",
        "mapping_status": "stable",
        "admission_decision": "include",
        "degrade_strategy": "skip malformed lanelet relation and continue",
        "parser_action": "use_way_as_right_lane_boundary",
        "requires_explicit_osm_structure": "lanelet relation member role=right type=way",
        "notes": "Stable lanelet boundary role required for lane polygon / centerline reconstruction.",
    },
    ("regulatory_element", "ref_line", "way"): {
        "scenarionet_feature_type": "",
        "mapping_status": "audited_only",
        "admission_decision": "exclude",
        "degrade_strategy": "ignore unsupported regulatory links and continue",
        "parser_action": "inventory_stop_or_light_reference_line",
        "requires_explicit_osm_structure": "regulatory_element relation member role=ref_line type=way",
        "notes": "Useful for future stop-line / signal association work.",
    },
    ("regulatory_element", "refers", "way"): {
        "scenarionet_feature_type": "",
        "mapping_status": "audited_only",
        "admission_decision": "exclude",
        "degrade_strategy": "ignore unsupported regulatory links and continue",
        "parser_action": "inventory_regulatory_target_way",
        "requires_explicit_osm_structure": "regulatory_element relation member role=refers type=way",
        "notes": "Useful for future traffic-light / controlled-lane association work.",
    },
}


@dataclass
class RecoveryCounter:
    total: int = 0
    recovered: int = 0

    @property
    def rate(self) -> float | None:
        if self.total == 0:
            return None
        return self.recovered / self.total


@dataclass
class OSMAuditData:
    tag_inventory: dict[tuple[str, str, str], dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    member_role_inventory: dict[tuple[str, str, str], dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    city_stats: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    recovery: dict[str, dict[str, RecoveryCounter]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(RecoveryCounter)))


@dataclass(frozen=True)
class TrainingMapAdmissionRules:
    included_patterns: frozenset[str]

    def allows(self, pattern: str) -> bool:
        return pattern in self.included_patterns

    def require_all(self, patterns: Iterable[str]) -> bool:
        return all(self.allows(pattern) for pattern in patterns)


def load_training_mapping_table(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_admission_rules(table: Iterable[dict[str, str]]) -> TrainingMapAdmissionRules:
    included_patterns = frozenset(
        row["osm_pattern"]
        for row in table
        if row.get("train_data_admission") == "include" and row.get("osm_pattern")
    )
    return TrainingMapAdmissionRules(included_patterns=included_patterns)


def default_training_admission_rules() -> TrainingMapAdmissionRules:
    table: list[dict[str, str]] = []
    for element, key, value in TAG_RULES:
        rule = _tag_rule(element, key, value)
        if rule["admission_decision"] in {"include", "include_if_recovered"} and rule["scenarionet_feature_type"]:
            table.append({"osm_pattern": _metric_key_for_tag(element, key, value), "train_data_admission": "include"})
    for relation_type, member_role, member_type in ROLE_RULES:
        rule = _role_rule(relation_type, member_role, member_type)
        if rule["admission_decision"] in {"include", "include_if_recovered"} and rule["scenarionet_feature_type"]:
            table.append(
                {
                    "osm_pattern": _metric_key_for_role(relation_type, member_role, member_type),
                    "train_data_admission": "include",
                }
            )
    return build_admission_rules(table)


def project_lon_lat_to_xy(lon: float, lat: float) -> np.ndarray:
    meters_per_degree = 111_320.0
    return np.array([lon * meters_per_degree, lat * meters_per_degree, 0.0], dtype=np.float32)


def _city_name_for_map(map_path: Path) -> str:
    return map_path.parent.name


def _format_city_counts(city_counts: dict[str, int]) -> str:
    return ";".join(f"{city}:{count}" for city, count in sorted(city_counts.items()))


def _tag_rule(element: str, key: str, value: str) -> dict[str, str]:
    return {
        "scenarionet_feature_type": "",
        "mapping_status": "unmapped",
        "admission_decision": "exclude",
        "degrade_strategy": "omit feature and continue",
        "parser_action": "ignore_unmapped_tag",
        "requires_explicit_osm_structure": "",
        "notes": "Observed during audit but not admitted into the final training map feature set.",
        **TAG_RULES.get((element, key, value), {}),
    }


def _role_rule(relation_type: str, member_role: str, member_type: str) -> dict[str, str]:
    return {
        "scenarionet_feature_type": "",
        "mapping_status": "unmapped",
        "admission_decision": "exclude",
        "degrade_strategy": "ignore unsupported relation role and continue",
        "parser_action": "ignore_unmapped_relation_role",
        "requires_explicit_osm_structure": "",
        "notes": "Observed during audit but not part of the current parser contract.",
        **ROLE_RULES.get((relation_type, member_role, member_type), {}),
    }


def _polyline_has_geometry(polyline: np.ndarray | None) -> bool:
    return polyline is not None and len(polyline) >= 2


def _polygon_has_geometry(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    return _polyline_has_geometry(left) and _polyline_has_geometry(right)


def _metric_key_for_tag(element: str, key: str, value: str) -> str:
    return f"{element}:{key}={value}"


def _metric_key_for_role(relation_type: str, member_role: str, member_type: str) -> str:
    return f"relation:{relation_type}:member:{member_role}:{member_type}"


def _increment_metric(audit: OSMAuditData, city: str, metric_key: str, recovered: bool) -> None:
    counter = audit.recovery[metric_key][city]
    counter.total += 1
    if recovered:
        counter.recovered += 1


def _collect_osm_audit_data(map_paths: list[Path]) -> OSMAuditData:
    audit = OSMAuditData()

    for map_path in map_paths:
        city = _city_name_for_map(map_path)
        root = ET.parse(map_path).getroot()
        audit.city_stats[city]["map_files"] += 1
        node_lookup = {
            node.attrib["id"]: project_lon_lat_to_xy(float(node.attrib["lon"]), float(node.attrib["lat"]))
            for node in root.findall("node")
            if "id" in node.attrib and "lon" in node.attrib and "lat" in node.attrib
        }
        way_lookup = {way.attrib["id"]: way for way in root.findall("way") if "id" in way.attrib}
        way_polylines = {way_id: _polyline_from_way(node_lookup, way) for way_id, way in way_lookup.items()}

        for elem_name in ["node", "way", "relation"]:
            for elem in root.findall(elem_name):
                audit.city_stats[city][f"{elem_name}_count"] += 1
                tags = {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in elem.findall("tag")}
                for key, value in tags.items():
                    audit.tag_inventory[(elem_name, key, value)][city] += 1

                if elem_name == "way":
                    way_type = tags.get("type", "")
                    metric_key = _metric_key_for_tag("way", "type", way_type)
                    if ("way", "type", way_type) in TAG_RULES:
                        _increment_metric(audit, city, metric_key, _polyline_has_geometry(way_polylines.get(elem.attrib.get("id", ""))))

                if elem_name != "relation":
                    continue

                relation_type = tags.get("type", "")
                relation_subtype = tags.get("subtype", "")
                members_by_role: dict[str, list[str]] = defaultdict(list)
                for member in elem.findall("member"):
                    role = member.attrib.get("role", "")
                    member_type = member.attrib.get("type", "")
                    audit.member_role_inventory[(relation_type, role, member_type)][city] += 1
                    members_by_role[role].append(member.attrib.get("ref", ""))

                left = way_polylines.get(members_by_role.get("left", [""])[0])
                right = way_polylines.get(members_by_role.get("right", [""])[0])
                if relation_type == "lanelet" and relation_subtype != "crosswalk":
                    recovered = _polygon_has_geometry(left, right)
                    _increment_metric(audit, city, "relation:type=lanelet", recovered)
                    _increment_metric(audit, city, "relation:lanelet:member:left:way", _polyline_has_geometry(left))
                    _increment_metric(audit, city, "relation:lanelet:member:right:way", _polyline_has_geometry(right))
                if relation_type == "lanelet" and relation_subtype == "crosswalk":
                    recovered = _polygon_has_geometry(left, right)
                    _increment_metric(audit, city, "relation:subtype=crosswalk", recovered)
                if relation_type == "regulatory_element":
                    ref_lines = members_by_role.get("ref_line", [])
                    refers = members_by_role.get("refers", [])
                    _increment_metric(audit, city, "relation:type=regulatory_element", bool(ref_lines or refers))
                    if relation_subtype == "traffic_light":
                        _increment_metric(audit, city, "relation:subtype=traffic_light", bool(ref_lines and refers))
                    for ref in ref_lines:
                        _increment_metric(audit, city, "relation:regulatory_element:member:ref_line:way", _polyline_has_geometry(way_polylines.get(ref)))
                    for ref in refers:
                        _increment_metric(audit, city, "relation:regulatory_element:member:refers:way", _polyline_has_geometry(way_polylines.get(ref)))
    return audit


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_report(
    output_dir: Path,
    map_paths: list[Path],
    city_stats: dict[str, dict[str, int]],
    mapping_rows: list[dict[str, Any]],
    member_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> Path:
    included = [row for row in mapping_rows if row["train_data_admission"] == "include"]
    excluded = [row for row in mapping_rows if row["train_data_admission"] == "exclude" and row["mapping_status"] != "unmapped"]
    observed_roles = {(row["relation_type"], row["member_role"], row["member_type"]) for row in member_rows}
    topology_note = (
        "No explicit predecessor/successor relation-member roles were observed in the audited OSM files. "
        "Lane topology beyond left/right boundaries will require dedicated lanelet graph inference or a lanelet2 API pass."
    )
    report_path = output_dir / "osm_audit_report.md"
    lines = [
        "# OSM Audit Report",
        "",
        "## Audited Maps",
        "",
    ]
    for map_path in map_paths:
        lines.append(f"- {_city_name_for_map(map_path)}: `{map_path}`")
    lines.extend(["", "## City Stats", ""])
    for city, stats in sorted(city_stats.items()):
        lines.append(
            f"- {city}: nodes={stats.get('node_count', 0)}, ways={stats.get('way_count', 0)}, relations={stats.get('relation_count', 0)}"
        )
    lines.extend(["", "## Final Training Map Feature Inputs", ""])
    for row in included:
        pattern = row.get("osm_pattern") or f"{row['relation_type']}:{row['member_role']}:{row['member_type']}"
        lines.append(f"- `{pattern}` -> `{row['scenarionet_feature_type']}` ({row['parser_action']})")
    lines.extend(["", "## Audited But Excluded", ""])
    for row in excluded:
        pattern = row.get("osm_pattern") or f"{row['relation_type']}:{row['member_role']}:{row['member_type']}"
        lines.append(f"- `{pattern}` ({row['parser_action']}): {row['admission_reason']}")
    lines.extend(["", "## Recovery Metrics", ""])
    for row in recovery_rows:
        lines.append(
            f"- `{row['osm_pattern']}`: total={row['total_count']}, recovered={row['recovered_count']}, "
            f"rate={row['recovery_rate']}, per_city={row['city_recovery_rates']}"
        )
    lines.extend(["", "## Topology Note", "", f"- {topology_note}"])
    if ("lanelet", "left", "way") in observed_roles and ("lanelet", "right", "way") in observed_roles:
        lines.append("- Stable `left/right` lanelet boundary roles were observed across audited maps.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _recovery_summary_for_pattern(
    pattern: str,
    recovery: dict[str, dict[str, RecoveryCounter]],
    audited_cities: list[str],
) -> dict[str, Any]:
    city_counters = recovery.get(pattern, {})
    total = sum(counter.total for counter in city_counters.values())
    recovered = sum(counter.recovered for counter in city_counters.values())
    rates = {
        city: (round(counter.rate, 6) if counter.rate is not None else None)
        for city, counter in sorted(city_counters.items())
    }
    observed_cities = sorted(city_counters)
    min_observed_rate = min((counter.rate for counter in city_counters.values() if counter.rate is not None), default=None)
    return {
        "recovery_total": total,
        "recovery_recovered": recovered,
        "recovery_rate": round(recovered / total, 6) if total else None,
        "min_city_recovery_rate": round(min_observed_rate, 6) if min_observed_rate is not None else None,
        "observed_cities": ";".join(observed_cities),
        "audited_city_coverage": round(len(observed_cities) / len(audited_cities), 6) if audited_cities else 0.0,
        "city_recovery_rates": ";".join(f"{city}:{rates[city]}" for city in observed_cities),
    }


def _train_admission(rule: dict[str, str], recovery_summary: dict[str, Any]) -> tuple[str, str]:
    if rule["admission_decision"] == "exclude":
        return "exclude", "No final ScenarioNet map feature mapping is defined for this pattern."
    if not rule["scenarionet_feature_type"]:
        return "exclude", "No ScenarioNet feature type is defined."
    min_rate = recovery_summary["min_city_recovery_rate"]
    if min_rate is None:
        return "exclude", "No recoverable instances were observed in the audited maps."
    if min_rate < MIN_RECOVERY_RATE_FOR_TRAINING:
        return "exclude", f"Minimum per-city recovery rate {min_rate} is below {MIN_RECOVERY_RATE_FOR_TRAINING}."
    return "include", "Explicit OSM structure is recoverable above the training admission threshold."


def _recovery_metric_rows(audit: OSMAuditData, audited_cities: list[str]) -> list[dict[str, Any]]:
    rows = []
    for pattern, city_counters in sorted(audit.recovery.items()):
        summary = _recovery_summary_for_pattern(pattern, audit.recovery, audited_cities)
        rows.append(
            {
                "osm_pattern": pattern,
                "cities": summary["observed_cities"],
                "audited_city_coverage": summary["audited_city_coverage"],
                "total_count": summary["recovery_total"],
                "recovered_count": summary["recovery_recovered"],
                "recovery_rate": summary["recovery_rate"],
                "min_city_recovery_rate": summary["min_city_recovery_rate"],
                "city_recovery_rates": summary["city_recovery_rates"],
                "city_recovery_counts": ";".join(
                    f"{city}:{counter.recovered}/{counter.total}" for city, counter in sorted(city_counters.items())
                ),
            }
        )
    return rows


def audit_osm_tags(map_paths: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = _collect_osm_audit_data(map_paths)
    audited_cities = sorted(audit.city_stats)

    tag_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for (element, key, value), city_counts in sorted(audit.tag_inventory.items()):
        rule = _tag_rule(element, key, value)
        pattern = f"{element}:{key}={value}"
        recovery_summary = _recovery_summary_for_pattern(pattern, audit.recovery, audited_cities)
        train_data_admission, admission_reason = _train_admission(rule, recovery_summary)
        row = {
            "source_kind": "tag",
            "element": element,
            "tag_key": key,
            "tag_value": value,
            "osm_pattern": pattern,
            "cities": ";".join(sorted(city_counts)),
            "city_count": len(city_counts),
            "occurrence_count": sum(city_counts.values()),
            "city_occurrence_counts": _format_city_counts(city_counts),
            **recovery_summary,
            "train_data_admission": train_data_admission,
            "admission_reason": admission_reason,
            **rule,
        }
        tag_rows.append(row)
        mapping_rows.append(row)

    member_rows: list[dict[str, Any]] = []
    for (relation_type, member_role, member_type), city_counts in sorted(audit.member_role_inventory.items()):
        rule = _role_rule(relation_type, member_role, member_type)
        pattern = f"relation:{relation_type}:member:{member_role}:{member_type}"
        recovery_summary = _recovery_summary_for_pattern(pattern, audit.recovery, audited_cities)
        train_data_admission, admission_reason = _train_admission(rule, recovery_summary)
        row = {
            "source_kind": "relation_member_role",
            "relation_type": relation_type,
            "member_role": member_role,
            "member_type": member_type,
            "osm_pattern": pattern,
            "cities": ";".join(sorted(city_counts)),
            "city_count": len(city_counts),
            "occurrence_count": sum(city_counts.values()),
            "city_occurrence_counts": _format_city_counts(city_counts),
            **recovery_summary,
            "train_data_admission": train_data_admission,
            "admission_reason": admission_reason,
            **rule,
        }
        member_rows.append(row)
        mapping_rows.append(row)

    tag_inventory_path = output_dir / "osm_tag_inventory.csv"
    member_inventory_path = output_dir / "osm_relation_member_role_inventory.csv"
    table_path = output_dir / "osm_training_mapping_table.csv"
    recovery_metrics_path = output_dir / "osm_recovery_metrics.csv"
    recovery_rows = _recovery_metric_rows(audit, audited_cities)

    _write_csv(
        tag_inventory_path,
        [
            "source_kind",
            "element",
            "tag_key",
            "tag_value",
            "osm_pattern",
            "cities",
            "city_count",
            "occurrence_count",
            "city_occurrence_counts",
            "recovery_total",
            "recovery_recovered",
            "recovery_rate",
            "min_city_recovery_rate",
            "observed_cities",
            "audited_city_coverage",
            "city_recovery_rates",
            "train_data_admission",
            "admission_reason",
            "scenarionet_feature_type",
            "mapping_status",
            "admission_decision",
            "degrade_strategy",
            "parser_action",
            "requires_explicit_osm_structure",
            "notes",
        ],
        tag_rows,
    )
    _write_csv(
        member_inventory_path,
        [
            "source_kind",
            "relation_type",
            "member_role",
            "member_type",
            "osm_pattern",
            "cities",
            "city_count",
            "occurrence_count",
            "city_occurrence_counts",
            "recovery_total",
            "recovery_recovered",
            "recovery_rate",
            "min_city_recovery_rate",
            "observed_cities",
            "audited_city_coverage",
            "city_recovery_rates",
            "train_data_admission",
            "admission_reason",
            "scenarionet_feature_type",
            "mapping_status",
            "admission_decision",
            "degrade_strategy",
            "parser_action",
            "requires_explicit_osm_structure",
            "notes",
        ],
        member_rows,
    )
    _write_csv(
        table_path,
        [
            "source_kind",
            "osm_pattern",
            "cities",
            "city_count",
            "occurrence_count",
            "city_occurrence_counts",
            "recovery_total",
            "recovery_recovered",
            "recovery_rate",
            "min_city_recovery_rate",
            "observed_cities",
            "audited_city_coverage",
            "city_recovery_rates",
            "train_data_admission",
            "admission_reason",
            "scenarionet_feature_type",
            "mapping_status",
            "admission_decision",
            "degrade_strategy",
            "parser_action",
            "requires_explicit_osm_structure",
            "notes",
        ],
        mapping_rows,
    )
    _write_csv(
        recovery_metrics_path,
        [
            "osm_pattern",
            "cities",
            "audited_city_coverage",
            "total_count",
            "recovered_count",
            "recovery_rate",
            "min_city_recovery_rate",
            "city_recovery_rates",
            "city_recovery_counts",
        ],
        recovery_rows,
    )

    summary = {
        "map_files": [str(path) for path in map_paths],
        "cities": audited_cities,
        "city_stats": {city: dict(sorted(stats.items())) for city, stats in sorted(audit.city_stats.items())},
        "training_mapping_table": str(table_path),
        "tag_inventory": str(tag_inventory_path),
        "relation_member_role_inventory": str(member_inventory_path),
        "recovery_metrics": str(recovery_metrics_path),
        "min_recovery_rate_for_training": MIN_RECOVERY_RATE_FOR_TRAINING,
        "included_patterns": [row["osm_pattern"] for row in mapping_rows if row["train_data_admission"] == "include"],
        "excluded_checked_patterns": [
            row["osm_pattern"] for row in mapping_rows if row["mapping_status"] != "unmapped" and row["train_data_admission"] == "exclude"
        ],
        "observed_predecessor_or_successor_roles": any(
            row["member_role"] in {"predecessor", "successor"} for row in member_rows
        ),
        "recovery": {row["osm_pattern"]: row for row in recovery_rows},
    }
    summary_path = output_dir / "osm_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown_report(output_dir, map_paths, audit.city_stats, mapping_rows, member_rows, recovery_rows)
    return table_path


def _polyline_from_way(node_lookup: dict[str, np.ndarray], way_elem: ET.Element) -> np.ndarray:
    points = [node_lookup[node_ref.attrib["ref"]] for node_ref in way_elem.findall("nd") if node_ref.attrib["ref"] in node_lookup]
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _average_centerline(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (left[:num_points] + right[:num_points]) / 2.0


def _polygon_from_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([left[:num_points], right[:num_points][::-1]], axis=0)


def parse_osm_map(
    osm_path: Path,
    admission_rules: TrainingMapAdmissionRules | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    admission_rules = admission_rules or default_training_admission_rules()
    root = ET.parse(osm_path).getroot()
    node_lookup = {
        node.attrib["id"]: project_lon_lat_to_xy(float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
        if "lon" in node.attrib and "lat" in node.attrib
    }
    way_lookup = {way.attrib["id"]: way for way in root.findall("way")}
    way_polylines = {way_id: _polyline_from_way(node_lookup, way) for way_id, way in way_lookup.items()}

    map_features: dict[str, dict[str, Any]] = {}
    lane_centers: dict[str, np.ndarray] = {}
    for way_id, way_elem in way_lookup.items():
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way_elem.findall("tag")}
        way_type = tags.get("type", "")
        if not admission_rules.allows(_metric_key_for_tag("way", "type", way_type)):
            continue
        feature_type = WAY_TYPE_TO_SCENARIONET.get(way_type)
        polyline = way_polylines[way_id]
        if feature_type is not None and len(polyline):
            map_features[f"boundary_{way_id}"] = {"type": feature_type, "polyline": polyline}

    for relation in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        members = {member.attrib["role"]: member.attrib["ref"] for member in relation.findall("member")}
        left = way_polylines.get(members.get("left", ""))
        right = way_polylines.get(members.get("right", ""))
        if left is None or right is None or len(left) == 0 or len(right) == 0:
            continue
        feature_name = tags.get("name", relation.attrib["id"])
        polygon = _polygon_from_boundaries(left, right)
        if tags.get("subtype") == "crosswalk":
            if not admission_rules.require_all(
                [
                    "relation:subtype=crosswalk",
                    "relation:lanelet:member:left:way",
                    "relation:lanelet:member:right:way",
                ]
            ):
                continue
            map_features[f"crosswalk_{feature_name}"] = {"type": "CROSSWALK", "polygon": polygon}
            continue
        if not admission_rules.require_all(
            [
                "relation:type=lanelet",
                "relation:lanelet:member:left:way",
                "relation:lanelet:member:right:way",
            ]
        ):
            continue
        centerline = _average_centerline(left, right)
        lane_centers[feature_name] = centerline
        map_features[f"lane_{feature_name}"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": centerline,
            "polygon": polygon,
            "entry_lanes": [],
            "exit_lanes": [],
        }
    return map_features, lane_centers
