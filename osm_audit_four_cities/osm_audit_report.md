# OSM Audit Report

## Audited Maps

- Tianjin: `map_relink_law_save.osm`
- Changchun: `Changchun_Pudong.osm`
- Chongqing: `NR_ll2.osm`
- Xi_an: `Xi_an_Shanglin.osm`

## City Stats

- Changchun: nodes=409, ways=59, relations=37
- Chongqing: nodes=455, ways=88, relations=52
- Tianjin: nodes=788, ways=100, relations=70
- Xi_an: nodes=827, ways=94, relations=56

## Final Training Map Feature Inputs

- `relation:subtype=crosswalk` -> `CROSSWALK` (construct_crosswalk_polygon_from_lanelet_relation)
- `relation:type=lanelet` -> `LANE_SURFACE_STREET` (construct_lane_from_lanelet_relation)
- `way:type=curbstone` -> `BOUNDARY_LINE` (map_way_type_to_boundary_polyline)
- `way:type=line_thick` -> `ROAD_LINE_SOLID_SINGLE_WHITE` (map_way_type_to_road_line_polyline)
- `way:type=line_thin` -> `ROAD_LINE_SOLID_SINGLE_WHITE` (map_way_type_to_road_line_polyline)
- `way:type=stop_line` -> `ROAD_LINE_SOLID_SINGLE_WHITE` (map_stop_line_way_to_road_line_polyline)
- `way:type=virtual` -> `ROAD_LINE_BROKEN_SINGLE_WHITE` (map_virtual_way_to_broken_road_line_polyline)
- `relation:lanelet:member:left:way` -> `LANE_SURFACE_STREET` (use_way_as_left_lane_boundary)
- `relation:lanelet:member:right:way` -> `LANE_SURFACE_STREET` (use_way_as_right_lane_boundary)

## Audited But Excluded

- `relation:subtype=traffic_light` (inventory_traffic_light_regulatory_relations): No final ScenarioNet map feature mapping is defined for this pattern.
- `relation:type=regulatory_element` (inventory_regulatory_element_for_future_linking): No final ScenarioNet map feature mapping is defined for this pattern.
- `way:type=traffic_light` (inventory_traffic_light_way_for_future_linking): No final ScenarioNet map feature mapping is defined for this pattern.
- `relation:regulatory_element:member:ref_line:way` (inventory_stop_or_light_reference_line): No final ScenarioNet map feature mapping is defined for this pattern.
- `relation:regulatory_element:member:refers:way` (inventory_regulatory_target_way): No final ScenarioNet map feature mapping is defined for this pattern.

## Recovery Metrics

- `relation:lanelet:member:left:way`: total=199, recovered=199, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `relation:lanelet:member:right:way`: total=199, recovered=199, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `relation:regulatory_element:member:ref_line:way`: total=8, recovered=8, rate=1.0, per_city=Chongqing:1.0;Tianjin:1.0
- `relation:regulatory_element:member:refers:way`: total=8, recovered=8, rate=1.0, per_city=Chongqing:1.0;Tianjin:1.0
- `relation:subtype=crosswalk`: total=4, recovered=4, rate=1.0, per_city=Tianjin:1.0
- `relation:subtype=traffic_light`: total=8, recovered=8, rate=1.0, per_city=Chongqing:1.0;Tianjin:1.0
- `relation:type=lanelet`: total=199, recovered=199, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `relation:type=regulatory_element`: total=8, recovered=8, rate=1.0, per_city=Chongqing:1.0;Tianjin:1.0
- `way:type=curbstone`: total=62, recovered=62, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `way:type=line_thick`: total=4, recovered=4, rate=1.0, per_city=Chongqing:1.0
- `way:type=line_thin`: total=59, recovered=59, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `way:type=stop_line`: total=16, recovered=16, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0
- `way:type=traffic_light`: total=8, recovered=8, rate=1.0, per_city=Chongqing:1.0;Tianjin:1.0
- `way:type=virtual`: total=153, recovered=153, rate=1.0, per_city=Changchun:1.0;Chongqing:1.0;Tianjin:1.0;Xi_an:1.0

## Topology Note

- No explicit predecessor/successor relation-member roles were observed in the audited OSM files. Lane topology beyond left/right boundaries will require dedicated lanelet graph inference or a lanelet2 API pass.
- Stable `left/right` lanelet boundary roles were observed across audited maps.
