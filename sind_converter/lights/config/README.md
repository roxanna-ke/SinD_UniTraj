Declarative traffic-light bindings live here.

- `channel_groups/<city>.yaml` maps raw traffic-light CSV columns to controlled `(stopline_id, movement)` groups.
- `lane_bindings/<city>.yaml` maps `(stopline_id, movement)` groups to ScenarioNet lane feature ids and stop points.
- The loader is dependency-light: files should use JSON-compatible YAML syntax.

If a city has no approved binding files yet, scenario conversion emits empty `dynamic_map_states` instead of using the old lane-order heuristic.
