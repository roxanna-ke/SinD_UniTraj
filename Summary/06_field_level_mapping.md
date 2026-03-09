# 06. SinD -> ScenarioNet -> UniTraj 字段级映射表

相关文档：
- [README](./README.md)
- [02_sind_deep_dive.md](./02_sind_deep_dive.md)
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)
- [07_repeatedly_verified_conclusions.md](./07_repeatedly_verified_conclusions.md)

## 1. 这份文档的目的

这份文档把三层结构对齐到字段级：

- SinD 原始文件里有什么
- ScenarioNet / MetaDrive-style scenario 里应该如何表达
- UniTraj 最终实际会读取哪些字段

它不是实现代码，而是一个“转换设计表”。

---

## 2. 轨迹层：SinD -> `tracks`

UniTraj 在 `BaseDataset.preprocess()` 中读取每个 track 的这些字段：

- `type`
- `state.position`
- `state.length`
- `state.width`
- `state.height`
- `state.heading`
- `state.velocity`
- `state.valid`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:172-181`

ScenarioNet 多个 converter 中反复出现的 track 模板也基本一致。
证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:153-168`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:37-43`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:130-143`

### 2.1 Vehicle 轨迹字段映射

| SinD 来源 | 目标 ScenarioNet 字段 | UniTraj 是否直接使用 | 说明 |
|---|---|---:|---|
| `track_id` | `tracks[object_id]` 的 key | 是 | 建议统一转字符串 |
| `agent_type` | `tracks[...]['type']` | 是 | 需要映射到 MetaDriveType / UniTraj object_type |
| `x`, `y` | `state.position[:, 0:2]` | 是 | `z` 可补 0 |
| 无 | `state.position[:, 2]` | 是 | 默认 0 |
| `length` | `state.length[:]` | 是 | 车辆可直接用 |
| `width` | `state.width[:]` | 是 | 车辆可直接用 |
| 无 | `state.height[:]` | 是 | 需要给默认值 |
| `yaw_rad` 或 `heading_rad` | `state.heading[:]` | 是 | 需要明确采用哪个语义 |
| `vx`, `vy` | `state.velocity[:, 0:2]` | 是 | 可直接用 |
| 由时间覆盖推导 | `state.valid[:]` | 是 | object 在该 timestep 是否有效 |

SinD 车辆字段来源：
- `SinD/Format.md:8-32`

### 2.2 Pedestrian 轨迹字段映射

| SinD 来源 | 目标 ScenarioNet 字段 | UniTraj 是否直接使用 | 说明 |
|---|---|---:|---|
| `track_id` | `tracks[object_id]` 的 key | 是 | 建议统一转字符串 |
| `agent_type=pedestrian` | `tracks[...]['type']` | 是 | 映射到 `PEDESTRIAN` |
| `x`, `y` | `state.position[:, 0:2]` | 是 | `z` 补 0 |
| 无 | `state.length[:]` | 是 | 需要默认值 |
| 无 | `state.width[:]` | 是 | 需要默认值 |
| 无 | `state.height[:]` | 是 | 需要默认值 |
| 无 | `state.heading[:]` | 是 | 需要默认值或从速度推导 |
| `vx`, `vy` | `state.velocity[:, 0:2]` | 是 | 可直接用 |
| 由时间覆盖推导 | `state.valid[:]` | 是 | 需要构造 |

SinD 行人字段来源：
- `SinD/Format.md:34-43`

### 2.3 推荐默认策略

如果先做 MVP：

- pedestrian `length = 0.5`
- pedestrian `width = 0.5`
- pedestrian `height = 1.7` 或 `1.0`
- pedestrian `heading`：若速度足够大则由 `atan2(vy, vx)` 推导，否则置 0

这里不是“官方要求”，而是为了满足 UniTraj 的 state contract。

---

## 3. 轨迹元信息层：track metadata

ScenarioNet converter 中，track 内部还常带：

- `metadata.track_length`
- `metadata.type`
- `metadata.object_id`
- `metadata.dataset`

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:153-168`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:37-43`

对 SinD 来说推荐映射：

| SinD 来源 | 目标字段 | 说明 |
|---|---|---|
| 由窗口长度确定 | `metadata.track_length` | 取该 scenario window 长度 |
| `agent_type` / `class` | `metadata.type` | 与 `type` 保持一致 |
| `track_id` | `metadata.object_id` | 与 track key 保持一致 |
| 固定值 | `metadata.dataset` | 如 `sind` |

---

## 4. 时间层：SinD -> `metadata.ts` / `length`

ScenarioNet converter 中，metadata 里反复出现 timestep 数组：

- Waymo: `timestamps_seconds`
- AV2: `np.array(range(track_length)) / 10`
- nuScenes: 0.1s interval 的 `np.arange(...)`

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:373,393`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:217`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:439-455`

UniTraj 则要求 metadata 中有 `ts`，并会改名为 `timestamps_seconds`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:311-313`

### 映射建议

| SinD 来源 | 目标字段 | 说明 |
|---|---|---|
| `timestamp_ms` | `metadata.ts` | 建议转成秒 |
| window 长度 | `length` | scenario timestep 总数 |
| 固定采样策略 | `metadata.sample_rate`（可选） | 可补，但 UniTraj 不直接硬依赖 |

### 关键注意

SinD 有：
- raw frame rate = `29.97 Hz`
- data frame 与 raw frame 的关系：`Frame * 3 = RawFrameID`

证据：
- `SinD/Format.md:79`
- `SinD/Format.md:118`
- `SinD/SIND-Vis-tool/intersection_visualizer.py:79`

因此 window 切片和 `ts` 生成时，必须先决定：

- 是按 SinD 当前数据帧频直接建 timestep
- 还是重采样到别的统一时间间隔

---

## 5. 场景元数据层：SinD -> `metadata`

UniTraj 实际依赖：

- `scenario_id`
- `dataset`
- `ts`
- `sdc_id`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:311-315`
- `UniTraj/unitraj/datasets/base_dataset.py:350-351`
- `UniTraj/unitraj/datasets/base_dataset.py:448-449`

ScenarioNet 多个 converter 反复出现：

- `metadata[id]`
- `coordinate`
- `timestep`
- `metadrive_processed`
- `sdc_id`
- `dataset`
- `scenario_id`

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:390-399`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:214-223`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:445-458`

### 推荐映射表

| 目标 metadata 字段 | 来源 / 生成方式 | 是否建议必填 | 说明 |
|---|---|---:|---|
| `id` | 与 scenario id 一致 | 建议 | ScenarioDescription 顶层也有 `id` |
| `scenario_id` | `city + record + window_idx` | 是 | UniTraj 直接使用 |
| `dataset` | 固定 `sind` | 是 | UniTraj 输出 `dataset_name` |
| `ts` | window 内时间序列 | 是 | UniTraj 硬依赖 |
| `sdc_id` | pseudo-SDC 选择策略 | 是 | UniTraj 硬依赖 |
| `coordinate` | 与 SinD 坐标约定一致 | 建议 | ScenarioNet 常见字段 |
| `track_length` | window 长度 | 建议 | 多个 converter 会写 |
| `current_time_index` | `past_len - 1` | 建议 | 对预测任务有帮助 |
| `sdc_track_index` | 可选缓存 | 可选 | 可从 tracks 推出 |
| `tracks_to_predict` | 显式目标集合 | 强烈建议 | 可减少集成风险 |
| `objects_of_interest` | 可选 | 可选 | 不是 UniTraj 最小依赖 |
| `object_summary` | 如不用 `tracks_to_predict` 时需要 | 条件必填 | 用于自动筛 target |

---

## 6. 地图层：SinD `.osm` -> `map_features`

UniTraj 支持的 feature 类型主要由 `polyline_type` 控制。

证据：
- `UniTraj/unitraj/datasets/types.py:40-70`
- `UniTraj/unitraj/datasets/base_dataset.py:209-212`

### 6.1 优先映射的 feature

| SinD / Lanelet2 语义 | 目标 `map_features` 类型 | UniTraj 支持情况 | 优先级 |
|---|---|---:|---:|
| lane centerline | lane-like type | 高 | 高 |
| lane boundary / divider | road_line / road_edge | 高 | 高 |
| stop line / stop region | stop_sign 或 lane-related stop structure | 中 | 中 |
| crosswalk | crosswalk polygon | 高 | 中 |
| speed bump | speed_bump polygon | 有支持 | 低 |

### 6.2 字段级建议

#### lane-like feature

| 目标字段 | 说明 |
|---|---|
| `type` | 映射成 UniTraj 认识的 lane 类型 |
| `polyline` | lane centerline 的点序列 |
| `entry_lanes` | 前驱 lane ids |
| `exit_lanes` | 后继 lane ids |
| `left_neighbor` / `right_neighbor` | 若容易构建则保留，否则可空 |
| `speed_limit_mph` | 可空 |
| `interpolating` | 可给固定值 |

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:214-239`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:130-154`
- `scenarionet/scenarionet/converter/waymo/utils.py:214-239`

#### road line / edge

| 目标字段 | 说明 |
|---|---|
| `type` | 映射为 boundary / line 类型 |
| `polyline` | 边界点序列 |
| 或 `polygon` | 某些分支会 fallback 到 polygon |

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:240-251`

#### crosswalk

| 目标字段 | 说明 |
|---|---|
| `type = CROSSWALK` | 类型 |
| `polygon` | 多边形点序列 |

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:257-262`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:170-174`

### 6.3 unsupported type 的后果

在 `BaseDataset` 中，不支持的 map feature type 会被直接跳过。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:210-212`

所以：

- `.osm` 不是直接放进去就能用
- 必须先把 Lanelet2 元素翻译成 UniTraj 认识的类型

---

## 7. 信号灯层：SinD -> `dynamic_map_states`

Waymo converter 给了当前仓库里最清晰的 traffic light schema 示例：

- `type = TRAFFIC_LIGHT`
- `state.object_state = [per timestep signal state]`
- `lane`
- `stop_point`
- `metadata.track_length`

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:245-288`

UniTraj 也正是按这个形状去取字段：

- `lane`
- `stop_point`
- `state['object_state']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:282-304`

### SinD 映射建议

| SinD 来源 | 目标字段 | 说明 |
|---|---|---|
| `Traffic light X` | dynamic state object id / lane-linked light | 建议以 lane id 或稳定 light id 做 key |
| light 变化事件流 | `state.object_state[t]` | 需先展开成逐 timestep |
| 几何先验 / lane stop point | `stop_point` | 如果可推导 |
| movement / lane relation | `lane` | 最关键但也最难 |

### 最小可运行策略

如果 lane-light 对齐还没完全搞定，可以先：

- 保留 `dynamic_map_states = {}`
- 或只构建部分可靠映射

因为 UniTraj 的基础路径允许它为空，只要 key 存在。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:153`
- `UniTraj/unitraj/datasets/base_dataset.py:282-304`
- `UniTraj/unitraj/datasets/SMART_dataset.py:197-202`

---

## 8. target 选择层：`tracks_to_predict` / `object_summary`

### 8.1 如果显式提供 `tracks_to_predict`

推荐结构：

```python
tracks_to_predict = {
  object_id: {
    "track_index": ...,
    "track_id": ...,
    "difficulty": 0,
    "object_type": ...
  }
}
```

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:408-420`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:238-246`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:463-471`

UniTraj 读取时实际上主要用它的 key 来定位对象。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:331-339`

### 8.2 如果不提供 `tracks_to_predict`

`BaseDataset` 会退回到 `trajectory_filter()`，而它依赖 `object_summary`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:322-330`
- `UniTraj/unitraj/datasets/base_dataset.py:991-1031`

因此在 SinD 场景下，更稳的方式是：

- **直接提供 `tracks_to_predict`**

---

## 9. `sdc_id` 的处理建议

`sdc_id` 在 UniTraj 里是硬依赖，因为它会被拿去做：

- `sdc_track_index = track_infos['object_id'].index(ret['sdc_id'])`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:314`

SinD 不是 ego-car 采集数据，所以需要人为定义一个 pseudo-SDC。

### 推荐策略

MVP 阶段：
- 对每个 scenario window，选一个主 vehicle 作为 `sdc_id`
- 可以选：
  - 当前 `tracks_to_predict` 的主目标
  - 或距离路口中心最近的 vehicle
  - 或持续出现时间最长的中心 vehicle

只要该 id：
- 在 tracks 里真实存在
- 与当前窗口对齐

UniTraj 就能工作。

---

## 10. 总结

从字段级角度看，SinD 接 UniTraj 最关键的不是“字段够不够多”，而是：

1. 是否能构造出稳定的 `tracks`
2. 是否能把 `.osm` 翻译成 UniTraj 认识的 `map_features`
3. 是否能提供合法的 `metadata.ts`、`scenario_id`、`dataset`、`sdc_id`
4. 是否显式提供 `tracks_to_predict`

只要这四件事做对，MVP 跑通的可能性就很高。