# 04. UniTraj 深度研究

相关文档：
- [README](./README.md)
- [01_relationships.md](./01_relationships.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)

## 1. UniTraj 的数据入口是什么

README 直接说明：

> UniTraj takes data from ScenarioNet as input.

证据：
- `UniTraj/README.md:111-114`

代码层面，它通过 `scenarionet.common_utils` 的两个函数读数据：

- `read_dataset_summary(...)`
- `read_scenario(...)`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:10`
- `UniTraj/unitraj/datasets/base_dataset.py:60`
- `UniTraj/unitraj/datasets/base_dataset.py:120`

因此，UniTraj 期待的不是 raw CSV，而是**ScenarioNet 风格的 scenario 数据集根目录**。

## 2. 配置层面如何指定数据

数据路径来自配置：

- `train_data_path`
- `val_data_path`
- `cache_path`

证据：
- `UniTraj/unitraj/configs/config.yaml:10-12`

此外还有：
- `past_len`
- `future_len`
- `object_type`
- `line_type`
- `only_train_on_ego`
- `trajectory_sample_interval`
- `use_cache`
- `overwrite_cache`

证据：
- `UniTraj/unitraj/configs/config.yaml:13-25`

这意味着接入新数据集时，除了格式对齐，还要对齐：

- 历史长度 / 未来长度
- 参与训练的 object 类型
- 是否只预测 ego

## 3. UniTraj 读进来的 scenario 最少要包含什么

`BaseDataset.preprocess()` 一开始就直接取：

- `scenario['dynamic_map_states']`
- `scenario['tracks']`
- `scenario['map_features']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:152-156`

随后把 `scenario['metadata']` 合并进内部结构。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:306-345`

所以对 UniTraj 来说，一个 scenario 最低限度至少要有：

- `tracks`
- `map_features`
- `dynamic_map_states`
- `metadata`

## 4. tracks 的精确契约

UniTraj 遍历 `tracks.items()`，每个 track 都要求：

- `v['type']`
- `v['state']['position']`
- `v['state']['length']`
- `v['state']['width']`
- `v['state']['height']`
- `v['state']['heading']`
- `v['state']['velocity']`
- `v['state']['valid']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:172-181`

它把这些拼成每个 agent 的时序状态：

```text
[x, y, z, l, w, h, heading, vx, vy, valid]
```

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:178-181`

这对 SinD 的含义是：

- 车辆基本能直接映射
- 行人缺少 size / heading / height，需要补默认值
- 必须构造 `valid` mask

## 5. metadata 的关键字段

UniTraj 实际依赖的 metadata 里，最关键的是：

- `ts`
- `sdc_id`
- `scenario_id`
- `dataset`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:311-315`
- `UniTraj/unitraj/datasets/base_dataset.py:350-351`
- `UniTraj/unitraj/datasets/base_dataset.py:448-449`

其中：

- `ts` 会被改名成 `timestamps_seconds`
- `sdc_id` 用来找 `sdc_track_index`
- `scenario_id` 用于样本标识
- `dataset` 会变成输出里的 `dataset_name`

### 可选但很重要的字段

- `tracks_to_predict`
- `map_center`
- `object_summary`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:322-343`

## 6. target agent 是怎么选的

UniTraj 有三种路径：

### 路径 A：只训练 ego
如果 `only_train_on_ego=True`，那只预测 `sdc_id` 对应对象。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:316-321`

### 路径 B：metadata 已提供 `tracks_to_predict`
如果 scenario metadata 里已经有 `tracks_to_predict`，UniTraj 直接用它的 key 作为目标对象。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:331-341`

### 路径 C：自动过滤
如果没有 `tracks_to_predict`，它会调用 `trajectory_filter(...)`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:322-330`
- `UniTraj/unitraj/datasets/base_dataset.py:991-1031`

这个过滤器依赖 `object_summary` 中的：

- `type`
- `valid_length`
- `track_length`
- `moving_distance`

并执行规则：

- 类型必须是 `VEHICLE / PEDESTRIAN / CYCLIST`
- 有效占比至少 0.5
- vehicle 至少移动 2m
- 当前时刻必须有效

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:998-1029`

### 这对 SinD 的含义

如果你不想自己定义 `tracks_to_predict`，那就必须在 metadata 里额外构造 `object_summary`。

## 7. map_features 的契约

UniTraj 读取 `map_features` 时，并不是“有 polyline 就行”，而是会根据 feature type 分不同处理路径。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:199-280`
- `UniTraj/unitraj/datasets/types.py:40-70`

它主要支持：

- lane
- road_line
- road_edge
- stop_sign
- crosswalk
- speed_bump

每类 feature 期待的字段不同，例如：

### lane-like
- `polyline`
- 可选：`speed_limit_mph`, `interpolating`, `entry_lanes`
- 可选：`left_neighbor`, `right_neighbor`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:214-239`

### road_line / road_edge
- `polyline`，有时 road_line 也可退化用 `polygon`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:240-251`

### stop_sign
- `lane`
- `position`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:252-256`

### crosswalk / speed_bump
- `polygon`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:257-262`

### 重要限制

如果 feature type 不在 `polyline_type` 映射里，会被直接跳过。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:209-212`

所以 SinD 的 Lanelet2 地图不是“能读 `.osm` 就自动能训练”，必须显式转换成这些可识别的 feature type。

## 8. dynamic_map_states 的契约

UniTraj 假定每条动态地图状态都至少带：

- `lane`
- `stop_point`
- `state['object_state']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:282-304`

它会把这些整理成：

- `lane_id`
- `state`
- `stop_point`

对于标准模型，这部分虽然被解析，但没有直接成为 `ret_dict` 的核心输出；
但对部分特殊路径（如 SMART）会更重要。

因此，从“最小可运行”角度看：

- `dynamic_map_states` 键最好存在
- 即使先做空，也比完全缺失更稳
- 但如果要完整利用 SinD 的红绿灯信息，还是应该认真构建这部分

## 9. preprocess -> process -> cache

UniTraj 的数据流程大致是：

1. 读取 ScenarioNet 数据集 summary
2. 并行读取每个 scenario
3. `preprocess(scenario)`
4. `process(internal_format)`
5. `postprocess(output)`
6. 写入 HDF5 cache

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:39-107`
- `UniTraj/unitraj/datasets/base_dataset.py:109-150`
- `UniTraj/unitraj/datasets/base_dataset.py:152-468`

cache 目录是：

```text
cache_path/<dataset_name>/<phase>
```

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:46-48`

这说明一旦 SinD 被转换成合法 scenario 数据集，UniTraj 后续流程其实是通用的。

## 10. 输出样本里最重要的张量

README 也给出了核心样本结构：

- `obj_trajs`
- `obj_trajs_mask`
- `track_index_to_predict`
- `map_polylines`
- `map_polylines_mask`
- `center_gt_trajs`
- `dataset_name`

证据：
- `UniTraj/README.md:156-215`

这意味着你的上游转换器要做的，不是直接产出训练张量，而是：

> 产出足够标准的 scenario，使 UniTraj 能自己生成这些张量。

## 11. 一个新数据集接入 UniTraj 的最小契约

如果要新增 SinD，最小契约可以概括成：

1. 数据集目录能被 `read_dataset_summary(...)` 与 `read_scenario(...)` 读取。
2. 每个 scenario 有：
   - `tracks`
   - `map_features`
   - `dynamic_map_states`
   - `metadata`
3. `metadata` 至少有：
   - `scenario_id`
   - `dataset`
   - `ts`
   - `sdc_id`
4. 每个 track 至少有：
   - `type`
   - `state.position`
   - `state.length`
   - `state.width`
   - `state.height`
   - `state.heading`
   - `state.velocity`
   - `state.valid`
5. target 选择要么提供 `tracks_to_predict`，要么提供 `object_summary`，或者只训练 ego。

## 12. 结论

UniTraj 对新数据集并不要求“必须是某个官方 benchmark”，但要求它满足一套相当明确的 scenario contract。

所以对 SinD 而言，最关键的不是训练代码，而是先把 SinD 做成：

> 一个能满足 UniTraj `BaseDataset.preprocess()` 读取契约的 ScenarioNet-style 数据集。

下一步见：
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)