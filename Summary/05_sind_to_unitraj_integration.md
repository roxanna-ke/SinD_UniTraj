# 05. SinD -> UniTraj 集成路径

相关文档：
- [README](./README.md)
- [01_relationships.md](./01_relationships.md)
- [02_sind_deep_dive.md](./02_sind_deep_dive.md)
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)

## 1. 最终目标

目标不是“让 UniTraj 直接解析 SinD 原始 CSV”，而是：

> 让 SinD 经过标准化转换后，作为一个新的 scenario dataset 被 UniTraj 正常读取、缓存、训练。

## 2. 推荐的总体路线

```text
SinD raw record
  -> parse trajectories / map / lights
  -> build ScenarioDescription-style scenarios
  -> write dataset summary + mapping + scenario pkls
  -> point UniTraj train_data_path / val_data_path to converted dataset
  -> UniTraj preprocesses and builds HDF5 cache
  -> train / eval
```

## 3. 为什么推荐这条路线

因为：

1. UniTraj README 已明确声明输入来自 ScenarioNet。
   证据：`UniTraj/README.md:111-114`
2. UniTraj 代码直接依赖 `read_dataset_summary(...)` 和 `read_scenario(...)`。
   证据：`UniTraj/unitraj/datasets/base_dataset.py:60`, `UniTraj/unitraj/datasets/base_dataset.py:120`
3. ScenarioNet 本身就提供 converter / summary / mapping / dataset merge 的基础设施。
   证据：`scenarionet/scenarionet/converter/utils.py:77-149`, `scenarionet/scenarionet/converter/utils.py:167-264`

## 4. 从 SinD 到 ScenarioNet，需要做哪些字段映射

## 4.1 trajectories -> `tracks`

SinD 里可直接利用的源字段：

- `track_id`
- `agent_type`
- `x, y`
- `vx, vy`
- `yaw_rad` / `heading_rad`
- `length, width`
- `frame_id`, `timestamp_ms`

证据：
- `SinD/Format.md:8-32`

### 建议映射

- `track_id` -> scenario object id
- `agent_type` / `class` -> scenario `type`
- `x, y` -> `state.position[:, 0:2]`
- `z` -> 默认 0
- `length, width` -> `state.length`, `state.width`
- `height` -> 默认值（需要明确策略）
- `yaw_rad` 或 `heading_rad` -> `state.heading`
- `vx, vy` -> `state.velocity[:, 0:2]`
- `valid` -> 根据该 object 在窗口内是否存在构造

### 行人需要补的内容

SinD 行人没有：
- size
- heading
- height

证据：
- `SinD/Format.md:34-43`

而 UniTraj 读取 track state 时要求这些字段存在。
证据：`UniTraj/unitraj/datasets/base_dataset.py:172-181`

因此行人一定需要补默认值。

## 4.2 `.osm` -> `map_features`

SinD 提供 Lanelet2 `.osm` map。
证据：`SinD/Format.md:123-123`

UniTraj 期待 `map_features` 中的 feature 具有：

- type
- polyline / polygon / lane / position 等几何信息

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:199-280`
- `UniTraj/unitraj/datasets/types.py:40-70`

### 最小建议

最小可运行版本优先提取：

- lane centerlines
- road boundaries / road lines
- crosswalk（如果容易提）

不建议一开始就追求 Lanelet2 全语义无损映射，而是优先确保：

- feature type 能映射到 UniTraj 支持的类别
- geometry 是稳定且可切片的

## 4.3 `TrafficLight_*.csv` -> `dynamic_map_states`

SinD 的 light 文件是状态变化点表。
证据：`SinD/Format.md:73-81`

读取器会将其展开为逐帧 light state。
证据：`SinD/SIND-Vis-tool/utils/DataReader.py:56-78`

UniTraj 期待的 dynamic map state 至少有：

- `lane`
- `stop_point`
- `state['object_state']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:282-304`

### 关键难点

要真正把 SinD 信号灯用起来，需要明确：

- `Traffic light X` 分别控制哪些 lane / movement
- `stop_point` 应该放在地图中的什么位置

如果这部分尚未完全明确，**最小可运行版本**可以先：

- 保留 `dynamic_map_states` 键
- 暂时填空或只做部分 lane-light 对应

因为基础 UniTraj 标准模型对这部分依赖没有 tracks/map 那么强。

## 4.4 record metadata -> `metadata`

UniTraj 至少依赖：

- `scenario_id`
- `dataset`
- `ts`
- `sdc_id`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:311-315`
- `UniTraj/unitraj/datasets/base_dataset.py:350-351`
- `UniTraj/unitraj/datasets/base_dataset.py:448-449`

### 需要特别决策的字段

#### `scenario_id`
建议由：
- city
- record name
- window index
组合生成。

#### `dataset`
固定写成例如：
- `sind`

#### `ts`
由窗口内每一步时间戳组成，单位与语义要与 UniTraj 预期一致（秒级时间序列更自然）。

#### `sdc_id`
这是最麻烦的字段之一，因为 SinD 不是 ego-vehicle 采集数据。

可选策略：
1. 固定选某个中心 vehicle 作为 pseudo-SDC
2. 每个样本以 target object 为中心时，令其自身充当 SDC
3. 只依赖 `tracks_to_predict` / `object_summary`，弱化 ego 语义

从工程角度看，**pseudo-SDC** 是最常见也最稳妥的折中。

## 5. scenario 切片策略是关键设计点

SinD 的原始 record 时长是 8-22 分钟量级。
证据：`SinD/README.md:41`

而 UniTraj 的训练样本逻辑是固定 past/future window。
证据：`UniTraj/unitraj/configs/config.yaml:15-16`

因此不应该：

- 一整条 record 直接变成一个训练 scenario

而应该：

- 把长 record 切成多个固定长度的 scenario window

### 建议切法

基于：
- `past_len`
- `future_len`
- `trajectory_sample_interval`
- SinD 的时间采样约定

构建滑窗或步进窗口。

这一步会直接影响：

- 样本数量
- 目标 agent 可用性
- map 裁剪范围
- train / val / test 的切分方式

## 6. target agents 怎么定义

有三种现实可选方案：

### 方案 A：显式提供 `tracks_to_predict`
优点：
- 最清晰
- 不依赖 UniTraj 的自动过滤逻辑

### 方案 B：提供 `object_summary`，让 UniTraj 自动筛
优点：
- 更接近现有通用逻辑

缺点：
- 你还要额外构造 `valid_length`、`moving_distance` 等 summary 字段

### 方案 C：只训练 ego / pseudo-ego
优点：
- 实现最简单

缺点：
- 损失多目标预测能力

### 我的建议

最小可运行版本优先选：

- **vehicle only**
- **显式提供 `tracks_to_predict`**

这样最容易绕开 `object_summary` 这条额外工作链。

## 7. 最小可运行版本（MVP）建议

如果目标是尽快跑通一版，不建议一开始就追求“完整复刻 SinD 所有语义”。

### 建议范围

1. **只做 vehicle**
2. **只选一个城市 / 一个 record**
3. **先不要求完整 traffic-light-to-lane 映射**
4. **先把 map 做到 lane + boundary 即可**
5. **显式提供 `tracks_to_predict`**
6. **先做 train/val 两个目录的最小 split**

### 这样做的好处

你可以先验证：

- ScenarioNet 数据能否正确落盘
- UniTraj 能否正确读 summary/scenario
- cache 能否成功生成
- 模型训练是否能真正启动

等这个闭环通了，再加：

- pedestrian
- full traffic light semantics
- richer map features
- 更正式的 split 方案

## 8. 当前最重要的未决问题

要真正开工实现前，最好先定掉这几个决策：

1. 用 `yaw_rad` 还是 `heading_rad` 作为下游 `heading`
2. 行人默认 size/height/heading 怎么设
3. 谁充当 `sdc_id`
4. 是否显式提供 `tracks_to_predict`
5. scenario window 的长度、步长、采样率
6. train / val / test 划分按 record 切，还是按 city 切
7. traffic light 与 lane 的对应关系要做到什么精度

## 9. 推荐结论

如果你接下来要真正落地实现，我建议优先按这个顺序：

```text
Step 1: single-record, vehicle-only, fixed-window conversion
Step 2: build ScenarioNet-style dataset on disk
Step 3: point UniTraj config to converted train/val paths
Step 4: verify cache generation
Step 5: start training
Step 6: add pedestrians / lights / richer maps
```

## 10. 总结

这次深度研究之后，最清晰的判断是：

- **SinD 本身没有问题，数据也足够有价值**
- **真正的门槛是“场景化转换”而不是“训练代码”**
- **最推荐的路径是做一个 SinD -> ScenarioNet-style converter，再接 UniTraj**

换句话说：

> 你现在最需要解决的不是模型，而是数据契约。