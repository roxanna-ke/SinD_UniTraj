# 07. 反复确认后的结论与硬性要求

相关文档：
- [README](./README.md)
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [06_field_level_mapping.md](./06_field_level_mapping.md)
- [08_open_decisions_and_risks.md](./08_open_decisions_and_risks.md)

## 1. 这份文档的定位

这份文档只记录“反复确认后比较稳定的结论”，并区分：

- **硬性要求（hard requirement）**
- **强烈建议（strongly recommended）**
- **可选项（optional）**
- **仍有不确定性（uncertain）**

我对这些结论的确认来源包括：

1. 本地仓库源码交叉阅读
2. 多个 ScenarioNet converter 互相比对
3. 官方文档/搜索结果的外部交叉验证

---

## 2. 已反复确认的顶层 scenario 结构

### 结论 2.1：顶层 key 是稳定的

一个 canonical scenario 的顶层结构稳定包含：

- `id`
- `version`
- `length`
- `tracks`
- `dynamic_map_states`
- `map_features`
- `metadata`

### 本地源码证据
- `scenarionet/scenarionet/converter/waymo/utils.py:366-395`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:441-461`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:192-223`

### 外部文档交叉确认
- MetaDrive Scenario Description
- ScenarioNet new dataset support / description

### 判断
- **硬性要求**

---

## 3. 已反复确认的数据集磁盘布局

### 结论 3.1：ScenarioNet-style dataset 至少围绕 summary + scenario pkl 组织

本地代码读取接口表明：

- 会读取 dataset summary
- 会根据 mapping 找 scenario 文件路径
- 每个 scenario 以 `.pkl` 形式落盘

证据：
- `scenarionet/scenarionet/common_utils.py:83-110`
- `scenarionet/scenarionet/converter/utils.py:207-257`

### 结论 3.2：`dataset_summary.pkl` 是核心入口

证据：
- `scenarionet/scenarionet/common_utils.py:83-95`
- `UniTraj/unitraj/datasets/base_dataset.py:60`

### 结论 3.3：`dataset_mapping.pkl` 很重要，但外部文档里被描述为可选

本地读取接口会依赖 mapping；
外部搜索/文档结果表明 `dataset_mapping.pkl` 常被描述为 optional。

### 判断
- `dataset_summary.pkl`：**硬性要求**
- `dataset_mapping.pkl`：**在你当前本地链路里几乎应视为必备**
- scenario `.pkl`：**硬性要求**

换句话说：

> 即便官方文档说 mapping 可能可选，你为了和当前本地 UniTraj/ScenarioNet 链路稳稳兼容，最好把它一并生成。

---

## 4. 已反复确认的 track contract

### 结论 4.1：下游实际读取的 track state 字段是固定的

UniTraj 读取：

- `position`
- `length`
- `width`
- `height`
- `heading`
- `velocity`
- `valid`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:172-181`

多个 converter 中也重复出现同样的模板。

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:153-168`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:37-43`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:130-143`

### 判断
- **硬性要求**

### 对 SinD 的含义

- vehicle：大部分字段原生具备
- pedestrian：需要补 size / height / heading

---

## 5. 已反复确认的 metadata contract

### 结论 5.1：`scenario_id`、`dataset`、`ts`、`sdc_id` 对 UniTraj 非常关键

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:311-315`
- `UniTraj/unitraj/datasets/base_dataset.py:350-351`
- `UniTraj/unitraj/datasets/base_dataset.py:448-449`

### 判断
- `scenario_id`：**硬性要求**
- `dataset`：**硬性要求**
- `ts`：**硬性要求**
- `sdc_id`：**硬性要求**

### 结论 5.2：metadata 的其余字段并不完全固定

不同 converter 里会出现：
- `coordinate`
- `source_file`
- `track_length`
- `current_time_index`
- `sdc_track_index`
- `objects_of_interest`
- `tracks_to_predict`

但并不是每个数据集都完全一样。

证据：
- `scenarionet/scenarionet/converter/waymo/utils.py:390-420`
- `scenarionet/scenarionet/converter/nuscenes/utils.py:445-475`
- `scenarionet/scenarionet/converter/argoverse2/utils.py:214-246`

### 判断
- 这些字段是 **强烈建议** 或 **条件必需**，不是绝对统一硬要求

---

## 6. 已反复确认的 `dynamic_map_states` 结论

### 结论 6.1：`dynamic_map_states` 这个 key 本身必须存在

UniTraj 使用的是直接索引，不是 `.get(...)`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:153`
- `UniTraj/unitraj/datasets/SMART_dataset.py:115`

### 判断
- **硬性要求：key 必须存在**

### 结论 6.2：它的内容可以为空

空 dict 不会立刻导致 `BaseDataset` 失败。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:282-304`
- `UniTraj/unitraj/datasets/SMART_dataset.py:366-384`

### 判断
- **内容可选**

这意味着 SinD 的 MVP 版本可以先：

- `dynamic_map_states = {}`

之后再逐步增加 lane-light 对齐。

---

## 7. 已反复确认的 `tracks_to_predict` / `object_summary` 结论

### 结论 7.1：`tracks_to_predict` 不是绝对硬性要求

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:316-341`
- `UniTraj/unitraj/datasets/SMART_dataset.py:395-403`

### 结论 7.2：但如果没有 `tracks_to_predict`，BaseDataset 常常会转而要求 `object_summary`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:322-330`
- `UniTraj/unitraj/datasets/base_dataset.py:991-1031`

### 判断
- `tracks_to_predict`：**可选，但强烈建议提供**
- `object_summary`：**条件必需**（当你不提供 `tracks_to_predict` 时）

### 最稳判断

对于 SinD：

> 为了降低风险，MVP 阶段最好显式生成 `tracks_to_predict`，不要依赖 `object_summary` 回退路径。

---

## 8. 已反复确认的 map feature 结论

### 结论 8.1：map feature type 必须落在 UniTraj 能识别的类别里，否则会被跳过

证据：
- `UniTraj/unitraj/datasets/types.py:40-70`
- `UniTraj/unitraj/datasets/base_dataset.py:210-212`

### 判断
- **硬性要求：至少要有一批可识别的 map features**

### 结论 8.2：BaseDataset 对空地图更宽容，SMART 更脆弱

BaseDataset 在地图空时会补 dummy zero polyline。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:405-407`

SMART 路径对空/不支持地图更容易出问题。

证据：
- `UniTraj/unitraj/datasets/SMART_dataset.py:236-248`
- `UniTraj/unitraj/datasets/SMART_dataset.py:565-581`

### 判断
- 如果你的目标只是先让 UniTraj 跑起来，优先考虑走普通 `BaseDataset` 路线而不是 SMART。

---

## 9. 已反复确认的 `sdc_id` 结论

### 结论 9.1：`sdc_id` 不只是 metadata 装饰字段，而是硬依赖

UniTraj 会直接用它去做 `.index(...)` 查找。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:314`
- `UniTraj/unitraj/datasets/SMART_dataset.py:394`

如果 `sdc_id` 缺失，或者它不在 track id 列表里，预处理会失败。

### 判断
- **硬性要求**

### 对 SinD 的实际含义

SinD 没有天然 ego，所以你必须显式设计一个 pseudo-SDC 规则。

---

## 10. 已反复确认的时间/采样结论

### 结论 10.1：不同 converter 的 `ts`/timestep 策略并不完全统一

证据：
- Waymo：`scenarionet/scenarionet/converter/waymo/utils.py:393`
- AV2：`scenarionet/scenarionet/converter/argoverse2/utils.py:217`
- nuScenes：`scenarionet/scenarionet/converter/nuscenes/utils.py:454-455`

### 判断
- **仍有策略空间，不是固定死的单一标准**

但对 UniTraj 来说，最重要的是：

- 你提供的 `ts` 与 track state 时间维度要自洽
- window 切片和 `past_len/future_len` 要匹配

---

## 11. 当前最稳的工程判断

经过多轮交叉验证后，当前最稳的实现判断是：

1. SinD -> UniTraj 的关键瓶颈是 **数据契约转换**，不是训练代码。
2. 最稳路线是 **先做 ScenarioNet-style dataset，再给 UniTraj**。
3. MVP 阶段最应该保证：
   - 车辆轨迹正确
   - metadata 完整
   - map_features 至少有 lane / boundary
   - `dynamic_map_states` key 存在
   - `tracks_to_predict` 明确
   - `sdc_id` 有定义且可索引

---

## 12. 最终结论

这轮“反复确认”后，可以把最小硬要求总结成：

### 必须有
- 顶层 scenario keys
- `tracks`
- `map_features`
- `dynamic_map_states` key
- `metadata.scenario_id`
- `metadata.dataset`
- `metadata.ts`
- `metadata.sdc_id`
- track state 的 `position/length/width/height/heading/velocity/valid`

### 强烈建议有
- `tracks_to_predict`
- `track_length`
- `current_time_index`
- `coordinate`
- `source_file`

### 可以后补
- 完整 traffic-light-to-lane 对齐
- pedestrian 支持
- richer map semantics
- object_summary fallback 路径

如果你下一步准备实现，这份文档可以当作“必须满足的验收前置条件”。