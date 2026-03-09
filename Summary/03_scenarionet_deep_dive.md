# 03. ScenarioNet 深度研究

相关文档：
- [README](./README.md)
- [01_relationships.md](./01_relationships.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)

## 1. ScenarioNet 的定位

ScenarioNet 是一个面向自动驾驶数据的 scenario 标准化平台。README 明确写到：

- 可以加载 Waymo、nuPlan、nuScenes、l5 等真实数据集
- 也可以接 synthetic / adversarial scenario
- 构建出的数据库可用于 ML 的 train/test set
- 底层由 MetaDrive 支撑，用于 simulation、RL、imitation learning 等

证据：
- `scenarionet/README.md:28-35`

这说明它在整个链路中的角色是：

- 不是原始数据集
- 不是单纯训练框架
- 而是一个 **统一场景表示 + converter 基础设施**

## 2. Canonical schema 来自哪里

README 直接把 ScenarioNet dataset / scenario description 指向 MetaDrive 的 `Scenario Description` 文档。

证据：
- `scenarionet/README.md:69-72`

在代码里，converter 直接使用：

- `from metadrive.scenario import ScenarioDescription as SD`

证据：
- `scenarionet/scenarionet/converter/utils.py:15`

所以可以把 ScenarioNet 理解为：

> 采用 MetaDrive 的 `ScenarioDescription` 作为场景 canonical schema，并围绕它构建数据集落盘、summary、mapping、converter workflow。

## 3. 数据集磁盘布局是怎样的

从 `read_dataset_summary(...)` 和 `read_scenario(...)` 看，读取路径以“数据集根目录”为入口。

证据：
- `scenarionet/scenarionet/common_utils.py:83-110`

### 读取接口的意义

`read_dataset_summary(dataset_path)` 返回：

1. summary dict
2. scenario file name list
3. scenario file name -> folder mapping

证据：
- `scenarionet/scenarionet/common_utils.py:83-95`

`read_scenario(dataset_path, mapping, scenario_file_name)` 则会拼出：

```python
os.path.join(dataset_path, mapping[scenario_file_name], scenario_file_name)
```

证据：
- `scenarionet/scenarionet/common_utils.py:98-110`

这意味着 ScenarioNet 数据集目录中至少需要：

- scenario `.pkl` 文件
- dataset summary file
- dataset mapping file

## 4. summary / mapping 文件是如何生成的

在 converter 基础设施中：

- `summary_file = SD.DATASET.SUMMARY_FILE`
- `mapping_file = SD.DATASET.MAPPING_FILE`
- 然后通过 `save_summary_and_mapping(...)` 落盘

证据：
- `scenarionet/scenarionet/converter/utils.py:207-212`
- `scenarionet/scenarionet/converter/utils.py:255-257`
- `scenarionet/scenarionet/common_utils.py:72-80`

每个 scenario 在转换后会：

1. 生成一个 export file name
2. 写进 `summary[export_file_name] = metadata`
3. 写进 `mapping[export_file_name] = ""`（同目录）
4. pickle 落盘整个 scenario dict

证据：
- `scenarionet/scenarionet/converter/utils.py:224-246`

因此，ScenarioNet 数据集的本质是：

- 一批场景级 `.pkl`
- 一个数据集级 summary
- 一个数据集级 mapping

## 5. converter 架构长什么样

最关键的入口是：

- `write_to_directory(...)`
- `write_to_directory_single_worker(...)`

证据：
- `scenarionet/scenarionet/converter/utils.py:77-149`
- `scenarionet/scenarionet/converter/utils.py:167-264`

### 它做了什么

1. 将待转换场景列表分给多个 worker
2. 每个 worker 调用 `convert_func(...)`
3. `convert_func` 返回一个 MetaDrive `ScenarioDescription`
4. 对结果做 `sanity_check`
5. 写出 scenario `.pkl`
6. 汇总 summary / mapping
7. 最后 `merge_database(...)`

这表明如果你要新增 SinD converter，最自然的方式就是：

- 准备一个 `convert_sind_scenario(...)`
- 让它返回合法的 `ScenarioDescription`
- 交给现有 `write_to_directory(...)` 基础设施统一落盘

## 6. 一个 canonical scenario 至少长什么样

从 nuScenes converter 的构造过程，可以很清晰地看出 scenario 的关键字段：

- `SD.ID`
- `SD.VERSION`
- `SD.LENGTH`
- `SD.METADATA`
- `SD.TRACKS`
- `SD.DYNAMIC_MAP_STATES`
- `SD.MAP_FEATURES`

证据：
- `scenarionet/scenarionet/converter/nuscenes/utils.py:441-479`

metadata 中又常见：

- `dataset`
- `scenario_id`
- `sample_rate`
- `ts / timestep`
- `sdc_id`
- `tracks_to_predict`（预测任务里可选但常用）

证据：
- `scenarionet/scenarionet/converter/nuscenes/utils.py:445-475`

## 7. tracks / map / dynamic map states 的语义

虽然本仓库把 schema 说明外链到 MetaDrive 文档，但从 converter 与 UniTraj 读取方式，已经能反推出最重要的结构层级：

### `tracks`
每个 object 需要：
- object type
- time-series state
- 通常包含 position / heading / velocity / valid 等

### `map_features`
以 feature id 为键，value 带：
- type
- polyline / polygon / lane / position 等几何字段

### `dynamic_map_states`
表达车道/信号灯等随时间变化的状态，至少要能描述：
- 关联 lane
- stop point
- object state over time

这些正好也是 UniTraj 会读取的核心字段。

## 8. 它和 MetaDrive 的关系

ScenarioNet 与 MetaDrive 是直接耦合的，而不是松散兼容。

证据：
- `scenarionet/README.md:33-35`
- `scenarionet/scenarionet/converter/utils.py:15`

所以如果把 SinD 转成 ScenarioNet，其实也是在转成：

> MetaDrive 能理解的一种标准场景表示。

这不仅对 UniTraj 有价值，也对后续仿真/回放/场景重构有价值。

## 9. 如果要新增 SinD converter，需要做什么

最小化理解下，需要完成四类工作：

### A. 读取 SinD raw record
输入来自：
- `Veh_smoothed_tracks.csv`
- `Ped_smoothed_tracks.csv`
- `Veh_tracks_meta.csv`
- `Ped_tracks_meta.csv`
- `TrafficLight_*.csv`
- `.osm`

### B. 生成一个合法的 `ScenarioDescription`
至少填入：
- `ID`
- `VERSION`
- `LENGTH`
- `METADATA`
- `TRACKS`
- `DYNAMIC_MAP_STATES`
- `MAP_FEATURES`

### C. metadata 里补齐下游需要的字段
尤其是：
- `dataset`
- `scenario_id`
- `ts`
- `sdc_id`
- `tracks_to_predict` 或能推导出它的 summary

### D. 用 `write_to_directory(...)` 落盘成完整数据集目录
这样下游就可以直接通过 `read_dataset_summary(...)` 和 `read_scenario(...)` 读取。

## 10. 对 SinD 来说最难的部分

新增 SinD converter 时，真正有技术不确定性的地方大概是：

1. **Lanelet2 `.osm` -> `map_features` 的映射策略**
2. **TrafficLight_X -> lane / stop line 的对应关系**
3. **谁是 `sdc_id`**
4. **是否要提供 `tracks_to_predict`**
5. **如何定义一个 record 内的 scenario 切片方式**

其中最关键的是：

- SinD 原始记录是长时连续录像
- ScenarioNet / UniTraj 更偏向“单个场景窗口”

因此你很可能需要把一条长 record 切成多个 fixed-length scenario window，而不是“一整个 record = 一个 scenario”。

## 11. 结论

ScenarioNet 在这个问题里最应该被理解成：

> SinD 原始数据和 UniTraj 训练框架之间的标准化桥梁。

如果不走这座桥，你就需要直接修改 UniTraj 的数据入口；
如果走这座桥，你可以顺着已有生态把 SinD 组织成一个标准 scenario dataset。

下一步见：
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)