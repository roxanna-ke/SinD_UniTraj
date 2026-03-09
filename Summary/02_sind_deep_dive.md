# 02. SinD 深度研究

相关文档：
- [README](./README.md)
- [01_relationships.md](./01_relationships.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)

## 1. SinD 是什么

SinD 是一个面向信号灯路口场景的无人机轨迹数据集，提供：

- 交通参与者轨迹
- 交通灯状态
- HD map（Lanelet2 `.osm`）

并且仓库中说明当前公开了 4 个城市的数据样本，完整数据需要额外申请。

证据：
- `SinD/README.md:3`
- `SinD/README.md:33-37`
- `SinD/README.md:41-45`

## 2. 目录与记录组织方式

文档里的逻辑结构是：每个 record 文件夹下面放轨迹与元数据文件，地图以 `.osm` 形式提供。

证据：
- `SinD/doc/File-Directory.md:3-30`

仓库当前 sample 的实际组织略有“按城市再按 record 分层”的样子，但可视化工具本质上只关心：

- 一个 record 目录
- 它下面的各类 CSV
- record 上层或相邻位置可找到 `.osm`

可视化器读取逻辑：
- record 路径 = `path/record_name`
- light 文件 = `TrafficLight*.csv`
- map 文件 = `path/*.osm`

证据：
- `SinD/SIND-Vis-tool/intersection_visualizer.py:36-52`

## 3. 原始文件清单

每个 record 主要包括：

- `Veh_smoothed_tracks.csv`
- `Ped_smoothed_tracks.csv`
- `Veh_tracks_meta.csv`
- `Ped_tracks_meta.csv`
- `TrafficLight_[record].csv`
- `recording_metas.csv`
- 对应 `.osm` map

证据：
- `SinD/README.md:41-45`
- `SinD/doc/File-Directory.md:22-30`
- `SinD/Format.md:8-123`

## 4. 车辆轨迹文件：`Veh_smoothed_tracks.csv`

字段如下：

- `track_id`
- `frame_id`
- `timestamp_ms`
- `agent_type`
- `x`, `y`
- `vx`, `vy`
- `yaw_rad`
- `heading_rad`
- `length`, `width`
- `ax`, `ay`
- `v_lon`, `v_lat`
- `a_lon`, `a_lat`

证据：
- `SinD/Format.md:8-32`
- `SinD/SIND-Vis-tool/utils/DataReader.py:8-13`

这已经足够支持下游构造：

- 位置
- 速度
- 朝向
- 尺寸
- 有效时间序列

### 下游转换时最关键的字段

如果目标是接 UniTraj，最关键的是：

- `x, y`
- `vx, vy`
- `yaw_rad`（或 `heading_rad`，需要统一语义）
- `length, width`
- `frame_id / timestamp_ms`

其中 `yaw_rad` 更像车身朝向，`heading_rad` 更像运动方向。转换时需要明确统一到下游所需的 `heading` 定义。

## 5. 行人轨迹文件：`Ped_smoothed_tracks.csv`

字段比车辆更简单：

- `track_id`
- `frame_id`
- `timestamp_ms`
- `agent_type`
- `x`, `y`
- `vx`, `vy`
- `ax`, `ay`

证据：
- `SinD/Format.md:34-43`
- `SinD/SIND-Vis-tool/utils/DataReader.py:34-36`

文档明确说：行人被当作**点**处理，不带尺度和朝向。

这意味着如果下游 schema 强制要求 `length/width/height/heading`，你需要自己补默认值或约定值。

## 6. 车辆元信息：`Veh_tracks_meta.csv`

它提供：

- `initialFrame`
- `finalFrame`
- `Frame_nums`
- `length`, `width`
- `class`
- `CrossType`
- `Signal_Violation_Behavior`

证据：
- `SinD/Format.md:45-60`

其中有两个很有价值的标签：

### `CrossType`
车辆穿越路口的运动类型：
- `StraightCross`
- `LeftTurn`
- `RightTurn`
- `Others`

### `Signal_Violation_Behavior`
车辆是否闯灯：
- `red-light running`
- `yellow-light running`
- `No violation of traffic lights`

这些标签不是 UniTraj 基础输入的必需字段，但很适合：

- 作为扩展 metadata
- 用于后续分析 / 分层评估
- 作为辅助监督或场景过滤条件

## 7. 行人元信息：`Ped_tracks_meta.csv`

该文件只提供轨迹起止与长度类信息，不提供更细的行为类型或违规标签。

证据：
- `SinD/Format.md:62-71`

因此，如果后续要做 pedestrian prediction，额外标签主要还是要靠轨迹本身，而不是依赖现成 annotation。

## 8. 交通灯文件：`TrafficLight_[record].csv`

字段包括：

- `RawFrameID`
- `timestamp(ms)`
- `Traffic light X`

并使用编码：
- `0 = red`
- `1 = green`
- `3 = yellow`

证据：
- `SinD/Format.md:73-81`

### 时间约定非常关键

文档明确说明：

```text
Frame * 3 = RawFrameID
```

也就是：轨迹数据帧与原始视频帧之间有一个 3 倍关系。

证据：
- `SinD/Format.md:79`

可视化器中也体现了同样的采样关系：

- `delta_time = 1 / 29.97 * 3`

证据：
- `SinD/SIND-Vis-tool/intersection_visualizer.py:79`

### 实现层面的读取方式

`read_light(...)` 会把“状态变化点”扩展成逐帧状态字典。

证据：
- `SinD/SIND-Vis-tool/utils/DataReader.py:56-78`

这说明原始 light 文件更像是**事件流 / 变化点表**，不是天然的逐 timestep 全状态表。转换到下游 scenario 时，需要先重建逐时间步的状态序列。

## 9. recording metadata

`recording_metas.csv` 提供：

- 城市
- 星期/时间段
- 天气
- 原始帧率
- 录制时长
- 参与者总量与各类别数量

证据：
- `SinD/Format.md:84-121`

其中最重要的技术约束之一是：

- 原始帧率固定为 `29.97 Hz`

证据：
- `SinD/Format.md:118`

## 10. 地图表示

SinD 提供的是 Lanelet2 `.osm` map，并明确说明地图原点与地面坐标系原点一致。

证据：
- `SinD/Format.md:123-123`
- `SinD/README.md:41-45`

这点非常重要，因为它说明：

- 轨迹坐标与地图坐标天然在同一参考系内
- 不需要额外做一层显式地图-轨迹配准

这对后续转换到 ScenarioNet 的 `map_features` 很有帮助。

## 11. 可视化工具暴露出来的读取约定

`DataReader.py` 很有参考价值，因为它等于给出了 SinD 原始格式的“程序化真相”：

- 精确断言了 CSV 列名
- 轨迹按 `track_id` 聚合
- 车辆可重建 bbox 与朝向三角形
- 行人默认按点处理

证据：
- `SinD/SIND-Vis-tool/utils/DataReader.py:6-52`
- `SinD/SIND-Vis-tool/utils/DataReader.py:91-142`

这意味着后续 converter 最稳妥的做法是：

- 先复用相同字段假设
- 不要自行猜测额外列名或隐藏字段

## 12. 对下游转换最重要的“可提取信息”

如果要把 SinD 转成 ScenarioNet / UniTraj 风格的数据，至少需要从 SinD 提取出以下几类信息：

### A. agent trajectories
- object id
- object class
- per-step position
- per-step velocity
- heading / orientation
- size（车辆直接有；行人需要补）
- valid mask

### B. scenario timing
- 全局 timestep 序列
- 统一采样间隔
- 当前历史/未来切分窗口

### C. map semantics
- lane centerline / boundary / crosswalk / stop line 等
- lane-level feature id

### D. traffic light states
- 将变化点表展开成逐时间步状态
- 如果要挂到 lane 上，需要建立 light 与 lane/map feature 的对应关系

## 13. 当前信息里仍然存在的模糊点

要真正实现 converter，下面这些问题还需要进一步确认：

1. `yaw_rad` 和 `heading_rad` 在下游应该选择哪个作为 canonical heading。
2. 行人在目标 schema 中的尺寸和朝向如何补默认值。
3. `Traffic light X(1-8)` 和具体 lane / stop line 的对应关系是否能从地图或文档中稳定恢复。
4. `.osm` 中哪些 Lanelet2 元素可以稳定映射到 ScenarioNet 所需的 `map_features` 类型。
5. train / val / test 如何划分（仓库未提供官方 split）。

## 14. 结论

SinD 的原始数据本身已经非常适合构建 trajectory prediction 数据集，因为它同时具备：

- 轨迹
- map
- 交通灯
- 一部分行为标签

但它仍然是**原始记录格式**，不是 UniTraj 直接消费的 scenario 数据格式。

因此，SinD 的角色应被理解为：

> 一个高价值的上游 raw dataset，需要经过结构化转换后才能进入 UniTraj。

下一步见：
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)