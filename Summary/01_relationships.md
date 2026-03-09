# 01. 三个项目之间的关系

相关文档：
- [README](./README.md)
- [02_sind_deep_dive.md](./02_sind_deep_dive.md)
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)

## 1. 一句话关系图

```text
SinD (raw dataset)
  -> ScenarioNet / MetaDrive-style scenario dataset
  -> UniTraj preprocessing + cache
  -> trajectory prediction training/evaluation
```

## 2. 三个项目分别负责什么

### SinD：原始数据源

SinD 是一个基于无人机采集的信号灯路口轨迹数据集，提供：

- 交通参与者轨迹
- 交通灯状态
- Lanelet2 HD map

证据：
- `SinD/README.md:3`
- `SinD/README.md:41`
- `SinD/Format.md:8`
- `SinD/Format.md:73`
- `SinD/Format.md:123`

它给的是 **raw data**，不是 UniTraj 现成可读的数据集格式。

### ScenarioNet：场景标准化层

ScenarioNet 的定位是把不同自动驾驶数据集转成 MetaDrive 的 `ScenarioDescription` 风格，并构建可读取的 scenario 数据库。

证据：
- `scenarionet/README.md:28-35`
- `scenarionet/README.md:69-72`
- `scenarionet/scenarionet/converter/utils.py:207-257`
- `scenarionet/scenarionet/common_utils.py:83-110`

它不是训练框架，而是：

- 统一 schema
- converter 基础设施
- 面向 MetaDrive / scenario simulation / 下游 ML 的中间层

### UniTraj：轨迹预测训练框架

UniTraj 明确说明其输入来自 ScenarioNet。

证据：
- `UniTraj/README.md:111-114`
- `UniTraj/unitraj/datasets/base_dataset.py:10`
- `UniTraj/unitraj/datasets/base_dataset.py:60`
- `UniTraj/unitraj/datasets/base_dataset.py:120`

它不会直接读 `SinD/Data/*.csv`，而是读：

- ScenarioNet dataset summary
- scenario `.pkl` 文件
- 然后生成自己的 HDF5/cache

## 3. 为什么不能直接把 SinD 喂给 UniTraj

因为 SinD 和 UniTraj 的数据层抽象不同。

### SinD 提供的是

- CSV 轨迹表
- CSV 元信息表
- CSV 信号灯状态变化
- `.osm` 地图

证据：
- `SinD/Format.md:8-32`
- `SinD/Format.md:45-60`
- `SinD/Format.md:73-81`
- `SinD/Format.md:123-123`

### UniTraj 读取的是

- `scenario['tracks']`
- `scenario['map_features']`
- `scenario['dynamic_map_states']`
- `scenario['metadata']`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:152-156`
- `UniTraj/unitraj/datasets/base_dataset.py:306-345`

也就是说，UniTraj 依赖的是**已经结构化好的 scenario object**，不是原始表格文件。

## 4. ScenarioNet 在这里是不是必须的？

从当前仓库设计上看，**是最自然的路径**。

原因有两个：

1. UniTraj README 已明确写了它接 ScenarioNet 数据。
2. UniTraj 代码直接调用 `read_dataset_summary(...)` 和 `read_scenario(...)`。

证据：
- `UniTraj/README.md:111-114`
- `UniTraj/unitraj/datasets/base_dataset.py:60`
- `UniTraj/unitraj/datasets/base_dataset.py:120`

严格地说，也可以直接在 UniTraj 内部实现一个原始 SinD loader，但那是在**绕开**它现有的数据接入方式，不是最顺的方案。

## 5. 最合理的理解

### 角色划分

- **SinD**：提供真实路口观测数据
- **ScenarioNet**：定义中间场景表示与落盘方式
- **UniTraj**：消费这些 scenario，生成训练样本并训练模型

### 你真正要做的事情

不是“直接训练 SinD”，而是：

> 先把 SinD 变成 UniTraj 已经会读的 scenario 数据集。

## 6. 对接工作的本质

如果目标是让 `UniTraj/` 跑 `SinD/`，真正的核心工作是：

1. 把 SinD 的 agent trajectory 转成 `tracks`
2. 把 SinD 的 `.osm` map 转成 `map_features`
3. 把 SinD 的 `TrafficLight_*.csv` 转成 `dynamic_map_states`
4. 组织出 ScenarioNet 风格的数据集目录
5. 提供足够的 metadata，让 UniTraj 能选 target agent 并构建 cache

这些细节见：
- [02_sind_deep_dive.md](./02_sind_deep_dive.md)
- [03_scenarionet_deep_dive.md](./03_scenarionet_deep_dive.md)
- [04_unitraj_deep_dive.md](./04_unitraj_deep_dive.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)

## 7. 最后结论

当前最推荐的路线是：

```text
SinD raw files
  -> write SinD converter
  -> export ScenarioNet / MetaDrive-style scenarios
  -> point UniTraj train_data_path / val_data_path to converted dataset
  -> let UniTraj build cache and train
```

而不是：

```text
SinD raw files
  -> directly hack UniTraj to parse csv/osm/light files
```

后者虽然理论可行，但维护成本更高，也偏离 UniTraj 当前的数据契约。