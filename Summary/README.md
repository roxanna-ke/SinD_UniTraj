# Summary

这组文档总结了对以下三个仓库的深度研究结果：

- `SinD/`：原始路口轨迹数据集
- `scenarionet/`：面向 MetaDrive 的统一 scenario 表示与转换层
- `UniTraj/`：消费 ScenarioNet 数据进行轨迹预测训练的框架

## 最核心结论

`UniTraj` **不能直接读取** `SinD` 的原始 CSV / OSM 数据。

推荐的数据路径是：

```text
SinD raw data
  -> convert to ScenarioNet / MetaDrive ScenarioDescription-style dataset
  -> UniTraj reads scenarios and builds its own cache
  -> training / evaluation
```

也就是说：

- **SinD** 是数据源
- **ScenarioNet** 是中间的场景标准化层
- **UniTraj** 是下游训练与评估框架

## 推荐阅读顺序

1. [`01_relationships.md`](./01_relationships.md)
   - 三个项目之间的关系
   - 为什么不能直接把 `SinD/` 丢给 `UniTraj/`

2. [`02_sind_deep_dive.md`](./02_sind_deep_dive.md)
   - SinD 的原始文件结构、字段、时间约定、地图和信号灯表示
   - 下游转换器必须从 SinD 提取什么

3. [`03_scenarionet_deep_dive.md`](./03_scenarionet_deep_dive.md)
   - ScenarioNet 的 canonical schema、磁盘布局、converter 架构
   - 新增 SinD converter 需要满足什么

4. [`04_unitraj_deep_dive.md`](./04_unitraj_deep_dive.md)
   - UniTraj 的输入契约、预处理/缓存流程、target-agent 选择逻辑
   - 一个新数据集最少要满足哪些字段

5. [`05_sind_to_unitraj_integration.md`](./05_sind_to_unitraj_integration.md)
   - 从 SinD 接到 UniTraj 的具体落地路径
   - 建议先做的最小可运行版本

6. [`06_field_level_mapping.md`](./06_field_level_mapping.md)
   - SinD -> ScenarioNet -> UniTraj 的字段级映射表
   - 逐层说明轨迹、metadata、地图、信号灯如何对齐

7. [`07_repeatedly_verified_conclusions.md`](./07_repeatedly_verified_conclusions.md)
   - 多轮源码/文档交叉确认后的稳定结论
   - 区分哪些是硬性要求，哪些只是建议

8. [`08_open_decisions_and_risks.md`](./08_open_decisions_and_risks.md)
   - 仍未完全定死的设计点
   - 每个决策的风险和当前推荐默认方案

## 文档之间的关系

- `02_sind_deep_dive.md` 解释 **源数据长什么样**。
- `03_scenarionet_deep_dive.md` 解释 **中间目标格式长什么样**。
- `04_unitraj_deep_dive.md` 解释 **下游消费者真正依赖什么**。
- `05_sind_to_unitraj_integration.md` 把前三者连起来，说明 **应该如何转换与接入**。
- `06_field_level_mapping.md` 进一步下钻到 **字段级映射**，适合实现前逐项对照。
- `07_repeatedly_verified_conclusions.md` 汇总 **已经多轮确认的稳定结论**，适合作为硬性约束清单。
- `08_open_decisions_and_risks.md` 记录 **仍需拍板的设计点与风险**，适合实现前先定方案。

## 研究范围说明

本次研究只做了：

- 仓库结构与源码/文档阅读
- 输入输出契约梳理
- 数据路径与转换约束分析

本次研究没有做：

- 代码实现
- 数据转换脚本编写
- 训练跑通验证
- 指标复现

因此，这组文档是 **架构与数据契约层面的研究总结**，适合作为下一步实现前的参考。