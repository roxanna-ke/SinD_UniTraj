# 08. 未决设计点与风险清单

相关文档：
- [README](./README.md)
- [05_sind_to_unitraj_integration.md](./05_sind_to_unitraj_integration.md)
- [06_field_level_mapping.md](./06_field_level_mapping.md)
- [07_repeatedly_verified_conclusions.md](./07_repeatedly_verified_conclusions.md)

## 1. 这份文档的目标

这份文档不再重复“已经确认的事实”，而是专门记录：

- 还没完全定死的设计选择
- 每个选择的风险
- 当前最推荐的默认方案

---

## 2. `yaw_rad` 还是 `heading_rad`

### 问题

SinD 同时提供：
- `yaw_rad`
- `heading_rad`

证据：
- `SinD/Format.md:11-24`

其中：
- `yaw_rad` 更像车身朝向
- `heading_rad` 更像运动方向

而 UniTraj / ScenarioNet 的 track state 只有一个：
- `heading`

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:178-181`

### 风险

如果选错：
- 车身朝向和速度方向可能不一致
- 转弯、低速、侧向滑动等情况的语义会变乱

### 当前推荐

- **优先使用 `yaw_rad` 作为 `heading`**
- 把 `heading_rad` 作为辅助分析字段（若你后续想保留扩展 metadata）

理由：
- 车辆 bounding box / 朝向通常更接近下游 `heading` 的设计意图
- 也更接近其他自动驾驶数据集中 object orientation 的习惯

### 仍需确认

如果后续发现 UniTraj 某些模型更依赖“运动方向”而不是“车身朝向”，这个决策可能需要重新评估。

---

## 3. pedestrian 默认尺寸与朝向

### 问题

SinD 行人没有：
- length
- width
- height
- heading

证据：
- `SinD/Format.md:34-43`

但 UniTraj 需要这些字段。

### 风险

默认值设置得太随意，会影响：
- 归一化统计
- 与 vehicle/cyclist 的类型区分
- 某些依赖尺寸的模型输入

### 当前推荐

MVP 先用固定默认值：
- length = 0.5
- width = 0.5
- height = 1.7 或 1.0
- heading = 由速度方向推导；低速时置 0

### 判断

- 这是工程折中，不是语义最优方案
- 如果第一阶段只做 vehicle，可以先完全跳过这个问题

---

## 4. 谁来当 `sdc_id`

### 问题

SinD 没有原生 ego vehicle，但 UniTraj 需要 `sdc_id`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:314`

### 风险

如果 `sdc_id` 选择策略不稳定：
- 不同 sample 的中心语义混乱
- ego-only fallback 路径不稳定
- 部分依赖 AV 标记的模型特征会有噪声

### 可选策略

1. 每个 scenario 固定一个中心 vehicle 作为 pseudo-SDC
2. 把当前主预测目标作为 pseudo-SDC
3. 每个 window 选离路口中心最近的 vehicle
4. 每个 window 选可见时间最长的 vehicle

### 当前推荐

MVP：
- **当前主预测目标 = pseudo-SDC**

理由：
- 最简单
- 最一致
- target 与 center 一致，比较符合样本构造直觉

---

## 5. 是否必须构造 `tracks_to_predict`

### 问题

理论上你可以不提供 `tracks_to_predict`，让 UniTraj 回退到 `object_summary`。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:322-330`
- `UniTraj/unitraj/datasets/base_dataset.py:991-1031`

### 风险

如果走回退路径：
- 你还要构造 `object_summary`
- target 选择标准变得更隐式
- debug 成本更高

### 当前推荐

- **显式提供 `tracks_to_predict`**

这是当前最稳的工程方案。

---

## 6. record 如何切成 scenario window

### 问题

SinD 的原始 record 很长，而 UniTraj 处理的是固定历史/未来窗口。

证据：
- `SinD/README.md:41`
- `UniTraj/unitraj/configs/config.yaml:15-16`

### 风险

window 设计不当会影响：
- 样本数量
- 轨迹完整性
- target 可用性
- map 裁剪稳定性
- train/val/test 分布

### 当前推荐

MVP：
- 固定长度窗口
- 与 `past_len + future_len` 对齐
- 用固定 stride 滑动生成样本

### 后续可优化

- 按交通灯变化点切
- 按车辆穿越路口事件切
- 按 interaction 密度筛选窗口

---

## 7. `dynamic_map_states` 做到什么程度

### 问题

SinD 的交通灯是强信号，但 lane-light 的精确对应不一定立刻能做完。

### 风险

如果第一版就追求完整 lane-light semantics：
- 实现周期会显著拉长
- debug 复杂度上升
- map/light 对齐容易成为最大阻塞点

### 当前推荐

MVP 分两阶段：

#### 第 1 阶段
- 保证 `dynamic_map_states` key 存在
- 允许为空或只做部分可靠映射

#### 第 2 阶段
- 再逐步建立：
  - light id -> lane id
  - stop_point
  - lane state over time

---

## 8. train/val/test 如何划分

### 问题

SinD 仓库本身没有看到现成 prediction benchmark split。

### 风险

如果切分不当：
- 数据泄漏
- 场景分布不均
- 评估结果不稳定

### 当前推荐

优先级从稳到快：

1. **按 record 切分**
2. 再考虑按城市切分
3. 不建议随机按 window 直接打散切分

理由：
- 同一个长 record 的相邻窗口高度相关
- 按 window 随机切容易泄漏近邻上下文

---

## 9. 是否一开始支持 pedestrian

### 问题

SinD 人车混合，但 pedestrian 需要补更多默认值和规则。

### 风险

第一版同时支持 vehicle + pedestrian 会让：
- 类型映射更复杂
- target 选择更复杂
- default state 设计更复杂

### 当前推荐

MVP：
- **只做 vehicle**

第二阶段再加 pedestrian。

---

## 10. 应该选 UniTraj 的哪条 dataset path

### 问题

UniTraj 有普通 `BaseDataset` 路线，也有 SMART 特化路线。

### 风险

SMART 对地图/动态状态更敏感，空 map 或不支持 feature 时更脆。

证据：
- `UniTraj/unitraj/datasets/base_dataset.py:405-407`
- `UniTraj/unitraj/datasets/SMART_dataset.py:565-581`

### 当前推荐

- 第一阶段优先面向普通 `BaseDataset` 兼容
- 不要先以 SMART 为目标

---

## 11. 当前建议的默认决策集

如果现在必须先定一版方案，我建议这样定：

- 只做 **vehicle**
- `heading = yaw_rad`
- `dataset = sind`
- `scenario_id = city_record_window`
- `sdc_id = 当前主预测目标`
- 显式提供 `tracks_to_predict`
- `dynamic_map_states` 第一版允许为空
- map 先做 lane centerline + boundary + crosswalk
- split 先按 record 切
- 先面向 UniTraj 普通 `BaseDataset`

---

## 12. 总结

这份风险清单的意义在于：

> 真正阻碍 SinD 接 UniTraj 的，不是“能不能写 converter”，而是“先把哪些设计决策定下来”。

如果这些决策先定了，后续实现路径会顺很多；
如果这些决策没定，代码实现会不断返工。