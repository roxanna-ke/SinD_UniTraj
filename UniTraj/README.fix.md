# UniTraj / ScenarioNet / SinD 调试与集成记录

## 1. 环境准备

```bash
conda create -n unitraj python=3.11
conda activate unitraj
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install natten==0.21.5+torch2100cu130 -f https://whl.natten.org
```

> NOTE: `torch` 的 CUDA 版本和 `natten` wheel 必须对应，否则 `python setup.py develop` 或运行时会报版本断言错误。

---

## 2. 这三个项目是怎么联系起来的

这次工作涉及 3 个相互衔接的项目：

### 2.1 SinD
路径：`../SinD/`

SinD 是**原始数据源**，提供：

- 车辆轨迹：`Veh_smoothed_tracks.csv`
- 行人轨迹：`Ped_smoothed_tracks.csv`
- 车辆 / 行人元数据：`Veh_tracks_meta.csv`、`Ped_tracks_meta.csv`
- 交通灯事件：`TrafficLight_*.csv`
- 高精地图：`map_relink_law_save.osm`

SinD 本身不是 UniTraj 可以直接训练的格式。

### 2.2 ScenarioNet
路径：`../scenarionet/`

ScenarioNet 在这里充当**中间数据契约 / 交换格式**。

UniTraj 的数据读取链路不是直接吃 SinD CSV/OSM，而是吃 **ScenarioNet 风格的数据集目录**，也就是：

- scenario `.pkl`
- `dataset_summary.pkl`
- `dataset_mapping.pkl`
- shard 目录（如 `sind_0/`）

换句话说：

> **SinD 原始数据必须先被转换成 ScenarioNet 风格，UniTraj 才能按现有 BaseDataset 流程读取。**

### 2.3 UniTraj
当前项目：`./`

UniTraj 是**训练与评估框架**。

它通过 `unitraj/datasets/base_dataset.py` 使用 ScenarioNet 的 summary / mapping / scenario 读取接口，把中间格式变成训练样本，然后交给 `AutoBot`、`Wayformer`、`MTR` 等模型训练。

### 2.4 当前这条链路的完整关系

```text
SinD 原始数据
  (CSV / TrafficLight / Lanelet2 OSM)
        ↓
自定义转换器
  unitraj/utils/sind_converter.py
        ↓
ScenarioNet 风格数据集
  dataset_summary.pkl / dataset_mapping.pkl / scenario pkl
        ↓
UniTraj BaseDataset
  unitraj/datasets/base_dataset.py
        ↓
UniTraj 模型训练
  unitraj/train.py -> AutoBot / others
```

---

## 3. 我做了哪些修改

下面按阶段列出本次真正做过的修改。

### 3.1 修复 UniTraj 训练入口和依赖问题

最开始运行：

```bash
python unitraj/train.py method=autobot
```

会报：

```python
ModuleNotFoundError: No module named 'unitraj'
```

根因是入口脚本的导入方式和包内部导入方式不一致。

我做了这些修复：

- 把入口脚本统一改成 `from unitraj...` 的绝对导入
- 给脚本运行模式增加 `sys.path` 引导，兼容：
  - `python -m unitraj.train ...`
  - `python unitraj/train.py ...`
- 给 `requirements.txt` 补上项目运行需要的 `metadrive` / `scenarionet`
- 把 `unitraj.models` / `unitraj.datasets` 改成**按需导入**，避免 `autobot` 训练时被 `SMART` 的可选依赖拖死
- 修复默认样例数据路径
- 增加 `remove_outliers: False` 默认值
- 自动重建空缓存 `file_list.pkl`
- 去掉当前 PyTorch 不兼容的 `MultiStepLR(..., verbose=True)`

涉及的关键文件：

- `unitraj/train.py`
- `unitraj/evaluation.py`
- `unitraj/k_means.py`
- `unitraj/data_analysis.py`
- `unitraj/models/__init__.py`
- `unitraj/datasets/__init__.py`
- `unitraj/configs/config.yaml`
- `unitraj/models/autobot/autobot.py`
- `requirements.txt`

### 3.2 跑通原始 UniTraj sample 训练

我验证了项目自带 sample 数据上的训练路径：

- train: `61 samples`
- val: `61 samples`
- GPU 训练可正常跑通
- WandB 使用 `offline` 模式避免 API key 阻塞

已验证可运行的命令：

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline python unitraj/train.py method=autobot debug=False 'devices=[0]' exp_name=test_gpu_restart
```

成功时通常会看到：

```text
GPU available: True (cuda), used: True
Trainer.fit stopped: max_epochs=100 reached.
```

在某些 Lightning 策略/版本下，还可能看到：

```text
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
```

### 3.3 新增 SinD -> ScenarioNet MVP 转换器

为了把 SinD 数据接到 UniTraj，我新增了：

- `unitraj/utils/sind_converter.py`

第一阶段做的是 MVP：

- vehicle-only
- fixed window slicing
- pseudo-SDC = 当前主预测目标
- 显式生成 `tracks_to_predict`
- 生成 ScenarioNet 风格：
  - `dataset_summary.pkl`
  - `dataset_mapping.pkl`
  - scenario `.pkl`

### 3.4 修复 UniTraj 对空地图 SinD scenario 的 ingest / NaN 问题

MVP 转换器最开始没有真实地图，因此暴露出两个问题：

1. `BaseDataset` 在空地图场景下处理不完整
2. `AutoBot` 的 map attention 在**所有 road token 都被 mask 掉**时会直接产生 `NaN`

我做了两个修复：

- `unitraj/datasets/base_dataset.py`
  - 允许空地图 scenario 继续构造训练样本
- `unitraj/models/autobot/autobot.py`
  - 对空地图 / 混合 batch 空地图做安全兜底，避免 attention 全 mask 导致数值炸掉

并加了回归测试，确保：

- 空地图 batch 前向不出 NaN
- mixed batch（空地图 + 非空地图）前向不出 NaN

### 3.5 增强 SinD 转换器：地图 / 信号灯 / 行人 / split

后续又把增强版功能补齐了：

#### 地图
- 直接解析 `map_relink_law_save.osm`
- 生成：
  - `LANE_SURFACE_STREET`
  - `ROAD_EDGE_BOUNDARY`
  - `ROAD_LINE_SOLID_SINGLE_WHITE`
  - `ROAD_LINE_BROKEN_SINGLE_WHITE`
  - `CROSSWALK`

#### 交通灯
- 解析 `TrafficLight_*.csv`
- 生成 `dynamic_map_states`
- 对每个 signal 导出长度等于 scenario length 的 `state.object_state`

#### 行人
- 接入：
  - `Ped_smoothed_tracks.csv`
  - `Ped_tracks_meta.csv`
- 导出 `PEDESTRIAN` tracks
- 对行人补默认长度 / 宽度 / 高度 / heading 逻辑

#### 数据集级 split
- 新增 `convert_sind_dataset(...)`
- 先扫描**完整可转换的 record**
- 再按 record 做 train / val 切分，避免窗口泄漏

涉及的关键文件：

- `unitraj/utils/sind_converter.py`
- `tests/test_sind_converter.py`

---

## 4. 当前已验证的内容

### 4.1 tests

我已经跑过：

```bash
python -m pytest tests/test_sind_converter.py -q
```

结果：

- `4 passed`
- `1 skipped`

`skip` 的原因不是实现失败，而是当前 sample 仓库里只有 **1 个完整可转换 record**（`Tianjin/8_2_1`），所以无法在 sample 数据上真实做“跨多个完整 record 的 split 验证”。

### 4.2 增强版 SinD 数据生成结果

增强版数据会生成到：

```bash
converted_data/sind_enhanced/train/sind
converted_data/sind_enhanced/val/sind
```

我已经确认其中一个 scenario 包含：

- `map_features`: 非空
- map feature types 包含：
  - `CROSSWALK`
  - `LANE_SURFACE_STREET`
  - `ROAD_EDGE_BOUNDARY`
  - `ROAD_LINE_BROKEN_SINGLE_WHITE`
  - `ROAD_LINE_SOLID_SINGLE_WHITE`
- `dynamic_map_states`: 8 个 signal
- `PEDESTRIAN` track: 至少 1 个

### 4.3 增强版 GPU smoke test

我已经真实运行过：

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline python unitraj/train.py \
  method=autobot \
  debug=False \
  'devices=[0]' \
  exp_name=sind_enhanced_smoke \
  'train_data_path=[converted_data/sind_enhanced/train/sind]' \
  'val_data_path=[converted_data/sind_enhanced/val/sind]' \
  'max_data_num=[null]' \
  'starting_frame=[0]' \
  method.max_epochs=1
```

结果：

- `GPU available: True (cuda), used: True`
- `Trainer.fit stopped: max_epochs=1 reached.`
- 指标为有限值，不是 NaN：
  - `val/minADE6: 2.351`
  - `val/minFDE6: 2.196`
  - `val/brier_fde: 2.891`

也就是说：

> **增强版 SinD -> ScenarioNet -> UniTraj 训练链路已经实际跑通。**

---

## 5. 现在可以怎么使用

### 5.1 跑 UniTraj 自带 sample 数据

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline python unitraj/train.py method=autobot debug=False 'devices=[0]' exp_name=test_gpu_restart
```

### 5.2 生成增强版 SinD 数据

```bash
python -m unitraj.utils.sind_converter \
  --sind-record-dir "/home/pejoy/code/epfl/SinD/Data/Tianjin/8_2_1" \
  --output-dir "/home/pejoy/code/epfl/UniTraj/converted_data/sind_enhanced" \
  --city Tianjin \
  --dataset-name sind \
  --dataset-version v1 \
  --stride 40 \
  --max-scenarios 8 \
  --train-ratio 0.75
```

### 5.3 用增强版 SinD 数据跑 UniTraj

```bash
PYTHONUNBUFFERED=1 WANDB_MODE=offline python unitraj/train.py \
  method=autobot \
  debug=False \
  'devices=[0]' \
  exp_name=sind_enhanced_smoke \
  'train_data_path=[converted_data/sind_enhanced/train/sind]' \
  'val_data_path=[converted_data/sind_enhanced/val/sind]' \
  'max_data_num=[null]' \
  'starting_frame=[0]' \
  method.max_epochs=1
```

---

## 6. 当前已提交的 milestone commits

本次相关 milestone commit：

- `8d5ca62` — `Fix UniTraj training entrypoints and sample GPU run.`
- `dc6613a` — `Add SinD-to-ScenarioNet MVP converter.`
- `961657c` — `Fix AutoBot empty-map instability for SinD smoke runs.`
- `dbab12f` — `Ignore generated SinD conversion outputs.`
- `b0625b7` — `Enhance SinD conversion with maps, signals, and pedestrians.`

---

## 7. 当前仍然存在的限制

### 7.1 sample SinD 仓库不是完整全集

当前 sample 仓库里，完整可转换的 record 实际只有：

- `Tianjin/8_2_1`

所以：

- `convert_sind_dataset(...)` 已经实现
- 但在 sample 数据上无法真实验证“多个完整 record 的 split 训练”
- 如果换成完整 SinD 数据集，这条路径就能真正使用 record-level split

### 7.2 交通灯 lane 对齐目前是 MVP 方案

当前交通灯已经能导出 `dynamic_map_states` 并让 UniTraj 正常读取，但**精确 lane / stop-point 对齐**仍然是后续可以继续增强的方向。

### 7.3 目前重点是跑通，不是最终高保真标注

当前这版更偏**工程可运行基线**：

- 可转换
- 可读取
- 可训练
- 数值稳定

如果后面要提升效果，可以继续在：

- lane connectivity
- light -> lane 精准绑定
- richer map semantics
- 多 city / 多 record split
- 更长训练与效果分析

上继续做。

---

## 8. 一句话总结

这次我做的事情可以概括为：

> **先修好 UniTraj 自己的训练入口和 sample 训练，再把 SinD 原始数据转换成 ScenarioNet 风格中间格式，最后让 UniTraj 能稳定读取并在 GPU 上完成增强版 SinD smoke training。**
