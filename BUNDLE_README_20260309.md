# SinD / ScenarioNet / UniTraj 集成包说明

## 这个包里有什么

这个压缩包包含 4 个目录：

- `SinD/`
  - 原始数据集仓库
  - 包含 sample 轨迹、交通灯、Lanelet2 OSM 地图、可视化工具与原始文档
- `scenarionet/`
  - ScenarioNet 源码仓库
  - 负责统一 scenario schema、converter、summary/mapping 落盘与读取
- `UniTraj/`
  - 轨迹预测训练框架
  - 当前已包含 SinD -> ScenarioNet -> UniTraj 的接入实现与修复
- `Summary/`
  - 研究与设计文档
  - 包括三个项目关系、字段映射、集成路径、风险与已确认结论

## 推荐阅读顺序

1. `Summary/README.md`
2. `Summary/01_relationships.md`
3. `Summary/05_sind_to_unitraj_integration.md`
4. `Summary/06_field_level_mapping.md`
5. `UniTraj/README.fix.md`

## 三个项目怎么联系起来

```text
SinD raw data
  -> converter in UniTraj (unitraj/utils/sind_converter.py)
  -> ScenarioNet-style dataset
  -> UniTraj BaseDataset / cache
  -> UniTraj model training
```

更具体地说：

- `SinD` 提供原始 CSV / traffic light / OSM 数据
- `ScenarioNet` 提供中间场景格式和 summary/mapping 落盘规范
- `UniTraj` 读取 ScenarioNet 风格的数据集进行训练与评估

## 当前已经实现了什么

在 `UniTraj/` 里已经完成并验证：

1. 修复训练入口和依赖问题
2. 修复 sample 数据训练路径
3. 实现 `unitraj/utils/sind_converter.py`
   - SinD -> ScenarioNet MVP 转换
4. 修复空地图情况下 UniTraj / AutoBot 的数值稳定性
5. 增强转换器，支持：
   - Lanelet2 OSM 地图导出
   - traffic light state 导出
   - pedestrian track 导出
   - record-level dataset split API
6. 增加回归测试
   - `tests/test_sind_converter.py`
7. 真实跑通增强版 SinD GPU smoke training

## 关键运行入口

### 1. 环境安装

在 `UniTraj/` 下参考：

- `UniTraj/README.fix.md`

核心安装命令：

```bash
conda create -n unitraj python=3.11
conda activate unitraj
cd UniTraj
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install natten==0.21.5+torch2100cu130 -f https://whl.natten.org
```

### 2. 生成增强版 SinD 数据

```bash
cd UniTraj
python -m unitraj.utils.sind_converter \
  --sind-record-dir "/path/to/SinD/Data/Tianjin/8_2_1" \
  --output-dir "/path/to/UniTraj/converted_data/sind_enhanced" \
  --city Tianjin \
  --dataset-name sind \
  --dataset-version v1 \
  --stride 40 \
  --max-scenarios 8 \
  --train-ratio 0.75
```

### 3. 用增强版 SinD 数据跑 UniTraj

```bash
cd UniTraj
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

## 已验证结果

当前已经验证：

- `pytest tests/test_sind_converter.py -q`
  - 结果：`4 passed, 1 skipped`
- 增强版 SinD 数据可成功生成
- 增强版 SinD 数据可被 UniTraj 成功读取
- 增强版 GPU smoke test 可正常完成，且指标为有限值

## 这次打包刻意排除了什么

为了减小压缩包体积并避免把本地构建/运行产物打进去，本包排除了：

- `.git/`
- `__pycache__/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.ruff_cache/`
- `node_modules/`
- `build/`
- `dist/`
- `outputs/`
- `cache/`
- `lightning_logs/`
- `tmp/`
- `wandb/`
- `unitraj_ckpt/`
- `converted_data/`
- `src/`（UniTraj 本地 editable 依赖 clone 产物）
- 其他常见运行期文件

## 重要限制

- 当前 sample SinD 仓库里只有一个完整可转换的 record：`Tianjin/8_2_1`
- 所以 `convert_sind_dataset(...)` 虽然已经实现，但在 sample 数据上无法真实验证多 record split
- 交通灯与 lane 的精确绑定目前是 MVP 方案，足够跑通训练，但不是最终高保真语义版

## 如果你只想快速看结论

优先看：

- `Summary/README.md`
- `UniTraj/README.fix.md`

如果你要看设计细节，再看：

- `Summary/05_sind_to_unitraj_integration.md`
- `Summary/06_field_level_mapping.md`
- `Summary/07_repeatedly_verified_conclusions.md`
- `Summary/08_open_decisions_and_risks.md`
