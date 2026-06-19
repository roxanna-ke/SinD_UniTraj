# SinD_UniTraj 四城全量数据处理 Pipeline 实现规范

  ## 摘要

  本文定义一套放置在 SinD_UniTraj 仓库内、独立于 UniTraj/unitraj/utils 的正式数据处理 pipeline，用于将完整四城 SinD/Dataset 原始数据转换为：

  1. 可复用的 ScenarioNet 风格中间产物
  2. 支持多种 split 策略的场景数据集
  3. 可直接供 UniTraj 中 MTR、AutoBot、Wayformer 训练与评估使用的 cache

  本规范以四城完整数据兼容、可复用中间产物、稳健地图提取、可切换 split 策略为核心目标。实现必须避免把原始解析、场景生成、split、cache 构建再次混成单个大文件。

  ## 1. 目标与成功标准

  ### 1.1 最终目标

  基于 SinD/Dataset 四个城市的完整数据，构建一套独立 pipeline，完成：

  SinD raw data
    -> canonical ScenarioNet scenarios
    -> split assignment
    -> UniTraj cache
    -> training / evaluation for MTR, AutoBot, Wayformer

  ### 1.2 成功标准

  实现完成后，必须满足：

  - 可扫描并兼容 SinD/Dataset 四个城市全部 record
  - 可自动处理四城文件命名差异和 meta 缺失差异
  - 可生成合法的 ScenarioNet 风格数据集目录
  - 可基于同一批 canonical scenarios 生成不同 split
  - 可构建 UniTraj cache，并被 MTR、AutoBot、Wayformer 直接读取
  - 不依赖 UniTraj/unitraj/utils/sind_converter.py 的单文件式结构
  - 允许 richer map best-effort 输出，但不能因单城地图语义缺失而整体失败
  - 正式全量转换和训练默认运行环境为远端集群
  - 在正式scenarionet conversion之前必须先完成四城OSM tag audit，并据此冻结地图输出范围

  ## 2. 仓库内模块边界

  ### 2.1 新模块位置

  新 pipeline 必须放在 SinD_UniTraj 仓库内的独立板块，不放在 unitraj 下。推荐结构：

  SinD_UniTraj/
    sind_converter/
      data/
      maps/
      lights/
      scenarios/
      splits/
      cache/
      scripts/
      config/

  ### 2.2 职责拆分

  实现必须拆成以下逻辑层：

  - data: 原始 SinD 目录发现、文件规范化、四城兼容加载
  - maps: OSM/Lanelet2 地图解析、静态地图特征提取
  - lights: 交通灯 CSV 解析与标准化
  - scenarios: 窗口生成、track 构造、metadata 构造、ScenarioNet 输出
  - splits: 基于 canonical scenarios 生成 record-level 与 city-level split
  - cache: 调用 UniTraj 数据入口生成 cache
  - scripts: 命令行入口
  - config: 本地/集群可切换配置

  ### 2.3 明确禁止

  禁止继续采用以下结构：

  - 将 raw 解析、地图解析、window 生成、ScenarioNet 落盘、split、cache 混成单个 .py
  - 在实现中写死本机路径或集群路径
  - 将 split 逻辑和 raw conversion 强耦合

  ## 3. 输入数据与兼容范围

  ### 3.1 正式支持的原始输入

  正式输入根目录为：

  - 兼容本地数据路径 SinD/Dataset
  - 本地SinD/Dataset仅用于调试、开发期检查和小规模smoke test
  - 正式 raw 数据根目录默认是：/scratch/izar/ke/sind_raw/ 该目录下包含四个城市子目录。这是四城完整数据的主输入来源。

  ### 3.2 地图回退目录

  由于 Dataset/Tianjin 不含 .osm，实现必须支持地图回退根目录：

  - SinD/Data

  城市地图映射规则固定为：

  - Tianjin -> Data/Tianjin/map_relink_law_save.osm
  - Changchun -> Dataset/Changchun/Changchun_Pudong.osm，若不存在则回退 Data/Changchun/Changchun_Pudong.osm
  - Chongqing -> Dataset/Chongqing/NR_ll2.osm，若不存在则回退 Data/Chongqing/NR_ll2.osm
  - Xi_an -> Dataset/Xi_an/Xi_an_Shanglin.osm，若不存在则回退 Data/Xi'an/Xi'an_Shanglin.osm

  ### 3.3 路径配置原则

  所有路径必须参数化，不得写死在代码中。至少支持以下配置项：

  - data_root
  - map_fallback_root
  - canonical_scenario_root
  - split_root
  - cache_root




  ### 3.4 四城文件兼容要求

  实现必须兼容以下差异：

  - 交通灯文件名不统一
  - Veh_tracks_meta.csv、Ped_tracks_meta.csv、recording/recoding_metas.csv 只在部分城市存在
  - 地图文件名按城市不同
  - Tianjin 与其他三城目录形态不同

  因此输入发现层必须输出统一 record 描述对象，至少包含：

  - city
  - record_name
  - vehicle_tracks_path
  - pedestrian_tracks_path
  - traffic_light_path | None
  - vehicle_meta_path | None
  - pedestrian_meta_path | None
  - recording_meta_path | None
  - map_path

  ## 4. 三段式 Pipeline 规范

  ### 4.0 Stage 0: audit-maps

  输入：

  - 四城OSM文件
  
  输出：

  - 四城 OSM tag inventory
  - stable mapping table
  - v1 mandatory map tags
  - best-effort map tags
  - 城市级降级策略

 只有当四城 stable mapping table 已生成，并且每个候选 tag 已被标记为 mandatory / best-effort / skip 后，才能开始正式 conversion。
 
  ### 4.1 Stage 1: convert-scenarios

  输入：

  - SinD/Dataset
  - SinD/Data 作为地图回退根目录

  输出：

  - 一套不含 split 语义的 canonical scenarios
  - 每个 scenario 为合法 ScenarioNet 风格 .pkl
  - 对应 dataset_summary.pkl
  - 对应 dataset_mapping.pkl

  要求：

  - raw 解析与 ScenarioNet 构造在此阶段完成
  - 不在此阶段写死 train/val/test
  - scenario id 必须稳定、可复现、与 split 无关

  ### 4.2 Stage 2: make-splits

  输入：

  - canonical scenarios
  - split 配置

  输出：

  - record-level split 数据集目录
  - city-level holdout split 数据集目录

  要求：

  - 同一批 canonical scenarios 可反复生成不同 split
  - 不重复做 raw 解析和地图解析
  - split assignment 必须独立保存并可追溯

  ### 4.3 Stage 3: build-cache

  输入：

  - split 后的 ScenarioNet 数据集目录
  - UniTraj 训练配置

  输出：

  - UniTraj cache
  - 结构兼容现有 BaseDataset 读取逻辑

  要求：

  - 产物必须可被 MTR、AutoBot、Wayformer 直接用于训练和评估
  - 不允许依赖手工改动 UniTraj 内部代码路径来完成普通构建

  ## 5. Scenario 组织方式

  ### 5.1 采用 shared multi-target

  正式规范采用：

  - shared multi-target per scenario

  即：

  - 一个 scenario 是一个共享时间窗口
  - 一个 scenario 内允许多个 tracks_to_predict
  - UniTraj 后续可将一个 shared scenario 展开为多个最终 training samples

  不采用：

  - 每个 scenario 只绑定单一 focal target 的长期默认设计

  ### 5.2 target 类型范围

  正式 target policy：

  - 仅 car / truck / bus 允许进入 tracks_to_predict

  其他 agent 的策略：

  - motorcycle 作为 context 保留，不作为 target
  - bicycle / tricycle 作为 context 保留，不作为 target
  - pedestrian 作为 context 保留，不作为 target

  ### 5.3 sdc_id 策略

  1. sdc_id 必须从 tracks_to_predict 中选择。
  2. 若有多个 target，按 track_id 升序排序。
  3. 排序后的第一个 target 作为 sdc_id。

  ### 5.4 tracks_to_predict 必须显式提供

  正式采用 shared multi-target per scenario。对每个 81 帧 scenario window，按以下规则生成 tracks_to_predict：

  1. 仅 car / truck / bus 有资格成为 target。
  2. motorcycle / bicycle / tricycle / pedestrian 只作为 context 保留，不进入 tracks_to_predict。
  3. 一个对象要进入 tracks_to_predict，必须满足：
      - 在当前 window 的中心时刻有效存在；
      - 在 past_len = 21 中至少有 21 帧有效历史；
      - 在 future_len = 60 中至少有 60 帧有效未来；
      - track_id 在该 scenario 内唯一且状态字段完整。

  4. 满足条件的所有 car / truck / bus 都进入 tracks_to_predict。
  5. 不设置每个 scenario 的 target 数上限；若后续出于训练效率需要限额，应作为单独配置项引入，而不是当前默认行为。
  6. 若一个 window 中没有任何满足条件的 target，则该 window 不生成 scenario。

  ## 6. 时间窗口与采样规范

  ### 6.1 采样率

  采用 SinD 原始近似 10Hz 时间分辨率，不额外重采样为其他频率。

  ### 6.2 窗口长度

  与 UniTraj 默认训练目标对齐：

  - past_len = 21
  - future_len = 60
  - total_length = 81

  ### 6.3 窗口步长

  规范默认 stride 设为可配置，推荐默认值：

  - stride = 40

  要求：

  - stride 必须是配置项
  - 文档与代码都必须允许后续切换为 20、80 等其他值
  - split 和 cache 构建不得依赖固定 stride

  ## 7. 轨迹与 metadata 契约

  ### 7.1 track state 必填字段

  每个 track 的 state 必须完整输出：

  - position
  - length
  - width
  - height
  - heading
  - velocity
  - valid

  ### 7.2 canonical heading

  正式采用：

  - yaw_rad 作为 canonical heading
  - heading_rad 仅作为 fallback 或辅助分析字段

  ### 7.3 行人默认值

  由于行人无尺寸与朝向，context-only 策略下仍需补齐最小状态：

  - length = 0.5
  - width = 0.5
  - height = 1.7
  - heading = atan2(vy, vx)；低速时允许回退到 0

  ### 7.4 metadata 必填字段

  实现必须稳定输出：

  - scenario_id
  - dataset
  - ts
  - sdc_id
  - tracks_to_predict

  建议同时输出：

  - track_length
  - coordinate
  - city
  - record_name
  - split_candidates 或其他便于后续 split 管理的字段

  ## 8. 地图规范

  ### 8.1 总体原则

  地图输出目标是：

  - robust core + best-effort rich map

  即：

  - 核心静态地图信息必须稳定输出
  - richer map 语义尽量多提取
  - 单城标签不统一或无法稳定恢复时必须允许降级，不能阻塞全流程

  ### 8.2 v1 强制地图范围

  实现必须稳定支持：

  - lane-like feature
  - boundary / road line / road edge
  - stop line，若城市 OSM 中存在且可稳定识别

  ### 8.3 v1 best-effort 地图范围

  实现尽量支持，但不作为阻塞项：

  - crosswalk
  - basic topology
      - predecessor
      - successor
      - 其他能稳定恢复的 lane connectivity

  - 其他可映射的静态语义

  ### 8.4 地图审计先行

  在正式实现 richer map 前，必须先完成四城 OSM tag inventory，并形成稳定映射表，内容至少包括：

  - OSM tag pattern
  - 出现城市
  - 可映射 ScenarioNet feature type
  - 映射置信度
  - v1 策略
  - 降级策略
  - OSM tag audit 必须产出 stable mapping table；只有被确认稳定的 tag 才能进入 v1 的 mandatory map scope。

  ### 8.5 不允许的实现方式

  - 不得把拓扑按遍历顺序硬连
  - 不得把 crosswalk 标签简单假定为四城统一
  - 不得因某城无 crosswalk/topology 而使整个 record 转换失败

  ## 9. 交通灯规范

  ### 9.1 基本原则

  交通灯文件 schema 在四城间不统一，因此灯态解析必须独立于 raw tracks 兼容层。

  ### 9.2 v1 要求

  dynamic_map_states key 必须存在。

  允许：

  - 部分城市输出可靠灯态
  - 对无法稳定映射 lane-light 关系的城市先输出降级版本
  - 在必要时输出 {}，但不得作为长期默认路线

  ### 9.3 v2 扩展方向

  后续可逐步增强：

  - light id -> lane id
  - stop point
  - lane state over time
  - pedestrian signal 与 vehicle signal 分离语义

  ## 10. Split 规范

  ### 10.1 正式支持两种 split 模式

  #### A. Record-level split

  正式支持 train / test

  要求：

  - 以 record 为最小切分单位
  - 同一 record 的所有 scenario 不得跨 split
  - 默认用于常规训练与验证
  - 比例为train=80%, test=20%

  #### B. City-level holdout

  正式支持跨城市泛化实验

  要求：

  - 以城市为 held-out 单位
  - 至少支持：
      - leave-one-city-out
      - 指定训练城市集合与测试城市集合

  ### 10.2 Split 实现要求

  1. split assignment 必须独立保存。
  2. 同一 canonical scenario 库必须可重复派生：
      - 默认 record-level split
  3. split 配置至少应记录：
      - split 模式
      - seed
      - train/test 的 record 或 city 列表
      - 生成时间和配置版本

  ## 11. Cache 构建与下游兼容

  ### 11.1 cache 目标

  最终 cache 必须兼容 UniTraj 当前标准输入路径，输出：

  - .h5
  - file_list.pkl

  ### 11.2 模型兼容目标

  必须支持以下三个模型直接训练与评估：

  - MTR
  - AutoBot
  - Wayformer

  ### 11.3 下游最关键约束

  ScenarioNet 输出必须满足 BaseDataset 的实际读取契约，确保后续能正确生成：

  - obj_trajs
  - obj_trajs_mask
  - map_polylines
  - map_polylines_mask

  ## 12. 验收与测试规范

  ### 12.1 Raw 兼容测试

  至少验证：

  - 四城 record 发现数量正确
  - 各城市文件名差异可被统一解析
  - 缺失 meta 的 record 可从 tracks 推断必要信息
  - Tianjin 地图回退规则生效

  ### 12.2 ScenarioNet 合法性测试

  至少验证：

  - 每个 scenario 顶层 key 完整
  - tracks / metadata / map_features / dynamic_map_states 结构合法
  - scenario_id 唯一且稳定
  - tracks_to_predict 非空且只包含 car/truck/bus

  ### 12.3 地图测试

  至少验证：

  - 四城都能稳定输出 lane-like features
  - boundary/road line 可稳定输出
  - stop_line 在存在的城市中可输出
  - crosswalk/topology 缺失时触发降级而非失败

  ### 12.4 Split 测试

  至少验证：

  - record-level split 无 record 泄漏
  - city-level split 无 city 泄漏
  - split 重建结果稳定可复现

  ### 12.5 Cache 测试

  至少验证：

  - train/val/test cache 可构建
  - MTR、AutoBot、Wayformer 均可完成至少一次 dataset load
  - 至少一个小批次可跑通前向或训练启动 smoke test

  ## 13. 默认假设与固定决策

  本规范固定以下默认决策：

  - 使用 SinD/Dataset 作为完整原始数据主输入
  - 使用 SinD/Data 作为 Tianjin 地图回退根目录
  - 保留 ScenarioNet 作为正式中间产物
  - 采用三段式 CLI：convert-scenarios / make-splits / build-cache
  - 采用 shared multi-target scenarios
  - 仅 car/truck/bus 进入 target 集合
  - motorcycle/bicycle/tricycle/pedestrian 均保留为 context
  - 默认使用 record-level train/val/test 与 city-level holdout 两类 split
  - 地图采用 robust core + best-effort rich map
  - richer map 的 mandatory set 必须先经过四城 OSM tag audit 才能冻结
  - 所有路径均参数化，本地与集群只通过配置切换

  ## 14. 对现有文档与 MVP 的关系

  本规范基于以下内容收敛而成：

  - Summary/ 中关于 SinD、ScenarioNet、UniTraj 的研究结论
  - 现有 MVP UniTraj/unitraj/utils/sind_converter.py 暴露出的可运行路径与结构问题

  本规范替代其“研究建议”角色中的实现边界定义，后续实现应以本规范为准，而不是继续扩展 sind_converter.py 的单文件结构。

  ## 测试计划

  - 构造四城最小 smoke subset，验证三段式 CLI 端到端可跑通
  - 对至少一个完整城市验证 canonical scenario 数量、map feature 数量、target 数量统计
  - 对 record-level 与 city-level 各生成一套 split，并分别构建 cache
  - 对 MTR、AutoBot、Wayformer 各做一次 dataset load 与小批次前向 smoke test
  - 对 richer map 审计表中的每一类 mandatory tag 做至少一个城市级示例验证

  ## 假设

  - 现有 UniTraj 版本继续以 ScenarioNet-style 数据集为主输入
  - 当前目标是稳定支持四城全量转换与训练，不以一次性做到最完整交通灯和地图语义为前提
  - 远端集群只改变配置路径，不改变数据 schema 和 pipeline 逻辑
