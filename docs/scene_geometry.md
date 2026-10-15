# 当前帧几何监督实验

`navsim/agents/drivoR/scene_geometry/` 中的实现参考了 SparseDriveV2 的独立检测／地图 queries、几何位置编码、迭代细化和匈牙利匹配监督，并适配 DrivoR 的 scene tokens。本次未移植 SparseDriveV2 的可变形图像采样、时序实例缓存或运动预测／规划头。

## 对照实验

| 模式 | 当前几何监督 | 生成器新增注意力输入 | scorer 的直接输入 |
|---|---|---|---|
| `off`（SG-BASE） | 无 | 无 | 原有 scene tokens |
| `aux`（SG-AUX） | 有 | 无 | 原有 scene tokens |
| `gen`（SG-GEN） | 有 | Agent／地图 tokens | 原有 scene tokens |

三组均保留最新时刻的四路相机图像、自车状态、DINOv2 + LoRA、动态生成的 64 条轨迹（每条 8 个 waypoint）、原有轨迹损失和 scorer。本次不加入其他交通参与者的速度／未来预测、occupancy 预测头、path-speed 词表或 B2 安全损失。共享场景特征仍可能间接改变 scorer，因此应分别评价候选轨迹的可行覆盖和轨迹选择效果。

默认配置使用 128 个 agent queries、100 个 map queries，每条地图折线包含 20 个点，两层细化模块（特征维度 256、4 个注意力头、FFN 维度 1024）。Agent 输出为归一化 XY、长宽的对数，以及航向角的 sin/cos；地图输出为归一化 XY 折线。Agent 的四个类别和地图的两个类别均采用 sigmoid 分类，所有类别均为负表示背景。生成器读取全部预测 queries，并额外加入一个空 token。置信度经过停止梯度、下限 0.05 截断和取对数后，作为注意力偏置。输入生成器位置编码 MLP 的预测几何经过停止梯度处理，语义 tokens 保留梯度。每个轨迹解码块在原有场景交叉注意力与 FFN 之间加入 agent／地图注意力，残差门控初始值为零。原有 scorer 不直接读取这些新增特征。

原有模块优先构建，新增模块在独立的随机数状态上下文中初始化，保持各组共有参数的初始权重和外部初始化随机数状态一致。`off` 不创建额外参数，仍支持严格加载旧 checkpoint。将旧规划 checkpoint 严格加载到 `gen` 会失败，除非显式选择现有的“仅初始化感知部分”功能。本对照实验仅从 DINO 预训练权重开始初始化。

## 标签与缓存

仅使用 `scene.frames[num_history_frames - 1]`。NAVSIM 的 `gt_boxes_oriented_box()` 直接通过 `ego_state.rear_axle` 将标注 XY 转换到全局坐标，因此本标签使用相同的自车后轴坐标系（x 向前、y 向左）。不读取其他交通参与者的速度或未来帧。

ROI 为 `[-32, -32, 64, 32]` 米，顺序为 `[xmin, ymin, xmax, ymax]`。Agent 类别包括车辆、行人、自行车和静态障碍物。静态障碍物合并了 barrier、traffic cone、construction sign 和 generic object；方向不明确的静态物体不计算航向监督。未知类别会计数并忽略。有效对象按距离排序，以原始索引作为稳定的次级排序依据，超出容量的目标被截断，并记录溢出数量。

道路边界从 ROADBLOCK、INTERSECTION、DRIVABLE_AREA 和 CARPARK_AREA 多边形的并集中提取，保留内部孔洞，并且**先提取边界，再按 ROI 裁剪**，避免产生虚假的裁剪框边界。这版 nuPlan 没有 DRIVABLE_AREA 对象查询接口，因此通过其带空间索引的矢量图层读取道路段、交叉口、一般可行驶区域和停车场。车道及车道连接段的基准线构成有方向的中心线。折线经过裁剪后，切分为最长 20 米的片段，按弧长重采样、去重并进行确定性排序。边界匹配允许点序反转；中心线匹配保留方向。这些标签描述地图几何，不提供导航路线信息。

独立缓存位于：

```
dataset/scene_geometry_v1/<spec_hash>/<log>/<token>.gz
```

哈希包含标签格式、几何标签生成源码哈希、ROI、容量、类别、采样规则及数据版本。每次运行的清单记录所选数据划分中的 tokens 和统计信息。原有 `drivor_target.gz` 保持不变。`strict` 模式在缓存缺失或版本不匹配时直接报错；`on_the_fly` 用于小规模试验，按需计算标签。使用已有特征缓存的数据集也会从独立几何缓存目录读取标签，并检查文件是否缺失。地图缺失时标记为不可用，不参与监督；地图 API 读取异常会直接抛出，避免错误地产生“没有道路”的负样本。

## 损失与监控

Agent 和地图分支分别使用 FP32 匈牙利匹配。Focal 分类损失采用 alpha 0.25、gamma 2；agent 的归一化 XY、对数尺寸和航向损失权重分别为 2、0.5、0.5。地图归一化点坐标的 L1 损失权重为 10，中心线方向一致性损失权重为 1。中间层／最终层权重为 0.5／1，按权重之和归一化。分布式训练使用各进程正样本数的全局均值作为归一化基数。空场景仍训练背景分类；不可用的地图分支贡献可微的零损失。

几何总损失权重在实际优化器更新步数的前 10% 中从 0 线性增加到 1，依据 Lightning 的 `global_step` 和 `estimated_stepping_batches` 计算，恢复训练时同样适用。训练和 DrivoR 的自定义验证流程均记录几何损失、匹配后的归一化 L1、匹配类别准确率、匹配比例、标签可用率及溢出数量。匹配类别准确率是诊断指标，**不能当作检测 AP 或召回率**。

## 运行方法

已提供包含环境配置的后台缓存启动脚本，默认使用 72 个 worker，并通过 `nohup` 运行：

```bash
cd /mnt/workspace/roa7sgh/DrivoR
bash scripts/nav2_cache_scene_geometry.sh
```

脚本默认使用 `/usr/local/bin/python3`，无需手动激活 conda 环境。它自动配置 Python 模块搜索路径、数据／地图目录及每进程线程数，并在 `exp/logs/` 中生成带时间戳的日志和 PID 文件。启动后会输出日志位置和 `tail -f` 命令。

```bash
# 只检查环境路径和启动命令，不执行缓存。
DRY_RUN=1 bash scripts/nav2_cache_scene_geometry.sh

# 自定义并行数或缓存目录。
CACHE_WORKERS=72 CACHE_WORKER_THREADS=1 \
  GEOMETRY_CACHE_ROOT=/mnt/workspace/roa7sgh/DrivoR/dataset/scene_geometry_v1 \
  bash scripts/nav2_cache_scene_geometry.sh

# 小样本前台运行；额外参数会转发给 Hydra。
BACKGROUND=0 CACHE_WORKERS=2 bash scripts/nav2_cache_scene_geometry.sh \
  ++train_test_split.scene_filter.max_scenes=2
```

还可通过 `PYTHON_BIN`、`DATA_ROOT`、`NUPLAN_MAPS_ROOT`、`LOG_FILE` 和 `PID_FILE` 覆盖默认设置。以下保留手动配置和执行方式，便于核对环境或分别启动训练。

训练与缓存脚本统一使用已验证的 `/usr/local/bin/python3`（NumPy 2.5.1、Torch 2.10.0+cu128），支持 H20，并能读取现有 metric 和 geometry 缓存。`nav-v2` 虽然支持 H20，但其 NumPy 1.26.4 在读取当前 metric cache 时会报 `No module named 'numpy._core.numeric'`。这里的系统默认环境不是 Conda base；后者缺少训练依赖。此次仅切换启动脚本的默认解释器，没有修改安装包或缓存内容。已有 geometry 缓存无需重建。

```bash
cd /mnt/workspace/roa7sgh/DrivoR
export PYTHONPATH="$PWD:$PWD/nuplan-devkit"
export NAVSIM_DEVKIT_ROOT="$PWD" NAVSIM_EXP_ROOT="$PWD/exp"
export OPENSCENE_DATA_ROOT=/mnt/workspace/hru4sgh/NAVSIM/dataset
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
PY=/usr/local/bin/python3

# 按启动脚本使用的 navtrain 训练／验证划分生成标签缓存，默认 72 个 worker。
"$PY" navsim/planning/script/run_scene_geometry_caching.py \
  agent=drivoR_scene_geometry train_test_split=navtrain

# 只查看启动命令，不启动训练。
DRY_RUN=1 GEOMETRY_MODE=gen bash scripts/nav2_train_scene_geometry.sh

# 完成缓存和小规模验证后，分别启动各组实验。
GEOMETRY_MODE=off bash scripts/nav2_train_scene_geometry.sh
GEOMETRY_MODE=aux bash scripts/nav2_train_scene_geometry.sh
GEOMETRY_MODE=gen bash scripts/nav2_train_scene_geometry.sh
```

启动脚本沿用 B0 的随机种子 2、10 个 epoch、4 张 GPU × 每卡 batch 16、基础学习率 0.0002 及初始化方式。可显式设置 `GEOMETRY_CACHE_MODE=on_the_fly` 进行小规模试验，额外 Hydra 参数也会转发。各组共有配置应保持一致，AUX 和 GEN 的几何分支维度也应相同。比较结果前应约定一致的 epoch／checkpoint 选择规则；现有按 top1 分数选最优 checkpoint 的回调，不能替代相同条件下的 oracle 覆盖率比较。

请从远程终端启动完整训练。当前文件／命令桥接限制单个文件最多写入 16 MiB，无法保存完整训练 checkpoint。冒烟测试的临时权重采用小型张量分片以满足此限制；正式 checkpoint 的加载方式保持不变。

### 缓存并行设置

缓存脚本默认使用 **72 个 worker 进程**，每个 worker 的 Torch／OpenMP／BLAS 线程数默认限制为 **1**。程序先按原有规则确定 train／val 的场景集合，再以 log 为单位提交到进程池；每个 worker 只加载当前任务的 log，并复用该进程中的地图资源。主进程的场景筛选阶段仍为串行，标签生成阶段并行。若选中的 log 任务少于 72 个，实际进程数会相应减少。

可通过以下参数调整，使用 `cache_num_workers=1` 即可串行运行：

```bash
"$PY" navsim/planning/script/run_scene_geometry_caching.py \
  agent=drivoR_scene_geometry train_test_split=navtrain \
  agent.config.scene_geometry.cache_num_workers=72 \
  agent.config.scene_geometry.cache_worker_threads=1
```

这些参数只控制几何标签缓存，与训练的 `batch_size` 和 `dataloader.params.num_workers` 无关，也不改变标签内容、缓存版本哈希或保存目录。已存在且有效的标签会直接读取；失败任务会报告对应的 split／log／token，全部任务成功后才写入本次完整清单。增加 worker 会提高内存和存储并发需求，不保证耗时按进程数同比下降。

## 导出推理代码

```bash
"$PY" scripts/export_scene_geometry.py \
  --destination /mnt/workspace/roa7sgh/navsim-v2 --mode gen
```

该命令仅在评测仓库新增 `navsim/agents/drivor_scene_geometry/` 和 `agent=drivoR_scene_geometry` 配置。导出内容包括模型、图像编码器、特征构建器和严格加载权重的推理 agent，不包含几何 GT 标签构建器、几何损失或地图标签提取逻辑。原有评测 agent 和正式 PDM 代码保持不变。导出清单用于防止后续导出覆盖独立修改。如果训练结构与默认配置不同，传入 `--training-config <run>/code/hydra/config.yaml`；AUX checkpoint 使用 `--mode aux`。在现有评测入口中传入 `agent.checkpoint_path=<new_checkpoint>` 指定新权重。

不要将整个训练仓库放在 `PYTHONPATH` 中评测仓库之前。每组“checkpoint + 导出代码哈希 + 完整解析后的评测配置”应使用新的实验／缓存目录，避免复用旧 proposals 缓存。导出配置统一将 NC／DAC／DDC／TTC／EP／comfort 排序权重设为 10／13／6／14／15／2，以保证 top1 选择结果可比较。Oracle PDM 仍由正式评测器计算。

## 验证与实验验收

```bash
"$PY" -m pytest tests/agents/test_scene_geometry.py tests/agents/test_checkpoint_initialization.py -q
"$PY" -m torch.distributed.run --standalone --nproc_per_node=2 tests/agents/scene_geometry_ddp_smoke.py
"$PY" scripts/smoke_scene_geometry.py --data-root "$OPENSCENE_DATA_ROOT" \
  --output /tmp/drivor_scene_geometry_pilot --samples 3
"$PY" scripts/smoke_scene_geometry_model.py --labels /tmp/drivor_scene_geometry_pilot \
  --output /tmp/drivor_scene_geometry_model_smoke
```

模型冒烟测试使用 224×112 图像、真实 DINO／LoRA 权重和真实当前帧标签。生成的 checkpoint 仅是一次性的未训练测试数据，不代表实验结果。单元测试覆盖：零门控输出一致性、共有参数初始化一致性、严格 checkpoint 加载、损失掩码与空目标集合、坐标旋转、地图孔洞／裁剪边界／方向、梯度传递、优化器参数覆盖、损失权重预热，以及独立缓存。双进程 CPU 测试在关闭未使用参数检测的条件下，验证各进程正样本数不同以及地图缺失的情况。

跨仓库一致性检查应从**评测仓库**运行 `scripts/verify_scene_geometry_export.py --fixture <model-smoke-output>`，并仅将评测仓库自身及其 nuplan-devkit 放入 `PYTHONPATH`。脚本读取全部真实张量分片，将组装后的权重交给导出 agent 的严格初始化／键名映射逻辑，再比较 proposals、评分和最终选择的轨迹。测试中替换读取函数仅用于适配分片测试数据，不修改推理 agent。

正式解释实验结果前，应在更大且有代表性的训练样本上检查标签质量和容量溢出，不能只依据三个相邻帧。使用官方模拟器评测全部 64 条 proposals，报告全零率（不加权及 GT 加权）、oracle 分数、修复及新增的全零场景、道路／碰撞失败分组，以及原始轨迹与模拟执行轨迹的 DAC 差异。模型选中的 top1 指标单独报告。仅通过实现检查，尚不能证明规划性能提升。
