# Scene geometry：正式 top1 two-stage 评测

## 一键执行

在服务器终端执行（不要通过有单文件大小限制的命令桥接启动完整任务）：

```bash
cd /mnt/workspace/roa7sgh/DrivoR
unset PY PYTHON_BIN
bash scripts/nav2_eval_scene_geometry_all.sh
```

脚本默认 nohup 后台运行，打印 PID、日志路径和查看日志命令。先检查场景级 metric cache，完整则复用，缺失则调用官方缓存入口补齐；然后根据训练配置导出推理模块，加载 checkpoint，执行正式 top1 two-stage 评测。

默认 Python 为 `/usr/local/bin/python3`，自动配置数据、地图、模块路径。默认在 `exp/ke/drivor_sg_gen` 中选择最近修改的训练运行，并使用其中唯一的 `best*.ckpt`；有多个 best 时要求指定文件，不会自动混用。新训练出现后默认选择会改变，复现实验时请显式设置 checkpoint。

## 分开执行

```bash
# 只检查／补齐 two-stage 场景级 metric cache，默认 72 worker。
bash scripts/nav2_cache_two_stage.sh

# 已有完整缓存时直接评测；缓存不完整会报错。
bash scripts/nav2_eval_scene_geometry.sh

# 展示命令，不生成缓存、不导出模型、不启动评测。
DRY_RUN=1 bash scripts/nav2_eval_scene_geometry_all.sh

# 检查场景覆盖、实际读取一个 metric cache、导出模型并严格加载 checkpoint。
# 不启动完整推理和打分。此检查会更新独立的推理导出模块。
CHECK_ONLY=1 bash scripts/nav2_eval_scene_geometry.sh
```

两个分开执行的脚本都默认后台运行：应等缓存脚本成功结束，再启动评测。需要自动等待并串联时使用 `nav2_eval_scene_geometry_all.sh`。设置 `BACKGROUND=0` 可前台执行。

## 默认路径和可选设置

| 环境变量 | 默认值／用途 |
|---|---|
| `EVAL_REPO_ROOT` | `/mnt/workspace/roa7sgh/navsim-v2` |
| `DATA_ROOT` | `/mnt/workspace/hru4sgh/NAVSIM/dataset` |
| `METRIC_CACHE_PATH` | 评测仓库 `exp/navhard_two_stage_metric_cache` |
| `CHECKPOINT_PATH` | 显式指定 checkpoint；含 `=` 的路径由脚本正确转义 |
| `TRAIN_CONFIG` | 对应训练 `code/hydra/config.yaml`；通常从 checkpoint 的父目录自动找到 |
| `TRAIN_RUN_DIR` | 指定训练运行，仅在没有显式 checkpoint 时用于选择 best |
| `GPU_IDS` | `0,1,2,3` |
| `GPU_BATCH_SIZE` | 每卡 `16` |
| `GPU_NUM_WORKERS` | 每卡数据读取 worker `8` |
| `CPU_WORKERS` | 缓存／正式打分 worker `72` |
| `MAX_SCENARIOS_PER_TASK` | 每个正式打分任务 `32` 个场景 |
| `OUTPUT_DIR` | 默认生成唯一的新目录；拒绝已有目录 |
| `LOG_FILE` | 默认评测仓库 `exp/logs/scene_geometry_<阶段>_<时间>_<PID>.log` |

例如固定本次训练的 best checkpoint：

```bash
CHECKPOINT_PATH='/mnt/workspace/roa7sgh/DrivoR/exp/ke/drivor_sg_gen/10.14_18.53/lightning_logs/version_0/checkpoints/best-epoch=8-step=11961.ckpt' \
  bash scripts/nav2_eval_scene_geometry_all.sh
```

## 缓存含义与结果

- `DrivoR/dataset/scene_geometry_v1` 是训练几何监督缓存，评测时不读取。
- `DrivoR/dataset/train_metric_cache_navtrain` 是训练场景的 metric cache，不能替代 navhard two-stage 缓存。
- `navsim-v2/exp/navhard_two_stage_metric_cache` 是模型无关的评测场景缓存，可跨 checkpoint 复用。脚本检查当前 split 的所有 token 均被索引且文件存在，并实际读取一个文件检测基本序列化兼容性；不是逐文件解压校验。缺失时采用 `force_feature_computation=false` 补齐，不强制重算已有文件。
- 正式 top1 入口直接推理模型选中的轨迹，再执行官方模拟和 two-stage 汇总，无需预先生成 64 候选 oracle 缓存。

输出默认在 `navsim-v2/exp/drivor_sg_gen_top1_two_stage/<时间>/`（aux/off 使用对应 mode）。包含官方 CSV、Hydra 配置和 `pipeline_manifest.json`。manifest 记录 checkpoint SHA256、导出代码 hash、命令、场景覆盖检查及最终状态。脚本检查 CSV 全场景覆盖、valid、有限分数及 stage one／stage two／combined 三个汇总行；发现失败时不会报告完成。

不同 pipeline 通过锁串行运行，避免共用导出模块在评测中被另一个脚本替换。不要在运行中手动重新导出其他模型。当前实现只做正式 top1，不包含此前 B2 的 GT 轨迹选择、oracle 分数和 64 候选全零率；这些指标不能直接与本次 top1 结果混作同一指标比较。
