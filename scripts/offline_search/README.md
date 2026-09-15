## Updated final selection policy

Only scenes satisfying (GT speed drop > 1 m/s AND relative drop > 10%) OR
relative drop > 20% from 0 to 4 seconds enable this gate. All comparisons are strict.
Relative drop uses GT initial speed; zero initial speed does not trigger the gate. Constant-speed, accelerating, and smaller net-drop scenes skip it.
The decision uses GT only, not candidate speed or inferred leading actors. Diagnostics
record enabled, initial/final GT speeds, drop, and threshold. Enabled candidates
must satisfy terminal speed <= original four-second GT
rollout terminal speed + min(0.5 m/s, 10% of that GT speed); numerical tolerance
is 1e-8 m/s. Existing path coverage and heading gates remain in place.
For GT PDMS >= 0.95, any strictly higher PDMS qualifies after constraints pass.
Below 0.95, the configured minimum gain (default 0.005) still applies.
If diversity filtering would select no representative, keep the best improving
archive member. At most four representatives are kept; do not pad with duplicates.
Reports identify actual selected IDs as final retained trajectories. Training
export remains disabled. Existing result files require rerunning to apply gates.

## Scene-level CPU parallelism

`run` and `demo` accept `--workers` (default 1) and `--worker-threads` (default 1).
The server wrapper exposes these as `SEARCH_WORKERS` and `WORKER_THREADS`:

```bash
export SEARCH_WORKERS=4
export WORKER_THREADS=1
export RESULT_ROOT=/mnt/workspace/roa7sgh/DrivoR/exp/offline_search/nav1_engineering_parallel
bash scripts/offline_search/nav1_engineering.sh full
```

This starts up to four independent scene processes. Actual workers are capped by
scene count, so the one-scene smoke run still uses one worker. Start with four
workers, then compare wall time and memory before increasing the count. Each
process loads its own scene, map, simulator and scorer; memory grows with workers.
Candidate/island search remains sequential within each scene and its seed is unchanged.

All runs, including one worker, use fresh `spawn` processes. Native BLAS/OpenMP
thread environment variables are set before child imports to avoid nested thread
oversubscription; this is not CPU affinity or a guarantee for libraries that do
not honor those variables. `execution.json` records requested/effective workers
and thread settings; scene reports record worker PID and inherited settings.
The parent alone writes root summaries in manifest order; individual scene
failures are recorded and cause exit code 2. Existing output directories are
never overwritten. There is no resume or single-scene parallel search yet.

The generic Bash/PowerShell wrappers forward the same CLI flags. For a synthetic
parallel correctness check without NAVSIM data:

```bash
bash scripts/offline_search/run_engineering.sh demo --scene-count 2 --workers 2 --worker-threads 1 --output /tmp/drivor_parallel_demo
```

Terminal heading gate: final executed heading error relative to the cached route centerline must not exceed spatially aligned 5-second GT rollout heading error plus 1 degree. Invalid/non-simple route, reversed GT progress, or unavailable GT coverage is unverifiable. Report endpoint rays: orange dashed = route tangent, green = candidate heading. No lateral-return constraint is added.

GT speed reference now uses a separate 5-second GT rollout from the available 10 future poses; candidate formal scoring remains 4 seconds. No endpoint extrapolation is used.

## Current experimental policy (2026-09-14)

Missing actor intervals are diagnostic only; observed intervals and endpoints still undergo collision checks. This supersedes the earlier reachable-region gate. Export remains disabled.

Known map limits remain mandatory. Unknown limits inside map INTERSECTION polygons use spatial GT rollout speed plus 1 m/s; outside intersections no extra speed ceiling applies. Ambiguous or out-of-range GT projection is rejected. Other gates remain unchanged. Use a fresh RESULT_ROOT to rerun smoke.

# V1 离线搜索工程验证

## 首轮 full_results 后的修复（报告 schema_version=2）

首轮14场景被生命周期整场拦截、2场景被2ms时间阈值拦截，没有进入CEM。此版本修复如下：

- **部分track保留**：缺失pose用NaN和显式mask保存，已知区间正常碰撞检查，不删除track、不补静止轨迹。
- **未知区间按候选核验**：使用最近的已知前/后位置，构造覆盖整个未知区间的中心可达圆盘及车体外接圆；有两个锚点时取区域交集。与ego连续包络无法证明分离的候选仍为`actor_lifecycle_unverifiable`。远处不相交的部分track不再自动让整个场景跳过搜索。
- **条件性范围必须明确**：缺失期间中心速度上界默认55m/s；如果已知中心位移已超过该值，则上调到观测值的1.01倍。这是保守工程假设，不是从缺失数据证明出的实际速度上界。诊断逐actor记录实际采用值；该假设只支持工程筛选，额外加入`missing_actor_motion_bound_validation`待验收项，全部输出仍`export_certified=false`。不能将“范围外排除”解释为现实无碰撞证明。
- **时间口径**：评分继续按官方0.5秒名义frame时钟和0.1秒rollout；原始时间戳、逐帧间隔、最大抖动完整记录。毫秒抖动不再按2ms整场拒绝；乱序/重复时间戳、相对名义时刻或相邻间隔偏离达到半帧(0.25s)仍报错并保存诊断。初始ego/cache的位姿和时间一致性检查保留。
- **限速**：记录匹配lane/connector、route关系和限速来源；未知值仍未知，已知超速仍失败。一个时刻限速缺失不再掩盖其他时刻的超速。优先使用覆盖当前中心的on-route lane，多个相关lane取最小已知值；相关lane中存在未知值则仍不可核验，不填默认限速。
- **状态**：明确区分`gt_upper_bound`、`precheck_blocked`、`search_not_started`、`no_verifiable_candidate_in_budget`、`no_feasible_candidate_in_budget`、`no_nominal_improvement_in_budget`和`nominal_improvement_found`。汇总直接记录是否启动、唯一候选/控制评估数及各代可核验数量。
- **新增证据**：每场景`diagnostics.json`与`actor_replay.npz`包含生命周期、时间、限速和完整部分actor回放。HTML对未知时刻输出null，不画虚构物体；页面显示CEM状态及候选数。

**复用原16场景和正式缓存，不重新prepare。** 同步更新代码到服务器后：

```bash
cd /mnt/workspace/roa7sgh/DrivoR
export WORK_ROOT=/mnt/workspace/roa7sgh/DrivoR/exp/offline_search/nav1_engineering
export RESULT_ROOT=/mnt/workspace/roa7sgh/DrivoR/exp/offline_search/nav1_engineering_fix1
bash scripts/offline_search/nav1_engineering.sh verify
bash scripts/offline_search/nav1_engineering.sh smoke
# 先检查候选数>0、运行无错误以及diagnostics，再跑：
bash scripts/offline_search/nav1_engineering.sh full
```

新报告为`$RESULT_ROOT/smoke_results/index.html`和`$RESULT_ROOT/full_results/index.html`。两个结果目录须不存在。修复不会保证每场景存在可核验候选：如果未知物体范围确实可能影响所有候选或地图限速缺失，仍会明确报告对应失败。

当前代码固定真实初始 ego、导航与图像，搜索未来控制残差。无模型训练，不要求 GPU；沿用已有 DrivoR Linux 环境与 metric cache。新增代码在 `navsim/offline_search/`，没有修改官方评分器或网络。

完整设计与实施边界见 [离线搜索方案 v2.2](../../docs/offline_search_plan.md)。

## 使用现有 NAV1 服务器路径的一键入口

已按 `scripts/nav1_train_frozen_backbones.sh` 和 `scripts/nav1_eval.sh` 配置：仓库 `/mnt/workspace/roa7sgh/DrivoR`，数据 `/mnt/workspace/hru4sgh/NAVSIM/dataset`，Python `python3`，地图版本 `nuplan-maps-v1.0`。先将本次新增的代码同步到服务器，并激活原训练环境：

```bash
cd /mnt/workspace/roa7sgh/DrivoR
bash scripts/offline_search/nav1_engineering.sh verify
bash scripts/offline_search/nav1_engineering.sh prepare
bash scripts/offline_search/nav1_engineering.sh smoke
# 检查单场景结果后再执行：
bash scripts/offline_search/nav1_engineering.sh full
```

`prepare` 从 navtrain 配置的 token/log 中选择最多16个场景，并排除 `default_train_val_test_log_split.yaml` 的 val_logs。使用正式 `MetricCacheProcessor` 为选中场景逐个生成缓存，随后写出单场景和完整工程 manifest。缓存失败保留并返回非零，不换场景掩盖失败。

**不能直接使用训练脚本里的 `dataset/train_metric_cache_navtrain`：其代码使用 `train_metric_chache.MetricCache`，没有正式 reference trajectory，centerline 类型也不同。** 评测脚本的 `metric_cache_navtest` 属于测试集，同样不用于这次搜索调参。

默认所有输出在 `/mnt/workspace/roa7sgh/DrivoR/exp/offline_search/nav1_engineering/`：

- `official_metric_cache/`：新增的正式缓存；原有缓存不变。
- `preparation_summary.json`、`engineering_manifest.jsonl`、`one_scene_manifest.jsonl`。
- `smoke_results/index.html`：单场景4候选/岛×1代。
- `full_results/index.html`：最多16场景32候选/岛×5代。

`prepare` 要求工作目录不存在；`smoke/full` 要求各自结果目录不存在。要启动另一次实验，在执行这组命令前统一 `export WORK_ROOT=/mnt/workspace/roa7sgh/DrivoR/exp/offline_search/nav1_engineering_run2`，不要分别使用不同 WORK_ROOT。无需 checkpoint、图像 backbone 或 GPU。

这组真实数据命令尚未在服务器执行。本机只做脚本语法、数据选择逻辑和已有核心测试；实际地图/cache适配仍需通过 `verify` 和 `smoke`。

**这是工程验证版本，不是完整 teacher 生产管线。** 实现了正式评分适配、控制重放、八岛初始化/CEM、部分保守名义检查、工程聚类及可视化。真实 16 场景尚需在有数据的机器执行。所有输出固定 `export_certified=false`，不会生成 `selected_targets.npz`。找到名义改进不能当作严格安全覆盖率。

## 1. 环境和一次验证

在已有 DrivoR 环境运行，至少需要已有项目依赖，以及 `scipy`、`shapely>=2`、`pytest`。不需要为了本实验重装全部 requirements。

```bash
cd /path/to/DrivoR
bash scripts/offline_search/verify_engineering.sh /path/to/new_output/synthetic_smoke
```

这会运行几何/搜索测试，以及一个使用**真实官方 LQR、动力学和评分器、合成场景**的集成测试，然后生成交互报告演示。DEMO 本身使用简化动力学和代理分数，不能解读为 NAVSIM 成绩。

Windows 可运行核心测试和 DEMO；vendored nuPlan map 模块依赖 `fcntl`，真实 NAVSIM 与官方集成测试请在 Linux 环境运行。Windows 下官方集成测试会明确 skip。

本次开发主机的组策略阻止启动 `.ps1` 文件，因此 PowerShell wrapper 未完成启动验证。不要修改执行策略来运行它；主要交付入口是 Linux Bash 脚本和 Python 模块。

```powershell
cd C:\Users\ROA7SGH\Downloads\code\DrivoR
python -m pytest tests/offline_search -q
.\scripts\offline_search\run_engineering.ps1 demo --output C:\path\to\new_demo_output
```

可指定 Python：PowerShell 使用 `-Python C:\path\python.exe`；Bash 设置 `PYTHON_BIN=/path/python`。

## 2. 准备工程样本清单

输入必需：原始 NAVSIM log pickle、正式 V1 metric cache、nuPlan 地图、**允许使用的训练 token 清单**。token 必须是字符串，可使用文本每行一个、JSON 字符串列表或 YAML `tokens: [...]`。不要使用 NAV1-test 或学生模型的验证集作为搜索调参数据。

```bash
bash scripts/offline_search/run_engineering.sh manifest \
  --scene-root /data/openscene/navsim_logs/trainval \
  --cache-root /data/metric_cache \
  --train-tokens /data/splits/search_train_tokens.txt \
  --exclude-tokens /data/splits/model_validation_tokens.txt \
  --count 16 \
  --output /data/offline_search/engineering_manifest.jsonl
```

`--exclude-tokens` 可重复传入；一旦与训练 allowlist 有交集就报错，不静默忽略。manifest 从 log 平衡抽样，每个 log 最多 2 个场景。工程、校准、审计需要使用**不同 log 的 allowlist**；本命令只创建当前工程组，不自动完成跨阶段 split 设计。

保留没有 cache 的入选场景，在运行时明确报错并计入分母；不以“有 cache”偷偷重选更容易的样本。当前 cache 索引使用 `<scene-token>/metric_cache.pkl`，不读取可能含旧服务器绝对路径的 metadata CSV。

manifest 每行：`token / log_name / source_log / frame_start / metric_cache / declared_split / allowlist_sha256`。仅加载本人的可信 pickle/cache；pickle 不是安全的第三方交换格式。

## 3. 真实工程验证脚本

先用 `--count 1` 生成单场景清单，以小预算检查接口和时间：

```bash
bash scripts/offline_search/run_engineering.sh run \
  --manifest /data/offline_search/one_scene_manifest.jsonl \
  --map-root /data/maps \
  --population 4 --generations 1 \
  --output /data/offline_search/one_scene_smoke
```

通过后再运行 16 场景的完整工程预算：

```bash
bash scripts/offline_search/run_engineering.sh run \
  --manifest /data/offline_search/engineering_manifest.jsonl \
  --map-root /data/maps \
  --population 32 --generations 5 \
  --output /data/offline_search/engineering_16
```

输出目录必须不存在，以免覆盖证据。任何场景运行异常都保存 `error.json`，计入汇总，进程最终返回 2；全部场景执行完成返回 0。**返回 0 只表示工程程序完成，不表示候选通过完整安全验收。** 没有可行/改进候选是正常实验结果，不算程序异常。

当前串行运行，每个场景独立 scorer/simulator。八岛 × 32 × 5 是上限 1280 个采样点，重复控制不重评，不可用岛不补预算；初始化拟合、GT 检查和档案复验另记评分次数。按 16 场景实测耗时后再考虑并行，当前不承诺完整预算耗时。

## 4. 搜索和已实现检查

- GT、所有候选均走原始 `navsim.evaluate.pdm_score.pdm_score`；每次仅评分 `[固定 PDM reference, 一个候选]`。
- GT 原始 8 pose 保留；记录 40 个加速度/转向率命令，原动力学重放误差必须小于 `1e-8`。
- 10 维有界控制残差，初始残差固定为 0。生成 dense reference 后取 8 pose，再经正式 LQR rollout。零残差 round-trip 与原始 GT 分开记录。
- 八岛种子：B0 GT 附近；B1–B5 改变同一 GT 几何路径的速度 schedule，再做有界控制拟合；B6/B7 左右小幅控制残差。只使用已知 5 秒 GT 路径，不对路径末端 clip，不凭空外推。路径不足/拟合失真标为 seed unavailable。
- 名义检查：官方 NC/DAC/TTC/C/DDC、缓存中全部物理 box 的全责任碰撞及 0.25 m 余量、连续 SE2 区间保守包络、完整车身道路覆盖及 0.10 m 余量、GT 几何 corridor、地图限速、倒车/加速度/几何侧向加速度/jerk/转向角/转向率。
- 间隔 0.1 秒，包络冲突二分到 0.0125 秒，不能证明分离即拒绝。接触和初始接触同样拒绝；不豁免官方 collided IDs。几何连续性只针对缓存节点定义的分段 SE2 回放。
- 搜索使用 `checked_nominal_pass`，**它不是原计划的完整 `G_search`**：尚未实现的 gate 列在每条记录的 `pending_gates`。
- 缓存 actor 形状无法重构、未知生命周期可达域可能影响候选、相关地图限速未知等仍不可核验。部分track的处理以本文开头schema_version=2修复说明为准，需要10Hz / `observation_sample_res=1` cache。
- 每代档案要求分数较 GT 增加至少 0.005；最多保留 32 条，并复验保存的 8 pose。
- 工程代表使用执行轨迹的 xy/速度/航向 complete-link 与两两距离筛选，最多 4 个。**尚未接入原计划的 anchor/stop 类别，不能称完整多样性审计。**

## 5. 可视化与输出

打开输出目录的 `index.html`，点击场景进入 `report.html`。无需服务器、网络或额外 JS 库。

- 对比 GT 输入/实际执行、候选控制生成 reference、候选 8 pose 插值 reference、候选实际执行。
- 等比例地图、道路孔洞、GT 路径参考、按实际车身尺寸绘制的每秒轮廓。
- 拖动 0–4 秒时间条或播放，同时查看 GT/candidate/actor 的当前位置和当前区间失败事件。
- 切换候选或叠加轨迹；保留高分失败项和不同失败原因样本，避免只看成功图。
- 显示所有正式子指标、PDMS 差值、速度/加速度/jerk 曲线、控制命令、搜索每代的可行数量和最好分数。
- 网页最多 60 条用于浏览，完整搜索记录及全部轨迹数组仍在文件中。

输出文件：

| 文件 | 内容 |
|---|---|
| `environment.json` | Git HEAD/工作区状态、模块 SHA256、Python/依赖版本 |
| `manifest.json` | 输入清单，token 保持字符串 |
| `summary.json` | 全部场景及失败数量；场景内另含检查淘汰统计、初始化、耗时、次数 |
| `candidate_archive.jsonl` | GT、round-trip、全部唯一控制评估的指标和 gate；逐条落盘 |
| `engineering_candidates.npz` | 各 candidate ID 对应 poses/reference/executed/generated/controls/theta，无 pickle |
| `report.html` | 交互检查页面 |

真实运行的 `report.html` 还保存原始 log/cache 的 SHA256、车辆参数与地图根路径。GT 自身失败会显示出来，不能把“GT + 合格候选”整体称为安全集合。

## 6. 从工程版到正式 teacher 的剩余工作

必须补齐：缓存 actor 与原始标注的覆盖核对、完整有序 route 拓扑约束、时变交通灯/停止线检查、4–5 秒固定延续、13 种压力复核、anchor/stop 行为聚类以及独立数据审计。当前未实现任何 `--export-teacher` 开关，不得通过重命名输出文件绕过这些 gate。

本次不加入 LK，不构造偏移初始状态的视觉训练样本。等 V1 搜索与后续 generator 学习收益得到验证，再评估 V2 扩展。
