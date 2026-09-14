# GT附近控制搜索：离线可行性实验最终方案 v2.2

2026-09-14 实施更新：先做 V1、固定真实初始状态的工程验证；增加第12节交互可视化与第13节工程执行范围。第1–11节仍是完整 teacher 验收目标。当前工程代码只实现部分 gate，不能将工程候选当成已完成严格验收的训练标签；准确边界见第13节。未开展真实16场景搜索或generator训练。

范围：只设计和验证离线teacher数据，不训练generator，不改线上推理，不把真实未来送进学生模型。代码基准为76ec37a9f6f1db694d185e87beba38d6bcc7bc26；当前轨迹8×3，0.5–4.0秒，正式rollout 0.1秒。已经完成独立agent的v1评审并据此修订；修订记录见review_log.md。所有数值阈值是预注册的保守实验默认值，不是车辆法规/现实安全认证。

## 1. 成功问题与数据
- 分别判断：(A)能否获得至少一条正式PDMS比GT高且严格约束合格的轨迹；(B)能否进一步获得至少2种差异化改进。允许零条，禁止放松安全条件凑数。
- 16个工程验证场景；64个训练场景校准；冻结配置后审计256场景：128个随机代表样本（先均匀抽log，再在log中均匀抽scene）+128个低GT分挑战样本（S_GT<0.95、相同log-balanced抽样）。组间不重复；工程/校准/审计按log隔离；每log每组最多2个scene。不足时减少并报告，不跨组借数据。随机组估计log-balanced覆盖率，不声称全场景总体比例。NAV1-test及模型验证集不参与搜索设计/阈值调节。
- 必需输入：原始scene/annotations、GT、初始ego state及完整vehicle参数、官方metric cache、带拓扑与限速的地图、0–5秒actor轨迹及灯态。仅已有CSV不足以执行。
- 记录坐标原点、rear_axle/center转换、真实timestamps、track生命周期、地图版本、源文件hash和全部软件版本。

## 2. 冻结评估口径
- 使用navsim/evaluate/pdm_score.py的正式入口；每个候选与固定PDM参考轨迹独立评分。不得把整个候选池一起按max progress归一化。
- 对GT也使用完全相同的8pose→transform_trajectory→get_trajectory_as_array→PDMSimulator→PDMScorer流程。
- 初始状态、轴距、车身尺寸、执行器延迟、LQR参数不作为搜索变量。所有有限性/范围错误、TTC缺失或异常均失败，不用训练评分器的TTC=2异常替代。
- 目标GT不可改善（S_GT+epsilon>1）单列；只有S_GT+epsilon>1才按上界直接跳过；恰好等于1仍可搜索。epsilon是实验阈值而不是安全证明。

## 3. 搜索变量与正式闭环
- 记录GT正式rollout的40步LQR command：u_GT=(a_cmd, steering_rate_cmd)。直接重放这些命令需复现同一动力学rollout，误差需在数值容差内。
- 使用t=[0,0.5,1,2,3,4]秒残差节点，第一个残差固定0，其余5个节点/通道，总10维；节点间线性插值，初始值连续。搜索a残差±1.0m/s²、steering-rate残差±0.03rad/s，禁止采样物理参数。
- 从同一初始状态将u_GT+Bθ经原车辆模型积分，得到dense参考，取0.5秒的8pose，然后必须再次经正式LQR+车辆模型rollout。CEM使用后者的实际评分和约束结果。
- 原始GT单独保存为baseline/seed；θ=0生成的dense轨迹再采样再跟踪不保证等于原始GT，必须单列round-trip变化，不用它替换GT分母。
- 生成控制只做廉价预筛，最终动力学与几何约束均检验正式rollout。保留生成参考、8pose、执行rollout三份，检查跟踪差异。

## 4. 八个初始化岛：速度差异与路径差异分别构造
共同要求：仍相对u_GT用第3节同一10维θ表示。初始化阶段增加的计算单独记账；初始化不是未经检验的teacher。

先从GT几何轨迹拟合按弧长的路径p(s)，剔除停车重复点；需要更长路径时只接续与原command一致、拓扑唯一且切线连续的已知GT后续/冻结route路径。路径覆盖不足或分支不明确则标记该模式seed_unavailable，不沿末端切线盲目外推，也不用PDMPath的clip伪造停在终点。

对B1–B5，先用下表加速度残差修改GT纵向schedule，结合当前v0积分得到s_seed(t)，沿同一p(s)生成8pose。其转向随空间曲率变化，而不是照搬GT按时间的转向。再经原LQR rollout记录seed控制，以有界最小二乘拟合到u_GT+Bθ的10维空间，a和steering residual界仍为±1.0和±0.03。有界最小二乘通常总有界内解，因此用轨迹失真判据决定是否可用：拟合后8pose正式rollout相对初始化目标执行轨迹RMS_xy>0.25m或最大横向偏差>0.50m时，按1、1/2、1/4缩小初始化幅度重试；仍超阈值则记录seed_fit_distortion，不修改硬界。拟合容差在校准阶段冻结。

节点对应t=[0,0.5,1,2,3,4]秒：
| 岛 | 行为起点 | 初始化schedule/控制残差 |
|---|---|---|
| B0 | GT附近 | θ=0，原始GT另保留作比较 |
| B1 | 早加速 | Δa=[0,+0.6,+0.6,0,0,0] m/s² |
| B2 | 晚加速 | Δa=[0,0,0,0,+0.6,0] |
| B3 | 早减速 | Δa=[0,-0.6,-0.6,0,0,0] |
| B4 | 晚减速 | Δa=[0,0,0,0,-0.6,0] |
| B5 | 早加速后回收 | Δa=[0,+0.4,+0.4,0,-0.4,0] |
| B6 | 车道内一侧微调 | Δa=0，Δsteering_rate=[0,+.015,+.015,-.015,-.015,0] rad/s |
| B7 | 反向微调 | B6转向残差取负 |

低速/停车位置没有可靠曲率时，已知路径不足不得补造；可用原GT steering作为明确标记的fallback，路径是否变化仍由正式rollout判定。不能对负速度直接clip后宣称相同动力学，产生负速则缩小seed或判不可用。
候选按GT同一分支的几何路径最大横向偏差0.75m约束，B6/B7不等于换道，也不能用左右转向符号直接命名实际行为。每岛保留独立CEM分布；最终是否有多种行为由执行轨迹距离决定。
拟合得到的θ在所有bounds内后，其种子生成轨迹→8pose→正式rollout重新评估并记录拟合失真；不要把未拟合前的路径当作最终candidate。

## 5. CEM与候选档案
- 每岛32候选×5代×8岛，上限1280个候选/场景。每代32个中包含当前均值及该岛历史最佳，其余30个采样；第一代用初始化替代两者中的一个，重复候选用hash缓存，不额外重评。不可用岛预算不擅自转给其他岛。
- 独立截断正态采样，不把越界样本直接clip到边界；每维拒绝采样最多100次，仍失败则用界内均值并记录。随机seed=hash(experiment_seed, scene_token, island, generation)。初始σ_a=.25、σ_steer=.008，下限分别.03/.001；更新均值与标准差的平滑系数0.5。
- 搜索可行G_search：正式8pose再rollout后通过H0–H6的0–4秒名义检查，但不要求比GT高；H7与第7节压力测试在档案阶段做。将search_feasible与export_certified明确分开。
- 有≥4个可行样本：取4个最高PDMS elite；有1–3个：只用这些点更新均值，σ至少保留旧值的0.8且不低于下限；无可行样本：记录约束违反向量，使用字典序最小的4个样本作恢复方向，均值取其均值与岛初始化中点，σ缩小至0.5倍（保持下限）。违反向量固定为（物理碰撞区间数、最大穿透/间距欠缺、道路越界区间数与最大越界面积、路线/交通灯违反次数、按各物理界归一化的动力学最大超限），只用于恢复方向；非可行点永不进入导出池，禁止EP抵消碰撞。数据不可核验不作为低违反程度样本参与恢复。
- 可行候选排序：PDMS；同分（差<1e-4）按更大最小间距、更小最大jerk、更小归一化控制残差范数；最后按candidate hash确定排序。不同量纲不直接相加。
- 每代将search_feasible且S>=S_GT+.005的候选纳入档案。每岛先保留至多4个去除数值重复的最高质量候选（此时不强求D>=1）；不足32的空位从全局剩余档案补齐。昂贵验收最多32个/场景，报告此前被budget截断的数量，承认该截断可能遗漏模式。
- 首版不对PDMS反传。后续代理梯度搜索仍使用相同独立验收，不能用学习scorer预测值充当正式标签。

## 6. 最终硬约束（全部必须通过）
H0 数据与初始状态：原始GT/缓存坐标时间一致；ego初始存在任意物理重叠则场景不进入strict_extra池。已存在collided IDs不得忽略；actor出生消失/缺标、地图或灯态未知影响候选时，标记unverifiable并不导出。
H1 官方指标：名义0–4秒NC=DAC=TTC=Comfort=DDC=1（数值容差1e-6）。仅导出额外标签时要求总PDMS≥S_GT+0.005，未四舍五入原值比较。压力测试不要求每次仍超过GT，见第7节。
H2 无碰撞：独立用完整ego footprint与所有物理actor/静态物体进行全责任碰撞检查，不使用官方fault过滤。接触算失败。ego膨胀0.25m后不得与物体相交；红灯虚拟polygon按交通规则处理，不混同物理碰撞。
H3 连续区间：在名义0.1秒节点之间，按线性位置+最短角插值定义连续运动；对ego与每个actor各自取区间两端footprint顶点凸包，并膨胀e=R*(Δyaw)^2/8+1e-6米（R为相对插值原点最远车角半径；actor一般box center，ego为rear axle）；该界适用于此处明确采用的线性位置+线性最短角插值。先做安全buffer再用包络认证，必要时二分到0.0125秒；区间包络分离才认证，仍有相交且无法排除时拒绝，禁止仅把20/50Hz无碰撞称为连续保证。间距也用buffer后的包络认证。
H4 道路与路线：ego完整多边形膨胀0.10m后被合法drivable union覆盖，包含上述连续包络，不仅四角；点/包络正好接触边界也拒绝（几何数值容差单独记录，不能变成安全余量）。沿GT实际通行route拓扑序列，不反向、不换出不同出口/分支；距GT几何路径横向偏差≤0.75m（按空间进度对应，不按相同时间），低速方向判定需独立处理；地图限速缺失不默认为15m/s。
H5 交通灯：原始逐帧灯态+停止线联合检查，禁止红灯越线。2Hz灯态切换区间采用保守状态；在未知状态区间跨线则unverifiable。route与command始终不变。
H6 动力学：正式rollout无倒车/NaN/限幅投机，速度≤地图限速；额外工程包络默认纵向a∈[-3.5,2.0]m/s²，几何侧向|a|=|v²tanδ/L|≤3.0m/s²，|纵向jerk|≤3.5m/s³，|δ|≤0.5rad，|steering rate|≤0.3rad/s。明确同时计算未平滑差分与官方舒适性，不把bicycle状态中被置零的lateral acceleration当侧向加速度。固定初始状态不满足则不可行并单列，不能从t=0之后偷偷忽略。
H7 末端：使用已有5秒回放额外检查4–5秒，不重算或替换名义4秒PDMS。为每条轨迹从其真实4秒执行状态出发，在冻结的同一route路径上构造1秒减速reference：起点pose/v/a/steering连续；若|v_end|≤0.05m/s且|a_end|≤0.05m/s²，采用静止保持reference并实际运行原LQR/车辆模型，不强迫进入-2m/s²；其他情况a以|jerk|≤3.5渐变至-2m/s²；临近零速时用满足同一jerk bound的停车profile，无法在所需边界内连续接入则reject。重新初始化同配置LQR，以该reference和真实末状态运行原车辆模型；不直接覆盖实际状态/裁剪负速度。需要路径贴合时用确定的五次横向连接（起点位置/一阶/二阶条件匹配执行状态，终点匹配冻结路径），连接长度按1秒reference行驶弧长，过短或超界则不强行连接。16场景工程阶段必须验证该延续策略；它本身造成的失败单列tail_policy_failure，不据此声称候选没有任何其他安全延续。尾段全程通过H0/H2–H6；没有足够map/actor/灯态则unverifiable。这里只认证1秒固定延续，不能称完整停止或长期闭环安全。

## 7. 固定压力复核：13个模板，按有效唯一case报告
对最多32个档案candidate及GT做同样压力测试；名义scene/cache不可原地修改。所有模板在64场景校准后冻结。

令dv=min(0.3m/s,0.5*v0)，dy=0.10m，dψ=0.5°；初始扰动横向方向取原ego局部y轴。v0=0时速度±case退化，去重并标degenerate，不能伪称13个独立case，也不因此排除全部停车场景。参考8pose先按原始初始ego状态转成绝对世界轨迹，在各初态扰动下保持该绝对reference不变，重新LQR跟踪；GT也相同。

| case | 初始Δv | 初始Δy | 初始Δψ | actor timewarp符号 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 1/2 | +dv / -dv | 0 | 0 | 0 |
| 3/4 | 0 | +dy / -dy | 0 | 0 |
| 5/6 | 0 | 0 | +dψ / -dψ | 0 |
| 7/8 | 0 | 0 | 0 | + / - |
| 9 | +dv | +dy | +dψ | + |
| 10 | +dv | -dy | -dψ | - |
| 11 | -dv | +dy | -dψ | - |
| 12 | -dv | -dy | +dψ | + |

actor warp统一应用于所有动态actor：φ(t)=t±0.1*t*(1-t/5)，0≤t≤5；它固定0和5秒，严格单调，不读取5秒外。用原始轨迹在φ(t)的pose/heading构造新actor状态，velocity乘φ'(t)，静态物体不动。然后明确以0.1秒warp节点重新定义分段线性位置+最短角SE2合成回放，连续认证仅针对这个定义，不把原非线性warp曲线套进线性包络证明。track出生/消失按同一warp映射，不在未知生命期之外外推；生命周期边界加入认证分段，只有单侧观测而不能认证的区间拒绝。所有对象查找、速度和时间metadata一起更新。该warp只是有限相位敏感性测试，不代表真实交通反应模型。

有效压力case需通过0–5秒的无接触、道路/路线、交通灯、动力学及尾段检查；另要求0–4秒官方安全/舒适子指标NC/DAC/TTC/DDC/C均为1，不要求PDMS改善或EP下界。扰动后的评分明确记stress_score，不能当官方benchmark成绩。GT在某个压力case失败不允许candidate也失败；candidate必须独立通过。
scenario构造的初始重叠/不完整生命周期/无地图或灯态等为unverifiable，候选不得作为strict_robust导出；背景互相碰撞等无效stress记录scenario_invalid，不静默跳过。官方统计只报告原始名义场景；压力集另报有效唯一case数量及失败分布。

## 8. 两个标签池：严格改进与差异化改进
1. strict_improvement_pool：通过名义、连续区间、尾段、全部有效唯一压力case，且名义S>=S_GT+.005的真实candidate。允许与GT很接近；不能因为D(GT)<1就把质量提升抹掉。
2. diverse_extra_pool：从strict_improvement_pool中选最多4个彼此充分不同、并与GT充分不同的代表。前者非空而后者为空是有效结果，表示质量改进存在但没有形成新模式。

聚类输入：实际保存的8pose重新走正式rollout所得0.1秒轨迹。对0–4秒计算RMS_xy（同一时刻，不用DTW抵消速度差异）、RMS_v、圆周heading差。速度<0.5m/s时航向项mask，并单列有效时间比例。
固定route锚点s*=s_GT(4秒)，若GT总进度<1米则不使用锚点特征，报告anchor_disabled；其余按冻结route的同一进度坐标，记录首次到达时间、未到达类别。停止类别定义v<0.3m/s持续≥0.5秒。
同一停止/锚点到达类别内，D=max(RMS_xy/0.75m,RMS_v/0.75m/s,RMS_heading/5°,abs(Δt_anchor)/0.5s)。两者均未到达或anchor_disabled时省略该项，不填∞；到达类别不同则先分组。跨类别分离视为行为分离，但另设防抖：最终代表对必须至少满足RMS_xy>=0.25m或RMS_v>=0.25m/s，避免刚好跨分类阈值制造模式。
每组采用complete-link、阈值D<1聚类。每簇代表从真实成员中按PDMS→鲁棒最小间距→较小jerk→hash排序选择，不取均值或未复验medoid。
代表选择后，再按上述质量顺序贪心保留，要求与GT及所有已选代表D>=1（跨类别按上述规则），不足则少导出；complete-link簇不同不自动代表满足这个距离。近GT和互相重复的合格轨迹保留在strict_improvement_pool，标valid_but_redundant，不删除证据。
GT单独保存为baseline_label并标注自己的每个gate；若GT不合格但搜索candidate合格，进入repair分类，不把GT与safe_extra混称安全集合。原始8pose、最后复验8pose必须字节/数值一致，所有分数以最后复验为准。

## 9. 输出、阶段门槛与预算
- scene_manifest.jsonl / candidate_archive.parquet / selected_targets.npz / summary.json / audit figures / config.yaml / environment.lock；两标签池、GT和repair状态分别存储；token永远字符串。
- 每场景保存原始GT、S_GT、K个真实候选、原始控制/参考/最终执行轨迹、正式subscores、nominal与stress结果、质量权重建议；可行性阶段不接训练loss。
- 分别报告：数据可核验比例、GT可改进比例、nominal安全提升率、strict_improvement覆盖率、diverse_extra至少1/至少2模式比例、所有场景无改进记0后的ΔPDMS均值、成功子集的均值/中位/p10、按log bootstrap区间、各gate淘汰数、round-trip失败率、每种初始化贡献、平均/p95评分次数与CPU时间。分母始终含失败，不只统计成功场景；代表样本与挑战样本分开。
- 16场景工程阶段通过坐标、碰撞、归一化及重放检查后才扩大。校准64场景冻结参数；审计256场景不再改阈值。预注册进入后续训练的实用门槛：随机128样本至少13个有strict improvement，可考虑先做少量单改进蒸馏；若其中至少7个场景有≥2种diverse extra，再考虑多目标蒸馏。每个导出样本都已满足≥0.5PDMS点改进，不再用平均分掩盖个别未提升。人工检查最多50个分层样本（不足50则全检），发现任何碰撞/越界/错误路线漏检则回到工程阶段并废弃旧审计结论，不能只删坏例。门槛仅用于资源决策，不是显著性或现实安全标准；阈值冻结后不能为达标而降安全余量。
- 算力先测16场景：1280候选+最多32候选×13压力复评/场景，约1700次基础轨迹评估（上界取1720，含GT），另记八岛seed构造/拟合最多21次、最终代表及strict池复验最多32次；最大约1773次基础评估。每次还包括PDM参考对、1秒延续、连续包络的不同成本，不能将它等同1773次简单网络forward；单场景超预算必须记budget_exhausted，不放宽gate。分worker独立simulator/scorer，避免共享可变状态。

## 10. 实施模块及检查
- 独立目录实现manifest、official_adapter、control_search、strict_validator、stress_suite、cluster_select、report；不修改官方计分行为。
- meaningful tests：GT命令重放；rear axle坐标；独立评分不受候选顺序/分批影响；责任豁免碰撞仍被reject；角点都在路内但车身跨洞reject；两个采样点之间交叉碰撞reject；红灯/缺失track拒绝；8pose二次rollout损失记录；初始overlap不豁免；聚类返回原始candidate而不是均值；多worker确定性；状态副作用检查。
- 该计划未执行搜索，无模型收益或安全通过结果。输出只是在明确回放、插值、地图与有限压力集合下通过约束的离线样本，不是现实无碰撞保证。


## 11. 实施顺序与完成定义
M0 数据与oracle适配：manifest、坐标/时间/GT提取、正式评分重复性、score量纲与GT upper-bound。失败不进入搜索。
M1 strict_validator：先用人工构造的反例验证全责任碰撞、连续区间、完整footprint道路覆盖、红灯/未知数据和初始状态处理。只有约束实现可靠才生成teacher。
M2 GT控制提取与八岛初始化：命令重放、不同速度同路径初始化、有界拟合、round-trip统计、路径末端无clip投机。
M3 16场景计时与CEM：输出全部gate淘汰与无可行elite原因；不先跑全训练集。
M4 64场景校准冻结config；128随机+128挑战审计、压力复核、两池聚类选择；生成复验可重放包。
M5 出离线可行性报告。此次完成不包含generator训练；下一阶段才设计GT与多目标一对一匹配、权重及固定候选评测。

成本控制不得改变约束语义：先廉价数值/节点筛选，再连续认证；缓存scene几何与PDM参考，worker内部批量simulate但每候选评分保持官方独立归一化。任何批量加速必须与原pdm_score逐条结果对齐后启用。每worker独占可变scorer/simulator/observation，压力case使用独立副本。

可行性结论须分开：搜索失败、数据不可核验、预算耗尽、找到单一改进、找到多种改进。这些都不是scorer已经提升的证据；离线成功还需后续学生训练验证。

## 12. 搜索轨迹与 GT 的可视化检查

每个工程场景生成可离线打开的 `report.html`，批量结果生成 `index.html`。报告无需联网、服务器或第三方前端库。人工检查须能查看失败候选，不能只导出通过项。

1. **四类参考分别保留**：原始 GT 的8pose插值和正式执行轨迹、控制搜索产生的dense reference、候选8pose插值reference、候选正式执行轨迹。不同颜色和线型区分，禁止把生成reference画成最终执行结果。
2. **空间图**：等米制比例；GT路径参考、完整道路几何及孔洞、每秒ego footprint、同一时刻所有已加载actor footprint。采用世界坐标平移以改善数值显示，rear axle轨迹与实际车身中心保持正确关系。可切换或叠加候选。
3. **时间检查**：0–4秒、0.1秒步长滑块和播放；GT、候选、actor同步。显示当前区间的碰撞/余量或道路认证失败事件，允许定位首次失败附近。当前工程版没有尾段或压力case播放，不能显示虚假的5秒安全结果。
4. **指标图**：GT与候选全部正式子指标、PDMS及差值；速度、纵向加速度、未平滑jerk曲线；可展开控制残差和LQR命令。分数按0–100显示、原始文件保留0–1。
5. **搜索图表**：每个初始化岛、每代可行数量、历史最佳、累计评估数、不可用seed原因。GT与zero-residual round-trip分开保存和说明。
6. **选图范围**：全部改进档案成员、不同失败原因代表、高分失败项；浏览器页面至多60条，完整候选记录/数组另存。报告必须展示未完成gate以及`export_certified=false`。
7. **人工验收**：优先检查大幅进度提升、低间距、明显转向、GT失败而候选通过、高分却失败的场景；逐时刻核对车身与动态物体，不能仅依据俯视轨迹线不相交认定无碰撞。发现漏检返回验证器修复，不删除坏例后继续报收益。

## 13. 工程版本、执行入口与阶段边界

代码位置：`C:/Users/ROA7SGH/Downloads/code/DrivoR/navsim/offline_search/`。
执行说明：`C:/Users/ROA7SGH/Downloads/code/DrivoR/scripts/offline_search/README.md`。
脚本：`scripts/offline_search/run_engineering.sh`、`run_engineering.ps1`；一次测试和演示：`verify_engineering.sh`。

当前已实现：训练allowlist与log平衡manifest；V1公共评分入口；GT控制记录/重放/重复评分检查；10维有界残差；速度同路径种子及拟合、左右种子；独立岛CEM及跨代档案；缓存回放的全责任box连续包络检查、全车身道路检查、工程动力学/限速检查；基础complete-link代表选取；完整轨迹保存及第12节交互报告。

**工程检查集合记为 G_engineering，而不是第5节完整 G_search。** 未完成的项目不能默认为通过：raw actor覆盖核对、有序route拓扑、时变灯态/停止线、1秒固定延续、13个压力case、anchor/stop行为类别和正式两池审计。这些逐条列入`pending_gates`，所有结果固定`export_certified=false`、`training_export_enabled=false`。工程代表不等于第8节的`diverse_extra_pool`，名义改进档案不等于`strict_improvement_pool`。不创建`selected_targets.npz`。

先运行合成几何/搜索测试与官方合成场景集成测试，再用1个真实场景、4候选×1代检查接口与耗时；最后执行16场景、32候选×5代。真实初始ego/图像不变，不加入LK，不使用初始偏移状态生成视觉训练标签。缺失输入和场景异常均计入分母；所有原始数据与cache仍需在Linux DrivoR环境中提供。

本机核心测试和DEMO仅验证工程机制，DEMO采用合成代理分数，不是NAVSIM成绩。真实官方适配集成测试在Windows上因vendored nuPlan的`fcntl`依赖明确skip；必须在Linux运行通过后才能使用真实工程结果。即使工程程序成功退出，也不代表已经通过完整teacher验收。
