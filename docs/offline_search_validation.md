# 工程实现验证记录 — 2026-09-14

首轮full_results后修复：28项测试通过、1项官方Linux集成测试在Windows跳过；新增部分track保留、候选级未知区间范围检查、时间抖动/缺帧判别、限速缺失不掩盖超速、真实搜索启动状态和报告缺帧序列测试。真实16场景修复后重跑尚未执行；所有候选仍不可导出为teacher。重跑步骤见scripts/offline_search/README.md。

代码基于 DrivoR `76ec37a9f6f1db694d185e87beba38d6bcc7bc26` 新增独立模块。未修改模型、训练入口或官方评分定义。

## 本机已执行

- Python 3.12 隔离环境（系统共享 NumPy 2.3.5；新装 Shapely 2.1.2、SciPy 1.18.1、pytest 9.1.1），未修改用户训练环境。
- `python -m pytest tests/offline_search -q`：**14 passed, 1 skipped**。
- `python -m compileall -q navsim/offline_search tests/offline_search`：通过。
- 两个 Bash 执行脚本的 `bash -n` 语法检查：通过。
- 合成 DEMO：8候选/岛、2代，成功生成候选记录、NPZ、场景HTML和总索引。
- `node tests/offline_search/check_report_js.cjs <report.html>`：60条候选、每条5个时间位置；绘图逻辑执行通过，无 NaN/undefined/Infinity。此测试使用 document stub，不是浏览器渲染。

测试覆盖：采样点之间交叉碰撞；初始接触/安全余量；道路孔洞与完整车身；旋转包络；安全平行运动；有界确定性CEM；失败elite不入档案；聚类保留真实成员且两两分离；GT上界边界；残差初始点；token字符串；报告转义与禁止teacher导出；manifest按log均衡且保留缺cache场景；训练/验证token重叠拒绝。

## 尚未验证的部分

- 真实 NAVSIM log + cache + map 的工程运行尚未执行。
- 官方合成场景集成测试在Windows因vendored nuPlan依赖`fcntl`跳过；必须在用户Linux DrivoR环境中执行。代码内没有用模拟测试替换这一检查。
- 本机组策略阻止启动PowerShell脚本，未绕过策略；因此PowerShell wrapper启动未验证。
- 浏览器工具的URL策略阻止打开本地HTML，未绕过策略；因此未完成浏览器视觉验收，仅完成JS逻辑检查。

## 阶段边界

该版本输出固定 `export_certified=false`、`training_export_enabled=false`。尚未完成raw actor覆盖核对、完整route/灯态、1秒尾段、13个压力case以及anchor/stop聚类，不能将名义改进当作完整teacher通过结果。完整方案见 `offline_search_plan.md`，使用方法见 `../scripts/offline_search/README.md`。
