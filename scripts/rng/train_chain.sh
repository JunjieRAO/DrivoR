#!/usr/bin/env bash
#
# train_chain.sh — 链式自动重投训练窗口，直到练满 max_epochs。
#
# 原理：
#   1. 提交一个训练窗口 (bsub)，再提交一个轻量“控制器”作业，依赖 ended(训练JOBID)。
#   2. 训练窗口被墙钟(-W)中断 → LSF 记为 EXIT → 控制器自动提交下一个窗口(从最新ckpt续训)。
#   3. 训练窗口正常跑完 max_epochs → trainer.fit 返回0 → LSF 记为 DONE → 控制器停止链条。
#   run_training_full.py 自带“glob 找最新 ckpt 续训”，实验名固定，故跨窗口续训是自动的。
#
# 安全机制（防止链条失控）：
#   - 最大窗口数上限 (--max-windows，默认 8)。
#   - 停滞检测：某窗口若未产生新 checkpoint(单epoch>窗口时长 或 启动即崩溃) → 停止并告警。
#   - 重复链检测：已有活动控制器时拒绝再起一条(除非 --force)。
#   - --dry-run：只打印将执行的操作，不真正提交。
#
# 用法：
#   scripts/train_chain.sh --bsub scripts/run_nav1_train_oneclick_a100.bsub
#   scripts/train_chain.sh --bsub scripts/run_nav1_train_oneclick_h200.bsub --max-windows 6
#   scripts/train_chain.sh --bsub <file> --dry-run     # 演练，不提交
#   scripts/train_chain.sh --bsub <file> --fresh       # 重置链条状态后重新开始（仍会自动续训已有ckpt）
#   scripts/train_chain.sh --bsub <file> --restart     # 不寻找/续训已有checkpoint，从零开始训练新一条链
#                                                       # (通常应与 --fresh 一起用；不会删除任何 checkpoint 文件)
#   scripts/train_chain.sh --help
#
set -uo pipefail

SELF="$(readlink -f "$0")"
REPO="$(cd "$(dirname "$SELF")/.." && pwd)"

# ------------------------------------------------------------------ args ----
BSUB_FILE=""
MAX_WINDOWS="${DRIVOR_CHAIN_MAX_WINDOWS:-8}"
CTRL_QUEUE="${DRIVOR_CHAIN_CTRL_QUEUE:-batch_cpu}"
PREV_JOBID=""          # 内部使用：上一训练窗口的 JOBID（由控制器传入）
DRY_RUN=0
FRESH=0
FORCE=0
RESTART=0

usage() { grep '^#' "$SELF" | sed 's/^#\{1,\} \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --bsub)         BSUB_FILE="$2"; shift 2 ;;
    --max-windows)  MAX_WINDOWS="$2"; shift 2 ;;
    --ctrl-queue)   CTRL_QUEUE="$2"; shift 2 ;;
    --prev-jobid)   PREV_JOBID="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --fresh)        FRESH=1; shift ;;
    --restart)      RESTART=1; shift ;;
    --force)        FORCE=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; echo "用 --help 查看用法。" >&2; exit 2 ;;
  esac
done

if [ -z "$BSUB_FILE" ]; then
  echo "❌ 必须用 --bsub 指定训练提交脚本。用 --help 查看用法。" >&2
  exit 2
fi
# 归一化成绝对路径（控制器作业里需要可访问）
BSUB_FILE="$(readlink -f "$BSUB_FILE" 2>/dev/null || echo "$BSUB_FILE")"
if [ ! -f "$BSUB_FILE" ]; then
  echo "❌ 找不到 bsub 文件: $BSUB_FILE" >&2
  exit 2
fi

command -v bsub  >/dev/null 2>&1 || { echo "❌ 未找到 bsub (需要 LSF 环境)" >&2; exit 1; }
command -v bjobs >/dev/null 2>&1 || { echo "❌ 未找到 bjobs (需要 LSF 环境)" >&2; exit 1; }

# 从训练 bsub 文件里提取 -P 项目号，一并传给控制器作业（集群策略要求所有作业都带项目号）
PROJECT="$(grep -oE '^#BSUB[[:space:]]+-P[[:space:]]+\S+' "$BSUB_FILE" 2>/dev/null | head -1 | awk '{print $3}')"
PROJECT_ARGS=()
[ -n "$PROJECT" ] && PROJECT_ARGS=(-P "$PROJECT")

# ------------------------------------------------------------- env & paths --
# 取 NAVSIM_EXP_ROOT / 实验名（与 oneclick 训练脚本保持一致）
# 注意：这里故意不 source setup_drivor_env.sh —— 它会 module purge/load 并 conda activate，
# 从而改写当前 shell 的 PATH/LD_LIBRARY_PATH 等环境变量，可能连带清掉集群预置的 LSF 环境
# (例如 -gpu "num=N" 新语法所需的 LSB_GPU_NEW_SYNTAX)，导致本脚本后面的 bsub 提交报
# "Bad argument. Job not submitted."（而在正常登录 shell 里手动 bsub 却能成功）。
# 所以只从该脚本里"读出" NAVSIM_EXP_ROOT 的值，不执行它、不污染当前环境。
if [ -z "${NAVSIM_EXP_ROOT:-}" ]; then
  NAVSIM_EXP_ROOT="$(grep -E '^export NAVSIM_EXP_ROOT=' "$REPO/scripts/setup_drivor_env.sh" 2>/dev/null \
                      | head -1 | cut -d= -f2- | tr -d "\"'")"
fi
NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO/exp}"

ONECLICK="$REPO/scripts/run_nav1_train_oneclick.sh"
EXPERIMENT="$(grep -E '^EXPERIMENT=' "$ONECLICK" 2>/dev/null | head -1 | cut -d= -f2 | tr -d "\"'")"
[ -z "$EXPERIMENT" ] && EXPERIMENT="training_drivoR_Nav1_traj_long_25epochs"
MAX_EPOCHS="$(grep -oE 'trainer\.params\.max_epochs=[0-9]+' "$ONECLICK" 2>/dev/null | head -1 | cut -d= -f2)"
[ -z "$MAX_EPOCHS" ] && MAX_EPOCHS="?"

EXP_DIR="$NAVSIM_EXP_ROOT/ke/$EXPERIMENT"
STATE_DIR="$EXP_DIR/.chain"
CKPT_GLOB="$EXP_DIR/*/lightning_logs/version_*/checkpoints/*.ckpt"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 最新 checkpoint 的 "mtime|path"，无则返回空串
newest_ckpt_stamp() {
  local f t newest="" nt=0
  shopt -s nullglob
  for f in $CKPT_GLOB; do
    t=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if [ "$t" -gt "$nt" ]; then nt=$t; newest="$f"; fi
  done
  shopt -u nullglob
  [ -n "$newest" ] && echo "${nt}|${newest}" || echo ""
}

# 查询作业最终状态：DONE / EXIT / 空(未知)
get_stat() {
  local jid="$1" s=""
  s=$(bjobs -a -noheader -o "stat" "$jid" 2>/dev/null | awk 'NF{print;exit}')
  [ -z "$s" ] && s=$(bjobs -a "$jid" 2>/dev/null | awk 'NR==2{print $3}')
  if [ -z "$s" ] && command -v bhist >/dev/null 2>&1; then
    if bhist -l "$jid" 2>/dev/null | grep -q "Done successfully"; then s="DONE";
    elif bhist -l "$jid" 2>/dev/null | grep -qEi "Exited|TERM_|Signal"; then s="EXIT"; fi
  fi
  echo "$s"
}

parse_jobid() { sed -n 's/^Job <\([0-9]\+\)>.*/\1/p' | head -1; }

# ----------------------------------------------------------------- fresh ----
if [ "$FRESH" = 1 ]; then
  log "🧹 --fresh: 重置链条状态目录 $STATE_DIR (不会删除任何 checkpoint)"
  [ "$DRY_RUN" = 0 ] && rm -rf "$STATE_DIR"
fi

echo "========================================================================"
echo " DrivoR 训练链  |  实验: $EXPERIMENT  |  目标: ${MAX_EPOCHS} epochs"
echo " 提交脚本: $BSUB_FILE"
echo " 最大窗口数: $MAX_WINDOWS   控制器队列: $CTRL_QUEUE   $([ "$DRY_RUN" = 1 ] && echo '[DRY-RUN]')$([ "$RESTART" = 1 ] && echo '  [RESTART：不续训，从零开始]')"
echo "========================================================================"

if [ "$RESTART" = 1 ] && [ -n "$PREV_JOBID" ]; then
  # 安全兜底：--restart 只应作用于手动发起的第一个窗口，控制器接力时绝不应重新携带该参数
  # (正常情况下控制器不会传 --restart；此分支只是防御性保护，避免因误传导致整条链反复从零开始)
  echo "⚠️ 检测到控制器接力窗口仍带有 --restart，已忽略以避免重复从零开始训练。" >&2
  RESTART=0
fi

# ---------------------------------------------- 评估上一窗口(仅控制器触发) --
if [ -n "$PREV_JOBID" ]; then
  st="$(get_stat "$PREV_JOBID")"
  log "上一训练窗口 JOBID=$PREV_JOBID 结束状态: ${st:-未知}"
  case "$st" in
    DONE) echo "✅ 训练已完成（达到 ${MAX_EPOCHS} epochs）。链条正常结束。"; exit 0 ;;
    EXIT) log "↻ 上一窗口被墙钟中断/退出，准备接力下一个窗口…" ;;
    *)    log "⚠️ 无法确定上一窗口状态，改用 checkpoint 进度判断是否继续…" ;;
  esac
fi

# --------------------------------------------- 重复链检测（仅首次手动启动）--
if [ -z "$PREV_JOBID" ] && [ "$FORCE" = 0 ] && [ "$DRY_RUN" = 0 ]; then
  active=$(bjobs -a -noheader -o "id stat job_name" 2>/dev/null \
            | awk '$3=="drivoR_chain_ctrl" && ($2=="PEND"||$2=="RUN"){print $1}' | head -1)
  if [ -n "$active" ]; then
    echo "⚠️ 检测到已有活动的链条控制器作业 ($active)。"
    echo "   若确实要另起一条链，请加 --force；否则先 bkill $active。"
    exit 1
  fi
fi

# --------------------------------------------------------- 窗口数上限保护 --
count=0
[ -f "$STATE_DIR/window_count" ] && count=$(cat "$STATE_DIR/window_count" 2>/dev/null || echo 0)
if [ "$count" -ge "$MAX_WINDOWS" ]; then
  echo "⛔ 已达最大窗口数 ($MAX_WINDOWS)，停止。"
  echo "   若训练仍未完成：可能单 epoch 太慢，或需要更多窗口。"
  echo "   处理：调大 --max-windows，或检查日志后 --fresh 重来。"
  exit 0
fi

# ----------------------------------------------------------- 停滞检测保护 --
now_stamp="$(newest_ckpt_stamp)"
base=""
[ -f "$STATE_DIR/baseline_stamp" ] && base=$(cat "$STATE_DIR/baseline_stamp" 2>/dev/null || echo "")
if [ -n "$PREV_JOBID" ] && [ "$now_stamp" = "$base" ]; then
  echo "⛔ 停滞检测：上一窗口没有产生新的 checkpoint。"
  echo "   最可能原因：单个 epoch 耗时 > 窗口时长(-W)，或训练启动即崩溃。"
  echo "   本链条已停止以防空转。请增大训练 bsub 的 -W，或检查日志：$EXP_DIR"
  exit 1
fi

# ------------------------------------------------------- 提交下一训练窗口 --
next=$((count + 1))

if [ "$RESTART" = 1 ]; then
  export DRIVOR_NO_RESUME=1
  log "▶ 提交第 ${next}/${MAX_WINDOWS} 个训练窗口（--restart：不寻找已有 checkpoint，从零开始训练）"
else
  unset DRIVOR_NO_RESUME 2>/dev/null || true
  log "▶ 提交第 ${next}/${MAX_WINDOWS} 个训练窗口"
fi
if [ "$DRY_RUN" = 1 ]; then
  echo "   [dry-run] $([ "$RESTART" = 1 ] && echo 'DRIVOR_NO_RESUME=1 ')bsub < $BSUB_FILE"
  TRAIN_JOBID="DRYRUN"
else
  sub_out="$(bsub < "$BSUB_FILE" 2>&1)"; echo "   $sub_out"
  TRAIN_JOBID="$(echo "$sub_out" | parse_jobid)"
  if [ -z "$TRAIN_JOBID" ]; then
    echo "❌ 训练作业提交失败，终止链条。"
    exit 1
  fi
  # 仅在提交成功后才推进窗口计数与基线戳（失败的提交不再白占一个窗口名额）
  mkdir -p "$STATE_DIR"
  echo "$next"      > "$STATE_DIR/window_count"
  echo "$now_stamp" > "$STATE_DIR/baseline_stamp"
fi

# ------------------------------------------- 提交控制器（依赖训练窗口结束）--
log "▶ 提交控制器作业（依赖 ended(${TRAIN_JOBID})，在训练窗口结束后决定是否接力）"
if [ "$DRY_RUN" = 1 ]; then
  echo "   [dry-run] bsub -J drivoR_chain_ctrl -q $CTRL_QUEUE -n 1 -M 1000 -W 0:15 ${PROJECT_ARGS[*]:-} \\"
  echo "                  -w \"ended(${TRAIN_JOBID})\" \\"
  echo "                  bash $SELF --bsub $BSUB_FILE --max-windows $MAX_WINDOWS --ctrl-queue $CTRL_QUEUE --prev-jobid ${TRAIN_JOBID}"
else
  ctrl_out="$(bsub -J "drivoR_chain_ctrl" -q "$CTRL_QUEUE" -n 1 -M 1000 -W 0:15 "${PROJECT_ARGS[@]}" \
               -w "ended(${TRAIN_JOBID})" \
               -o "$STATE_DIR/ctrl.%J.out" -e "$STATE_DIR/ctrl.%J.err" \
               bash "$SELF" --bsub "$BSUB_FILE" --max-windows "$MAX_WINDOWS" \
                            --ctrl-queue "$CTRL_QUEUE" --prev-jobid "$TRAIN_JOBID" 2>&1)"
  echo "   $ctrl_out"
  CTRL_JOBID="$(echo "$ctrl_out" | parse_jobid)"
  if [ -z "$CTRL_JOBID" ]; then
    echo "⚠️ 控制器作业提交失败：链条不会自动接力！"
    echo "   训练窗口 $TRAIN_JOBID 仍会运行，结束后请手动再次运行本脚本继续。"
    exit 1
  fi
fi

echo "------------------------------------------------------------------------"
echo "已提交窗口 #$next（训练 JOBID=${TRAIN_JOBID}）。"
echo "训练结束后控制器会自动判断：DONE→停止，被中断→提交下一窗口。"
echo "查看状态: bjobs -w    停止整条链: bkill <控制器JOBID> 及正在运行的训练JOBID"
echo "链条状态目录: $STATE_DIR"
echo "========================================================================"
