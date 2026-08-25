#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/helper/profile_lsf_job_perf.sh <JOBID> [DURATION_SECONDS]
# Example:
#   bash scripts/helper/profile_lsf_job_perf.sh 12606056 600

JOB_ID="${1:-}"
DURATION="${2:-600}"

if [[ -z "$JOB_ID" ]]; then
  echo "Usage: $0 <JOBID> [DURATION_SECONDS]"
  exit 1
fi

if ! [[ "$DURATION" =~ ^[0-9]+$ ]] || [[ "$DURATION" -le 0 ]]; then
  echo "[ERROR] DURATION_SECONDS must be a positive integer, got: $DURATION"
  exit 1
fi

if ! command -v bjobs >/dev/null 2>&1; then
  echo "[ERROR] bjobs not found in PATH. Please run inside your LSF environment."
  exit 1
fi

JOB_INFO="$(bjobs -noheader -o 'user stat exec_host' "$JOB_ID" 2>/dev/null || true)"
if [[ -z "$JOB_INFO" ]]; then
  echo "[ERROR] Job $JOB_ID not found or not visible."
  exit 1
fi

JOB_USER="$(awk '{print $1}' <<<"$JOB_INFO")"
JOB_STAT="$(awk '{print $2}' <<<"$JOB_INFO")"
EXEC_HOST_RAW="$(awk '{$1=""; $2=""; sub(/^  */, ""); print}' <<<"$JOB_INFO")"

if [[ "$JOB_STAT" != "RUN" ]]; then
  echo "[ERROR] Job $JOB_ID is not RUN (current stat: $JOB_STAT)."
  echo "        Start this while the target job is running."
  exit 1
fi

# exec_host often looks like: 24*rng-dl01-w25n01 or 1*hostA:1*hostB
TARGET_HOST="$(awk '{
  split($1, segs, ":");
  h = segs[1];
  sub(/^[0-9]+\*/, "", h);
  print h;
}' <<<"$EXEC_HOST_RAW")"

if [[ -z "$TARGET_HOST" ]]; then
  echo "[ERROR] Failed to parse execution host from: $EXEC_HOST_RAW"
  exit 1
fi

TIMESTAMP="$(date +%F_%H%M%S)"
OUT_DIR="jobperf_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "[INFO] JOB_ID      : $JOB_ID"
echo "[INFO] JOB_USER    : $JOB_USER"
echo "[INFO] JOB_STAT    : $JOB_STAT"
echo "[INFO] EXEC_HOST   : $EXEC_HOST_RAW"
echo "[INFO] TARGET_HOST : $TARGET_HOST"
echo "[INFO] DURATION    : ${DURATION}s"
echo "[INFO] OUTPUT_DIR  : $OUT_DIR"

run_sampler_block() {
  local out_dir="$1"
  local duration="$2"
  local job_user="$3"

  sample_job_proc_block() {
    local jout_dir="$1"
    local jduration="$2"
    local juser="$3"
    local jpattern='python|torchrun|run_training_full.py|run_training.py|run_training_simscale.py|run_training_full_simscale.py'

    : > "$jout_dir/job_pid_trace.log"
    : > "$jout_dir/job_cpu_trace.log"
    : > "$jout_dir/job_io_trace.log"

    local t=0
    while [[ "$t" -lt "$jduration" ]]; do
      local ts pids pid_csv cpu_line io_line
      ts="$(date +%s)"
      pids="$(pgrep -u "$juser" -f "$jpattern" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+$//' || true)"
      echo "$ts|$pids" >> "$jout_dir/job_pid_trace.log"

      if [[ -n "$pids" ]]; then
        pid_csv="$(tr ' ' ',' <<<"$pids" | sed 's/^,*//; s/,*$//; s/,,*/,/g')"

        cpu_line="$(ps -p "$pid_csv" -o pcpu=,pmem=,rss= 2>/dev/null | awk '
          NF >= 3 {cpu += $1; pmem += $2; rss += $3; n++}
          END {
            if (n > 0) printf "%.2f %.2f %.0f %d", cpu, pmem, rss, n;
            else printf "NA NA NA 0";
          }
        ')"
        echo "$ts $cpu_line" >> "$jout_dir/job_cpu_trace.log"

        io_line="$(awk -v pids="$pids" '
          BEGIN {split(pids, a, " "); readb = 0; writeb = 0;}
          function parse_io(path,   line, key, val) {
            while ((getline line < path) > 0) {
              split(line, kv, /:[[:space:]]*/);
              key = kv[1]; val = kv[2] + 0;
              if (key == "read_bytes") readb += val;
              if (key == "write_bytes") writeb += val;
            }
            close(path);
          }
          END {
            for (i in a) {
              if (a[i] ~ /^[0-9]+$/) {
                path = "/proc/" a[i] "/io";
                if ((getline tmp < path) > 0) {
                  close(path);
                  parse_io(path);
                }
              }
            }
            printf "%lld %lld", readb, writeb;
          }
        ')"
        echo "$ts $io_line" >> "$jout_dir/job_io_trace.log"
      else
        echo "$ts NA NA NA 0" >> "$jout_dir/job_cpu_trace.log"
        echo "$ts 0 0" >> "$jout_dir/job_io_trace.log"
      fi

      t=$((t + 1))
      sleep 1
    done
  }

  echo "[INFO] Sampler node: $(hostname)"
  echo "[INFO] Start time  : $(date)"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits > "$out_dir/gpu_snapshot_begin.csv" 2>/dev/null || true
    # Use an explicit sample count so the run length is deterministic.
    (timeout "$((duration + 30))" nvidia-smi dmon -s pucvmet -d 1 -c "$duration" > "$out_dir/dmon.log" 2>&1) &
    p1=$!
    (timeout "$((duration + 30))" nvidia-smi pmon -s um -d 1 -c "$duration" > "$out_dir/pmon.log" 2>&1) &
    p2=$!
  else
    echo "[WARN] nvidia-smi not found on sampler node." | tee "$out_dir/gpu_warn.log"
    p1=""
    p2=""
  fi

  if command -v mpstat >/dev/null 2>&1; then
    (timeout "$duration" mpstat -P ALL 1 > "$out_dir/mpstat.log" 2>&1) &
    p3=$!
  else
    echo "[WARN] mpstat not found (install sysstat)." | tee "$out_dir/mpstat_warn.log"
    p3=""
  fi

  if command -v iostat >/dev/null 2>&1; then
    (timeout "$duration" iostat -x 1 > "$out_dir/iostat.log" 2>&1) &
    p4=$!
  else
    echo "[WARN] iostat not found (install sysstat)." | tee "$out_dir/iostat_warn.log"
    p4=""
  fi

  # Fallback path: collect CPU/IO signals via vmstat when sysstat tools are unavailable.
  if [[ -z "$p3" || -z "$p4" ]]; then
    if command -v vmstat >/dev/null 2>&1; then
      (timeout "$duration" vmstat 1 > "$out_dir/vmstat.log" 2>&1) &
      p5=$!
    else
      echo "[WARN] vmstat not found; CPU/IO fallback metrics unavailable." | tee "$out_dir/vmstat_warn.log"
      p5=""
    fi
  else
    p5=""
  fi

  (sample_job_proc_block "$out_dir" "$duration" "$job_user") &
  p6=$!

  for pid in "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"; do
    if [[ -n "${pid:-}" ]]; then
      wait "$pid" || true
    fi
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits > "$out_dir/gpu_snapshot_end.csv" 2>/dev/null || true
  fi

  echo "[INFO] End time    : $(date)"
}

execute_on_target() {
  local target_host="$1"
  local out_dir="$2"
  local duration="$3"
  local job_user="$4"

  local local_host
  local_host="$(hostname -s)"

  if [[ "$local_host" == "$target_host" ]] || [[ "$(hostname)" == "$target_host" ]]; then
    echo "[INFO] Already on target host: $target_host"
    run_sampler_block "$out_dir" "$duration" "$job_user"
    return 0
  fi

  mkdir -p "$out_dir"

  if command -v ssh >/dev/null 2>&1 && ssh -o BatchMode=yes -o ConnectTimeout=5 "$target_host" "hostname" >/dev/null 2>&1; then
    echo "[INFO] Running sampler through ssh on $target_host"
    ssh -o BatchMode=yes "$target_host" "OUT_DIR='$PWD/$out_dir' DURATION='$duration' JOB_USER='$job_user' bash -s" <<'REMOTE'
set -euo pipefail

out_dir="${OUT_DIR}"
duration="${DURATION}"
job_user="${JOB_USER}"
jpattern='python|torchrun|run_training_full.py|run_training.py|run_training_simscale.py|run_training_full_simscale.py'

mkdir -p "$out_dir"

echo "[INFO] Sampler node: $(hostname)"
echo "[INFO] Start time  : $(date)"

# Job-scoped CPU/IO tracing loop
job_trace_loop() {
  : > "$out_dir/job_pid_trace.log"
  : > "$out_dir/job_cpu_trace.log"
  : > "$out_dir/job_io_trace.log"

  local t=0
  while [[ "$t" -lt "$duration" ]]; do
    local ts pids pid_csv cpu_line io_line
    ts="$(date +%s)"
    pids="$(pgrep -u "$job_user" -f "$jpattern" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+$//' || true)"
    echo "$ts|$pids" >> "$out_dir/job_pid_trace.log"

    if [[ -n "$pids" ]]; then
      pid_csv="$(tr ' ' ',' <<<"$pids" | sed 's/^,*//; s/,*$//; s/,,*/,/g')"

      cpu_line="$(ps -p "$pid_csv" -o pcpu=,pmem=,rss= 2>/dev/null | awk '
        NF >= 3 {cpu += $1; pmem += $2; rss += $3; n++}
        END {
          if (n > 0) printf "%.2f %.2f %.0f %d", cpu, pmem, rss, n;
          else printf "NA NA NA 0";
        }
      ')"
      echo "$ts $cpu_line" >> "$out_dir/job_cpu_trace.log"

      io_line="$(awk -v pids="$pids" '
        BEGIN {split(pids, a, " "); readb = 0; writeb = 0;}
        function parse_io(path,   line, key, val) {
          while ((getline line < path) > 0) {
            split(line, kv, /:[[:space:]]*/);
            key = kv[1]; val = kv[2] + 0;
            if (key == "read_bytes") readb += val;
            if (key == "write_bytes") writeb += val;
          }
          close(path);
        }
        END {
          for (i in a) {
            if (a[i] ~ /^[0-9]+$/) {
              path = "/proc/" a[i] "/io";
              if ((getline tmp < path) > 0) {
                close(path);
                parse_io(path);
              }
            }
          }
          printf "%lld %lld", readb, writeb;
        }
      ')"
      echo "$ts $io_line" >> "$out_dir/job_io_trace.log"
    else
      echo "$ts NA NA NA 0" >> "$out_dir/job_cpu_trace.log"
      echo "$ts 0 0" >> "$out_dir/job_io_trace.log"
    fi

    t=$((t + 1))
    sleep 1
  done
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits > "$out_dir/gpu_snapshot_begin.csv" 2>/dev/null || true
  (timeout "$((duration + 30))" nvidia-smi dmon -s pucvmet -d 1 -c "$duration" > "$out_dir/dmon.log" 2>&1) &
  p1=$!
  (timeout "$((duration + 30))" nvidia-smi pmon -s um -d 1 -c "$duration" > "$out_dir/pmon.log" 2>&1) &
  p2=$!
else
  echo "[WARN] nvidia-smi not found on sampler node." | tee "$out_dir/gpu_warn.log"
  p1=""
  p2=""
fi

if command -v mpstat >/dev/null 2>&1; then
  (timeout "$duration" mpstat -P ALL 1 > "$out_dir/mpstat.log" 2>&1) &
  p3=$!
else
  echo "[WARN] mpstat not found (install sysstat)." | tee "$out_dir/mpstat_warn.log"
  p3=""
fi

if command -v iostat >/dev/null 2>&1; then
  (timeout "$duration" iostat -x 1 > "$out_dir/iostat.log" 2>&1) &
  p4=$!
else
  echo "[WARN] iostat not found (install sysstat)." | tee "$out_dir/iostat_warn.log"
  p4=""
fi

if [[ -z "$p3" || -z "$p4" ]]; then
  if command -v vmstat >/dev/null 2>&1; then
    (timeout "$duration" vmstat 1 > "$out_dir/vmstat.log" 2>&1) &
    p5=$!
  else
    echo "[WARN] vmstat not found; CPU/IO fallback metrics unavailable." | tee "$out_dir/vmstat_warn.log"
    p5=""
  fi
else
  p5=""
fi

(job_trace_loop) &
p6=$!

for pid in "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"; do
  if [[ -n "${pid:-}" ]]; then
    wait "$pid" || true
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits > "$out_dir/gpu_snapshot_end.csv" 2>/dev/null || true
fi

echo "[INFO] End time    : $(date)"
REMOTE
    return 0
  fi

  echo "[ERROR] Cannot execute on target host $target_host."
  echo "        Tried: local, ssh."
  return 1
}

summarize_results() {
  local out_dir="$1"
  local summary_file="$out_dir/summary.txt"

  local gpu_avg="NA" gpu_peak="NA" gpu_imbalance="NA"
  local cpu_iowait_avg="NA" cpu_idle_avg="NA"
  local disk_util_peak="NA"
  local vm_bi_avg="NA" vm_bo_avg="NA"
  local job_gpu_avg="NA" job_gpu_peak="NA" job_gpu_gpus="NA"
  local job_cpu_avg="NA" job_cpu_peak="NA" job_rss_avg_mb="NA" job_proc_avg="NA"
  local job_read_avg_mbps="NA" job_write_avg_mbps="NA"

  local job_pid_set_file="$out_dir/job_pid_set.txt"
  if [[ -f "$out_dir/job_pid_trace.log" ]]; then
    awk -F'|' '{if (NF >= 2) print $2}' "$out_dir/job_pid_trace.log" \
      | tr ' ' '\n' | awk '/^[0-9]+$/ {print $1}' | sort -u > "$job_pid_set_file"
  fi

  if [[ -f "$out_dir/pmon.log" ]]; then
    gpu_avg="$(awk '
      ($1 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/) {sum += $4; n++}
      END {if (n > 0) printf "%.1f", sum / n; else print "NA"}
    ' "$out_dir/pmon.log")"

    gpu_peak="$(awk '
      ($1 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/) {if ($4 > max) max = $4; seen = 1}
      END {if (seen) printf "%.1f", max; else print "NA"}
    ' "$out_dir/pmon.log")"

    gpu_imbalance="$(awk '
      ($1 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/) {sum[$1] += $4; cnt[$1]++}
      END {
        min = -1; max = -1; seen = 0;
        for (g in sum) {
          avg = sum[g] / cnt[g];
          if (min < 0 || avg < min) min = avg;
          if (max < 0 || avg > max) max = avg;
          seen = 1;
        }
        if (!seen) print "NA";
        else printf "%.1f", (max - min);
      }
    ' "$out_dir/pmon.log")"

    if [[ -s "$job_pid_set_file" ]]; then
      job_gpu_avg="$(awk 'NR == FNR {pid[$1] = 1; next}
        ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && ($2 in pid)) {sum += $4; n++}
        END {if (n > 0) printf "%.1f", sum / n; else print "NA"}
      ' "$job_pid_set_file" "$out_dir/pmon.log")"

      job_gpu_peak="$(awk 'NR == FNR {pid[$1] = 1; next}
        ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+$/ && ($2 in pid)) {if ($4 > max) max = $4; seen = 1}
        END {if (seen) printf "%.1f", max; else print "NA"}
      ' "$job_pid_set_file" "$out_dir/pmon.log")"

      job_gpu_gpus="$(awk 'NR == FNR {pid[$1] = 1; next}
        ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && ($2 in pid)) {g[$1] = 1}
        END {c = 0; for (k in g) c++; if (c > 0) print c; else print "NA"}
      ' "$job_pid_set_file" "$out_dir/pmon.log")"
    fi
  fi

  if [[ -f "$out_dir/job_cpu_trace.log" ]]; then
    job_cpu_avg="$(awk '$2 != "NA" {sum += $2; n++} END {if (n > 0) printf "%.2f", sum / n; else print "NA"}' "$out_dir/job_cpu_trace.log")"
    job_cpu_peak="$(awk '$2 != "NA" {if ($2 > max) max = $2; seen = 1} END {if (seen) printf "%.2f", max; else print "NA"}' "$out_dir/job_cpu_trace.log")"
    job_rss_avg_mb="$(awk '$4 != "NA" {sum += $4 / 1024; n++} END {if (n > 0) printf "%.1f", sum / n; else print "NA"}' "$out_dir/job_cpu_trace.log")"
    job_proc_avg="$(awk '$5 ~ /^[0-9]+$/ {sum += $5; n++} END {if (n > 0) printf "%.1f", sum / n; else print "NA"}' "$out_dir/job_cpu_trace.log")"
  fi

  if [[ -f "$out_dir/job_io_trace.log" ]]; then
    job_read_avg_mbps="$(awk '
      NR == 1 {t0 = $1; r0 = $2; next}
      {tn = $1; rn = $2}
      END {
        dt = tn - t0;
        if (NR > 1 && dt > 0) printf "%.2f", (rn - r0) / dt / 1024 / 1024;
        else print "NA";
      }
    ' "$out_dir/job_io_trace.log")"

    job_write_avg_mbps="$(awk '
      NR == 1 {t0 = $1; w0 = $3; next}
      {tn = $1; wn = $3}
      END {
        dt = tn - t0;
        if (NR > 1 && dt > 0) printf "%.2f", (wn - w0) / dt / 1024 / 1024;
        else print "NA";
      }
    ' "$out_dir/job_io_trace.log")"
  fi

  if [[ -f "$out_dir/mpstat.log" ]]; then
    cpu_iowait_avg="$(awk '
      /%iowait/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "CPU") cpu_col = i;
          if ($i == "%iowait") io_col = i;
          if ($i == "%idle") idle_col = i;
        }
        next;
      }
      cpu_col && io_col && idle_col {
        if ($(cpu_col) == "all" && $(io_col) ~ /^[0-9.]+$/ && $(idle_col) ~ /^[0-9.]+$/) {
          io_sum += $(io_col); io_n++;
          idle_sum += $(idle_col); idle_n++;
        }
      }
      END {
        if (io_n > 0) printf "%.2f", io_sum / io_n;
        else print "NA";
      }
    ' "$out_dir/mpstat.log")"

    cpu_idle_avg="$(awk '
      /%iowait/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "CPU") cpu_col = i;
          if ($i == "%idle") idle_col = i;
        }
        next;
      }
      cpu_col && idle_col {
        if ($(cpu_col) == "all" && $(idle_col) ~ /^[0-9.]+$/) {
          idle_sum += $(idle_col); idle_n++;
        }
      }
      END {
        if (idle_n > 0) printf "%.2f", idle_sum / idle_n;
        else print "NA";
      }
    ' "$out_dir/mpstat.log")"
  fi

  if [[ -f "$out_dir/iostat.log" ]]; then
    disk_util_peak="$(awk '
      /%util/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "%util") util_col = i;
        }
        next;
      }
      util_col && NF >= util_col && $1 !~ /^(Linux|avg-cpu:|Device)$/ {
        if ($(util_col) ~ /^[0-9.]+$/) {
          if ($(util_col) > max) max = $(util_col);
          seen = 1;
        }
      }
      END {
        if (seen) printf "%.1f", max;
        else print "NA";
      }
    ' "$out_dir/iostat.log")"
  fi

  if [[ "$cpu_iowait_avg" == "NA" || "$cpu_idle_avg" == "NA" || "$disk_util_peak" == "NA" ]]; then
    if [[ -f "$out_dir/vmstat.log" ]]; then
      # vmstat columns (typical): ... bi bo in cs us sy id wa st
      # Skip first two lines and parse only numeric rows.
      if [[ "$cpu_iowait_avg" == "NA" ]]; then
        cpu_iowait_avg="$(awk '
          NR > 2 && $0 ~ /^[[:space:][:digit:]-]/ && NF >= 17 {
            wa = $(NF-1);
            if (wa ~ /^-?[0-9.]+$/) {sum += wa; n++}
          }
          END {if (n > 0) printf "%.2f", sum / n; else print "NA"}
        ' "$out_dir/vmstat.log")"
      fi

      if [[ "$cpu_idle_avg" == "NA" ]]; then
        cpu_idle_avg="$(awk '
          NR > 2 && $0 ~ /^[[:space:][:digit:]-]/ && NF >= 17 {
            id = $(NF-2);
            if (id ~ /^-?[0-9.]+$/) {sum += id; n++}
          }
          END {if (n > 0) printf "%.2f", sum / n; else print "NA"}
        ' "$out_dir/vmstat.log")"
      fi

      if [[ "$disk_util_peak" == "NA" ]]; then
        vm_bi_avg="$(awk '
          NR > 2 && $0 ~ /^[[:space:][:digit:]-]/ && NF >= 17 {
            bi = $9;
            if (bi ~ /^-?[0-9.]+$/) {sum += bi; n++}
          }
          END {if (n > 0) printf "%.2f", sum / n; else print "NA"}
        ' "$out_dir/vmstat.log")"

        vm_bo_avg="$(awk '
          NR > 2 && $0 ~ /^[[:space:][:digit:]-]/ && NF >= 17 {
            bo = $10;
            if (bo ~ /^-?[0-9.]+$/) {sum += bo; n++}
          }
          END {if (n > 0) printf "%.2f", sum / n; else print "NA"}
        ' "$out_dir/vmstat.log")"
      fi
    fi
  fi

  {
    echo "=== Job Perf Summary ==="
    echo "job_id: $JOB_ID"
    echo "target_host: $TARGET_HOST"
    echo "duration_s: $DURATION"
    echo
    echo "gpu_avg_sm_util_from_pmon: $gpu_avg"
    echo "gpu_peak_sm_util_from_pmon: $gpu_peak"
    echo "gpu_sm_imbalance_delta: $gpu_imbalance"
    echo "job_gpu_avg_sm_util_from_pmon: $job_gpu_avg"
    echo "job_gpu_peak_sm_util_from_pmon: $job_gpu_peak"
    echo "job_gpu_active_count: $job_gpu_gpus"
    echo "cpu_iowait_avg_percent: $cpu_iowait_avg"
    echo "cpu_idle_avg_percent: $cpu_idle_avg"
    echo "disk_peak_util_percent: $disk_util_peak"
    echo "job_cpu_sum_avg_percent: $job_cpu_avg"
    echo "job_cpu_sum_peak_percent: $job_cpu_peak"
    echo "job_rss_avg_mb: $job_rss_avg_mb"
    echo "job_proc_avg_count: $job_proc_avg"
    echo "job_read_avg_mbps: $job_read_avg_mbps"
    echo "job_write_avg_mbps: $job_write_avg_mbps"
    if [[ -f "$out_dir/vmstat.log" ]]; then
      echo "vmstat_bi_avg_blocks_per_sec: $vm_bi_avg"
      echo "vmstat_bo_avg_blocks_per_sec: $vm_bo_avg"
    fi
    echo
    echo "=== Bottleneck Hints ==="

    if [[ "$job_gpu_avg" != "NA" ]]; then
      awk -v g="$job_gpu_avg" 'BEGIN {
        if (g < 80) print "- GPU average SM util < 80%: likely input pipeline / sync bottleneck.";
        else if (g < 92) print "- GPU average SM util is moderate: check dataloader and DDP sync overlap.";
        else print "- GPU average SM util is high: compute path likely saturated.";
      }'
    elif [[ "$gpu_avg" != "NA" ]]; then
      awk -v g="$gpu_avg" 'BEGIN {
        if (g < 80) print "- Node-wide GPU average SM util < 80%; job PID matching may be incomplete.";
        else if (g < 92) print "- Node-wide GPU util is moderate; verify job PID capture for scoped diagnosis.";
        else print "- Node-wide GPU util is high; verify job PID capture for scoped diagnosis.";
      }'
    else
      echo "- GPU utilization not parsed (check pmon.log)."
    fi

    if [[ "$gpu_imbalance" != "NA" ]]; then
      awk -v d="$gpu_imbalance" 'BEGIN {
        if (d > 15) print "- Large cross-GPU util imbalance (>15%): investigate DDP load imbalance / stragglers.";
        else print "- Cross-GPU util imbalance looks acceptable.";
      }'
    fi

    if [[ "$cpu_iowait_avg" != "NA" ]]; then
      awk -v iow="$cpu_iowait_avg" 'BEGIN {
        if (iow > 10) print "- CPU iowait > 10%: storage I/O likely limiting throughput.";
        else print "- CPU iowait is low: disk wait is likely not the primary bottleneck.";
      }'
    fi

    if [[ "$job_cpu_avg" != "NA" ]]; then
      awk -v c="$job_cpu_avg" 'BEGIN {
        if (c < 400) print "- Job CPU sum is relatively low (<400%): preprocessing parallelism may be under-utilized.";
        else print "- Job CPU sum indicates active host-side processing.";
      }'
    fi

    if [[ "$disk_util_peak" != "NA" ]]; then
      awk -v u="$disk_util_peak" 'BEGIN {
        if (u > 85) print "- Disk peak util > 85%: possible storage saturation windows.";
        else print "- Disk peak util does not indicate strong saturation.";
      }'
    elif [[ "$vm_bi_avg" != "NA" || "$vm_bo_avg" != "NA" ]]; then
      echo "- Disk %util unavailable (no iostat). Using vmstat bi/bo as I/O activity proxy."
    fi

    echo "- Next step: correlate this with train-step p50/p95 latency and val epoch duration."
  } | tee "$summary_file"

  echo "[INFO] Summary written to: $summary_file"
}

# Run
execute_on_target "$TARGET_HOST" "$OUT_DIR" "$DURATION" "$JOB_USER"
summarize_results "$OUT_DIR"

echo "[DONE] Logs are in: $OUT_DIR"
