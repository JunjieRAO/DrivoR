#!/usr/bin/env bash
#
# gpu_queue_status.sh — Summarize GPU capacity per LSF queue.
#
# For each GPU queue it reports:
#   TOTAL : number of GPUs backing the queue (from its host group)
#   USED  : GPUs currently running >=1 job  (bhosts -gpu, NJOBS>0)
#   FREE  : TOTAL - USED
#   UTIL% : USED / TOTAL
#   PEND  : pending (queued) jobs on the queue   (bqueues PEND)
#   RUN   : running jobs on the queue            (bqueues RUN)
#
# It also reports, for every advance reservation you can REALLY use (auto-detected
# by checking your LSF usergroup membership via bugroup — NOT unix groups, which
# can differ), a per-node breakdown of GPU usage plus the reservation's reserved
# CPU-slot usage. Use '-r <rsv>' to view a specific reservation you don't own.
#
# Usage:
#   ./gpu_queue_status.sh                      # all GPU queues + your reservations
#   ./gpu_queue_status.sh batch_a100 batch_h200
#   ./gpu_queue_status.sh -a                   # include queues with 0 GPUs too
#   ./gpu_queue_status.sh -r resv_cr_tfx_h200  # force a specific reservation id
#   ./gpu_queue_status.sh -R                   # skip the reservation report
#   ./gpu_queue_status.sh -h                   # help
#
set -o pipefail

show_all=0
no_rsv=0
rsv_override=()
args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -a|--all)    show_all=1 ;;
    -R|--no-rsv) no_rsv=1 ;;
    -r|--rsv)    shift; [ -n "${1:-}" ] && rsv_override+=("$1") ;;
    --rsv=*)     rsv_override+=("${1#*=}") ;;
    -h|--help)   grep '^#' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
    -*)          echo "Unknown option: $1" >&2; exit 2 ;;
    *)           args+=("$1") ;;
  esac
  shift
done

command -v bqueues >/dev/null 2>&1 || { echo "ERROR: 'bqueues' not found (LSF required)" >&2; exit 1; }
command -v bhosts  >/dev/null 2>&1 || { echo "ERROR: 'bhosts' not found (LSF required)"  >&2; exit 1; }

# --- pending/running jobs per queue + ordered queue list (single bqueues call) ---
declare -A Q_PEND Q_RUN Q_STAT
QALL=()
while IFS='|' read -r q pend run st; do
  [ -z "$q" ] && continue
  Q_PEND["$q"]="$pend"; Q_RUN["$q"]="$run"; Q_STAT["$q"]="$st"
  QALL+=("$q")
done < <(bqueues 2>/dev/null | awk 'NR>1 && NF>=11 {print $1"|"$9"|"$10"|"$3}')

# --- queues to display ---
if [ "${#args[@]}" -gt 0 ]; then
  QUEUES=("${args[@]}")
else
  QUEUES=("${QALL[@]}")
fi

# --- helpers ---
# queue -> space separated host groups (trailing '/' stripped)
hostgroups_of() {
  bqueues -l "$1" 2>/dev/null \
    | awk '/^HOSTS:/{for(i=2;i<=NF;i++){g=$i; sub(/\/+$/,"",g); if(g!="") printf "%s ", g}}'
}

# reads `bhosts -gpu` output on stdin -> "total used model"
count_gpus() {
  awk '
    /HOST_NAME/ { next }
    {
      if      (NF==9) { nj=$6; md=$3 }   # host row:  HOST GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV
      else if (NF==8) { nj=$5; md=$2 }   # cont. row:      GPU_ID MODEL MUSED MRSV NJOBS RUN SUSP RSV
      else            { next }
      tot++
      if (nj+0 > 0) used++
      if (fm=="") fm=md
    }
    END { printf "%d %d %s", tot+0, used+0, (fm==""?"-":fm) }'
}

declare -A GPU_CACHE   # host-group-string -> "total used model"
gpu_counts() {         # args: host groups; echoes "total used model"
  local groups; groups=$(echo "$*" | xargs)
  local key="${groups:-__none__}"
  if [ -n "${GPU_CACHE[$key]+x}" ]; then echo "${GPU_CACHE[$key]}"; return; fi
  local res
  if [ "$key" = "__none__" ]; then
    res="0 0 -"
  elif [[ " $groups " == *" all "* ]]; then
    res=$(bhosts -gpu 2>/dev/null | count_gpus)
  else
    res=$(bhosts -gpu $groups 2>/dev/null | count_gpus)
  fi
  GPU_CACHE[$key]="$res"
  echo "$res"
}

# --- advance reservation helpers ---

# Authorized principals (owner user/group + allowed usergroups) of a reservation.
rsv_principals() {
  brsvs -w "$1" 2>/dev/null | awk -v rid="$1" '
    { for (i=1;i<=NF;i++) {
        t=$i
        if (t ~ /:/) continue                     # host:slots token
        if (t ~ /^[0-9]+(\/[0-9]+)?$/) continue    # numbers / slot counts
        if (t == "/") continue                     # bare slash (wide format)
        if (t ~ /^[0-9]{4}\//) continue            # time window
        if (t ~ /^(RSVID|TYPE|USER|NCPUS|RSV_HOSTS|TIME_WINDOW|user)$/) continue
        if (t == rid) continue                     # the reservation id itself
        print t
      } }' | sort -u
}

# List advance reservations the current user is REALLY authorized to use.
# LSF authorizes '-U' by its own usergroup membership (bugroup), which can differ
# from unix groups (id -Gn) even when a group of the same name exists. So we must
# check bugroup membership of each reservation's authorized principals, not id -Gn.
detect_my_reservations() {
  command -v brsvs >/dev/null 2>&1 || return 0
  local me r p; me=$(whoami)
  while read -r r; do
    [ -z "$r" ] && continue
    while read -r p; do
      [ -z "$p" ] && continue
      if [ "$p" = "$me" ]; then echo "$r"; break; fi
      if command -v bugroup >/dev/null 2>&1 \
         && bugroup -w "$p" 2>/dev/null | tr ' ' '\n' | grep -qxF "$me"; then
        echo "$r"; break
      fi
    done < <(rsv_principals "$r")
  done < <(brsvs -w 2>/dev/null | awk 'NR>1 && /^[^[:space:]]/{print $1}' | sort -u)
}

# Per-host GPU totals for the given hosts -> lines: "host total used model".
gpu_per_host() {
  [ "$#" -gt 0 ] || return 0
  bhosts -gpu "$@" 2>/dev/null | awk '
    /HOST_NAME/ { next }
    {
      if      (NF==9) { host=$1; md=$3; nj=$6 }   # host row (hostname in $1)
      else if (NF==8) { nj=$5 }                   # continuation row (same host)
      else            { next }
      if (host=="") next
      if (!(host in seen)) { seen[host]=1; order[++n]=host; mod[host]=md }
      tot[host]++
      if (nj+0 > 0) used[host]++
    }
    END { for (i=1;i<=n;i++){ h=order[i]; printf "%s %d %d %s\n", h, tot[h], used[h], mod[h] } }'
}

# Print per-node GPU usage + reserved-slot usage for each reservation id.
reservation_report() {
  [ "$#" -gt 0 ] || return 0
  command -v brsvs >/dev/null 2>&1 || { echo; echo "(brsvs not available; skipping reservation report)"; return 0; }
  local rid tok h su status t u md f gtot gused gfree nodes
  declare -A SLOT
  for rid in "$@"; do
    SLOT=()
    while read -r tok; do
      [ -z "$tok" ] && continue
      h=${tok%%:*}; su=${tok#*:}
      SLOT["$h"]="$su"
    done < <(brsvs -w "$rid" 2>/dev/null | tr -s ' \t' '  ' \
              | grep -oE 'rng-dl01-[a-z0-9]+:[0-9]+ +/ +[0-9]+' | sed 's/ *\/ */\//')
    if [ "${#SLOT[@]}" -eq 0 ]; then
      printf "\nReservation %-20s : no hosts (not accessible / empty)\n" "$rid"
      continue
    fi
    status=$(brsvs -l "$rid" 2>/dev/null | awk -F': ' '/Reservation Status/{print $2; exit}')
    printf "\nReservation %s  [%s]  @ %s\n" "$rid" "${status:-?}" "$(date '+%Y-%m-%d %H:%M:%S')"
    printf "%-18s %-12s %5s %5s %5s  %-13s\n" "NODE" "GPU_MODEL" "GPUS" "USED" "FREE" "RSV_SLOTS"
    echo "$line"
    gtot=0; gused=0; nodes=0
    while read -r h t u md; do
      [ -z "$h" ] && continue
      f=$((t - u))
      gtot=$((gtot + t)); gused=$((gused + u)); nodes=$((nodes + 1))
      printf "%-18s %-12s %5d %5d %5d  %-13s\n" "$h" "${md:--}" "$t" "$u" "$f" "${SLOT[$h]:--}"
    done < <(gpu_per_host "${!SLOT[@]}" | sort -V)
    gfree=$((gtot - gused))
    echo "$line"
    printf "%-18s %-12s %5d %5d %5d\n" "TOTAL ($nodes nodes)" "-" "$gtot" "$gused" "$gfree"
  done
}

# --- table ---
line="-------------------------------------------------------------------------------"
printf "GPU queue status @ %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "%-16s %-16s %5s %5s %5s %6s %7s %6s\n" "QUEUE" "GPU_MODEL" "TOTAL" "USED" "FREE" "UTIL%" "PEND" "RUN"
echo "$line"

for q in "${QUEUES[@]}"; do
  if [ "${#args[@]}" -gt 0 ] && [ -z "${Q_STAT[$q]+x}" ]; then
    printf "%-16s %-16s %5s %5s %5s %6s %7s %6s\n" "$q" "(unknown queue)" "-" "-" "-" "-" "-" "-"
    continue
  fi
  groups=$(hostgroups_of "$q")
  read -r tot used model < <(gpu_counts "$groups")
  tot=${tot:-0}; used=${used:-0}
  # in auto mode, hide queues with no GPUs unless -a
  if [ "$tot" -eq 0 ] && [ "$show_all" -eq 0 ] && [ "${#args[@]}" -eq 0 ]; then
    continue
  fi
  free=$((tot - used))
  if [ "$tot" -gt 0 ]; then
    util=$(awk -v u="$used" -v t="$tot" 'BEGIN{printf "%.0f%%", 100*u/t}')
  else
    util="-"
  fi
  printf "%-16s %-16s %5d %5d %5d %6s %7s %6s\n" \
    "$q" "$model" "$tot" "$used" "$free" "$util" "${Q_PEND[$q]:-0}" "${Q_RUN[$q]:-0}"
done

echo "$line"
# de-duplicated physical totals across every GPU host in the cluster
read -r ptot pused pmodel < <(bhosts -gpu 2>/dev/null | count_gpus)
ptot=${ptot:-0}; pused=${pused:-0}; pfree=$((ptot - pused))
putil=$(awk -v u="$pused" -v t="$ptot" 'BEGIN{if(t>0)printf "%.0f%%",100*u/t; else printf "-"}')
printf "%-16s %-16s %5d %5d %5d %6s\n" "PHYSICAL_TOTAL*" "(all GPU hosts)" "$ptot" "$pused" "$pfree" "$putil"
echo
echo "* PHYSICAL_TOTAL is de-duplicated across all GPU hosts. Per-queue rows may share the same"
echo "  nodes, so summing the queue rows can exceed the physical total."
echo "  USED counts GPUs running >=1 job (bhosts -gpu NJOBS>0); PEND/RUN are jobs (bqueues)."

# --- advance reservation report (nodes you can target via '#BSUB -U <rsv>') ---
if [ "$no_rsv" -eq 0 ]; then
  if [ "${#rsv_override[@]}" -gt 0 ]; then
    RSVS=("${rsv_override[@]}")
  else
    mapfile -t RSVS < <(detect_my_reservations)
  fi
  if [ "${#RSVS[@]}" -gt 0 ]; then
    reservation_report "${RSVS[@]}"
    echo
    echo "  Reservation rows: USED = GPUs on the node running >=1 job (any user); FREE = idle GPUs."
    echo "  On these CLOSED reservation nodes you claim idle GPUs by submitting with '#BSUB -U <rsv>'."
    echo "  RSV_SLOTS = reserved CPU slots used/total for the reservation on that node (brsvs)."
  fi
fi
