#!/usr/bin/env bash
# Runs the evaluation. From the repo root:
#   DATA=data/data ./scripts/eval_all.sh             # all stages
#   ./scripts/eval_all.sh write report               # some stages
#   QUICK=1 ./scripts/eval_all.sh                    # synthetic data, small sizes
# Stages: selftest stash churn write writepir pir compare ypir spiral paired report
# Not in `all`: oppoint scale
set -euo pipefail
DATA=${DATA:-data/data}
OUT=${OUT:-results}
QUICK=${QUICK:-0}
STAGES=${*:-all}
export RUSTC_BOOTSTRAP=1
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
named() { [[ " $STAGES " == *" $1 "* ]]; }
rerun() { [[ ${FORCE:-0} == 1 ]] || named "$1"; }
want() {
  CUR=$1
  if named "$1"; then return 0; fi
  [[ " $STAGES " == *" all "* ]] || return 1
  if [[ ${FORCE:-0} != 1 && -f "$OUT/logs/stage_$1.done" ]]; then log "stage $1 done before, skipped"; return 1; fi
  return 0
}
done_stage() { touch "$OUT/logs/stage_$1.done"; }
CUR=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT/logs" "$OUT/design" "$OUT/stash" "$OUT/pir"
cargo build --release --bin uokvs_bench
B=./target/release/uokvs_bench

if [[ $QUICK == 1 ]]; then
  SRC=(--synthetic 3000,20000,40); N=1000000; MFS="2.0 1.333"; PIR_ROWS="16384"; DOC_ROWS="65536"; TRIALS=1
  QPIR=(--synthetic 1000,5000,20)
  QCAP=(--val-cap-frac 0.5)
  DEC_EL=20000
else
  if [[ ! -f $DATA/keyword_file_map.json ]]; then echo "no $DATA/keyword_file_map.json (set DATA or QUICK=1)"; exit 1; fi
  SRC=(--data "$DATA"); N=24000000; MFS="2.0 1.5 1.333 1.111"; PIR_ROWS="131072 262144 524288"; DOC_ROWS="2097152 4194304"; TRIALS=2
  QPIR=(); QCAP=()
  DEC_EL=1000000
fi
CLIENTS=100,1000,10000,100000,1000000
PIR_APPENDS=${PIR_APPENDS:-200000}
PREFIX=(); (( PIR_APPENDS > 0 )) && PREFIX=(--max-appends $PIR_APPENDS)
COMMON=(--build-frac 0.8 --order fileid --del-frac 0.02 --upd-frac 0.02 --docdel-frac 0.002
        --clients $CLIENTS --search-every 20000 --search-sample 20 --snap-every 10000
        --summary "$OUT/design_summary.csv" "${QCAP[@]}")
design() { # label, extra args...
  local l=$1; shift
  if ! rerun "$CUR" && grep -q "^\[$l\] done:" "$OUT/logs/design_$l.log" 2>/dev/null; then
    log "design $l done before, skipped"; return 0
  fi
  log "design $l"
  if ! $B design "${SRC[@]}" "${COMMON[@]}" --label "$l" --out-prefix "$OUT/design/$l" "$@" > "$OUT/logs/design_$l.log" 2>&1; then
    echo "design $l FAILED; see $OUT/logs/design_$l.log" >&2; grep -m1 "panicked" -A1 "$OUT/logs/design_$l.log" >&2; return 1
  fi
  tail -10 "$OUT/logs/design_$l.log"
}
dry_caps() {
  local l=$1 what=$2; shift 2
  local f="$OUT/logs/caps_$l.txt"
  if rerun "$CUR" || [[ ! -s $f ]]; then
    mkdir -p "$OUT/dry"
    dry_run() {
      local sfx=$1; shift
      log "dry run for $l$sfx ($*)" >&2
      if ! $B design "${SRC[@]}" "${COMMON[@]}" --summary "$OUT/dry/summary_$l$sfx.csv" --label "dry_$l$sfx" \
           --out-prefix "$OUT/dry/$l$sfx" "${ARGS[@]}" "$@" > "$OUT/logs/dry_$l$sfx.log" 2>&1; then
        echo "dry run for $l$sfx failed; see $OUT/logs/dry_$l$sfx.log" >&2; tail -5 "$OUT/logs/dry_$l$sfx.log" >&2; return 1
      fi
    }
    val() { sed -n "s/^\[dry_$l$1\] $2 //p" "$OUT/logs/dry_$l$1.log"; }
    up() { awk -v m="$1" -v h="$2" -v a="$3" 'BEGIN { c = m * h; if (c > int(c)) c = int(c) + 1; printf "%d", c + a }'; }
    local ARGS=("$@") ex=() K=- P=- km pm sfx=""
    [[ $what == *K* ]] && ex+=(--k measure)
    [[ $what == *P* ]] && ex+=(--pend-cap measure)
    dry_run "" "${ex[@]}" || return 1
    if [[ $what == *K* ]]; then
      local kkey=k_max_ops_per_batch; [[ ${K_FROM:-max} == steady ]] && kkey=k_max_ops_steady
      km=$(val "" $kkey); [[ -n $km ]] || { echo "dry run for $l: no $kkey line" >&2; return 1; }
      K=$(up "$km" "${K_HEADROOM:-1.1}" 2)
      if [[ $what == *P* && ${K_FROM:-max} == steady ]]; then
        sfx=_k; dry_run "$sfx" --k "$K" --pend-cap measure || return 1
      fi
    fi
    if [[ $what == *P* ]]; then
      pm=$(val "$sfx" pend_max_at_compaction); [[ -n $pm ]] || { echo "dry run for $l: no pend_max_at_compaction line" >&2; return 1; }
      P=$(up "$pm" "${PEND_HEADROOM:-1.1}" 64)
    fi
    echo "$K $P" > "$f"
    log "dry run for $l: K $K (from ${km:--}), P-hat $P (from ${pm:--} pending postings at a log compaction)" >&2
  fi
  cat "$f"
}
pend_cap() { local l=$1; shift; local c; c=$(dry_caps "$l" P "$@") || return 1; echo "${c#* }"; }
k_cap() { local l=$1; shift; local c; c=$(dry_caps "$l" K "$@") || return 1; echo "${c% *}"; }

# correctness checks
if want selftest; then
  log "selftest"; $B selftest | tee "$OUT/logs/selftest.log" | tail -3
  done_stage selftest
fi

# stash vs load
if want stash; then
  MEM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 8)
  STASH_JOBS=${STASH_JOBS:-$(( MEM_GB / 10 < $(nproc) ? MEM_GB / 10 : $(nproc) ))}
  (( STASH_JOBS < 1 )) && STASH_JOBS=1
  JOBS="$OUT/logs/stash_jobs.txt"; : > "$JOBS"
  for mf in $MFS; do
    for ev in "rw 10" "bfs 20" "bfs 10"; do set -- $ev
      for d in paper legacy noinv; do
        for jf in 0.0 0.8; do
          f="$OUT/stash/${1}a${2}_${d}_mf${mf}_jf${jf}.csv"
          [[ -s $f ]] && continue
          echo "$B stash --n $N --mu-factor $mf --evict $1 --alpha $2 --disc $d --joint-frac $jf --checkpoints 50 --branch 2 --depth 64 --out $f" >> "$JOBS"
        done
      done
    done
  done
  for ev in "rw 10" "bfs 20"; do set -- $ev
    for d in paper legacy; do
      f="$OUT/stash/p256_${1}a${2}_${d}.csv"; [[ -s $f ]] && continue
      echo "$B stash --n $N --p 256 --mu-factor 2.0 --evict $1 --alpha $2 --disc $d --checkpoints 50 --out $f" >> "$JOBS"
    done
  done
  log "stash: $(wc -l < "$JOBS") runs, $STASH_JOBS at a time"
  xargs -P "$STASH_JOBS" -I{} bash -c '{} > /dev/null || { echo "FAILED: {}"; f=$(echo "{}" | sed "s/.*--out //"); rm -f "$f"; }' < "$JOBS"
  done_stage stash
fi

# stash under churn at constant load
if want churn; then
  mkdir -p "$OUT/churn"
  if [[ $QUICK == 1 ]]; then CN=100000; else CN=${CHURN_N:-1000000}; fi
  JOBS="$OUT/logs/churn_jobs.txt"; : > "$JOBS"
  for mf in 2.0 1.5; do
    for ev in "rw 10 paper" "rw 10 legacy" "bfs 20 paper" "bfs 10 paper"; do set -- $ev
      for cfg in "replace 0 0" "replace 0 $CN" "replace 1 0" "replace 1 $CN" "update 0 0" "update 0 $CN"; do
        set -- $1 $2 $3 $cfg
        f="$OUT/churn/${1}a${2}_${3}_mf${mf}_${4}_z${5}_rb${6}.csv"
        [[ -s $f ]] && continue
        echo "$B churn --n $CN --mu-factor $mf --evict $1 --alpha $2 --disc $3 --mode $4 --zipf $5 --rebuild-every $6 --ops-factor 4 --checkpoints 40 --out $f" >> "$JOBS"
      done
    done
  done
  NJ=${STASH_JOBS:-$(nproc)}
  log "churn: $(wc -l < "$JOBS") runs, $NJ at a time"
  xargs -P "$NJ" -I{} bash -c '{} > /dev/null || { echo "FAILED: {}"; f=$(echo "{}" | sed "s/.*--out //"); rm -f "$f"; }' < "$JOBS"
  done_stage churn
fi

# write path, no PIR (accounting only)
if want write; then
  design main           --policy batched --log-every 1000000 --major-every 1000000
  DEC=(--policy batched --log-every $DEC_EL --major-every 0 --mu-factor 2.2)
  PC=$(pend_cap main_decoupled "${DEC[@]}")
  design main_decoupled "${DEC[@]}" --pend-cap "$PC"
  AG=(--policy batched --age-appends 100000 --log-every 1000000 --major-every 1000000)
  KC=$(k_cap main_age_1e5 "${AG[@]}")
  design main_age_1e5 "${AG[@]}" --k "$KC"
  design main_a10       --policy batched --alpha 10
  design logonly        --policy rebuild --r 0
  for r in 100000 300000 1000000; do design rebuild_$r --policy rebuild --r $r; done
  design perop          --policy perop --theta 101 --major-every 0 --mu-factor 2.2 --search-every 100000
  for b in 1000 3000 30000; do design main_b$b --policy batched --b $b; done
  for th in 50 101; do design main_t$th --policy batched --theta $th; done
  design main_major300000 --policy batched --major-every 300000
  design main_major0 --policy batched --major-every 0 --mu-factor 2.2
  # RW needs a much larger stash to get through setup
  design main_rw        --policy batched --evict rw --sigma 8192
  design main_age       --policy batched --age 64
  design main_noreuse   --policy batched --no-reuse --mu-factor 2.6
  design docmap_cuckoo  --policy batched --doc-layout cuckoo
  design docmap_rows_4k --policy batched --doc-row 4096
  design logonly_bf05   --policy rebuild --r 0 --build-frac 0.5
  design main_bf05      --policy batched --build-frac 0.5
  done_stage write
fi

# operating point: age rule, no ValMap rebuild during the replay
if named oppoint; then
  CUR=oppoint
  for A in 100000 1000000; do
    OP=(--policy batched --age-appends $A --log-every 1000000 --major-every 0 --mu-factor 2.2)
    C=$(dry_caps main_op_$A KP "${OP[@]}") || exit 1
    design main_op_$A "${OP[@]}" --k "${C% *}" --pend-cap "${C#* }" || exit 1
  done
  done_stage oppoint
fi

# write path with FrodoPIR databases kept up to date
if want writepir; then
  DOCPIR=(--no-doc-pir); [[ ${DOC_PIR:-0} == 1 ]] && DOCPIR=()
  design pir_main       --policy batched --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  design pir_rebuild    --policy rebuild --r 100000 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  design pir_logonly    --policy rebuild --r 0 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  done_stage writepir
fi

# FrodoPIR fork microbenchmarks
if want pir; then
  for rows in $PIR_ROWS; do
    log "pir ValMap $rows x 8 KB"
    $B pir --rows $rows --elem-bytes 8192 --batches 16,64,256 --widths 8,16,32,64 \
       --update-rows 11,21,110,210,420 --trials $TRIALS --out "$OUT/pir/frodo_val_${rows}.csv"
  done
  for rows in $DOC_ROWS; do
    log "pir DocMap $rows x 1059 B"
    $B pir --rows $rows --elem-bytes 1059 --batches 25,100,400 --widths 8,32,64 \
       --update-rows 4,16 --trials $TRIALS --out "$OUT/pir/frodo_doc_${rows}.csv"
  done
  done_stage pir
fi

# fork vs upstream FrodoPIR
if want compare; then
  ( cd pir_compare && cargo build --release )
  : > "$OUT/pir_compare.txt"
  SMALLEST=$(echo $PIR_ROWS | tr ' ' '\n' | sort -n | head -1)
  for rows in $PIR_ROWS; do
    SKIP=(); [[ $rows -gt $SMALLEST && ${UPSTREAM_REBUILD_ALL:-0} != 1 ]] && SKIP=(--skip-upstream-rebuild)
    log "fork vs upstream $rows ${SKIP[*]}"
    ./pir_compare/target/release/pir_compare --rows $rows --elem-bytes 8192 --update-rows 11,110,210 \
       --batches 4,16,64,256 "${SKIP[@]}" --out "$OUT/pir/fork_vs_upstream_${rows}.csv" | tee -a "$OUT/pir_compare.txt"
  done
  done_stage compare
fi

# YPIR+SP
if want ypir; then
  log "ypir"; QUICK=$QUICK ./scripts/run_ypir.sh "$OUT/pir/ypir" 2>&1 | grep -E "^(impl|ok|ypir_fast)" | tee "$OUT/ypir.txt"
  done_stage ypir
fi

# Spiral: Blyss server vs spiral_fast
if want spiral; then
  ( cd spiral_fast && cargo build --release )
  SB=./spiral_fast/target/release/spiral_bench
  $SB check | grep -E "^ok" | tee "$OUT/spiral.txt"
  if [[ $QUICK == 1 ]]; then SHAPES="2,9,5 1,9,6"; else SHAPES="2,9,5 2,9,6 2,9,7 2,9,8 1,9,7 1,9,8"; fi
  for s in $SHAPES; do
    log "spiral $s"
    $SB bench $s --impl blyss --batches 1,8,32 --trials 2 | grep -E "^impl" | tee -a "$OUT/spiral.txt"
    $SB bench $s --impl fast  --batches 1,8,32,128 --trials 2 | grep -E "^impl" | tee -a "$OUT/spiral.txt"
  done
  if [[ $QUICK != 1 && ${SPIRAL_DOC:-1} == 1 ]]; then
    for s in 1,9,12 2,9,10; do
      log "spiral docmap $s"
      $SB bench $s --impl fast --batches 1,16,64 --trials 1 | grep -E "^impl" | sed 's/^impl=fast/impl=fast_docmap/' | tee -a "$OUT/spiral.txt" \
        || echo "spiral docmap $s failed (see above)" | tee -a "$OUT/spiral.txt"
    done
  fi
  done_stage spiral
fi

# log vs stateless clients, FrodoPIR/Spiral per map
if want paired; then
  design paired_log_only     --policy rebuild --r 0
  design paired_log_rebuild  --policy rebuild --r 100000
  design paired_log_batched  --policy batched
  for b in 1 10 100 1000; do
    design paired_stateless_b$b --stateless --b $b --search-every 100000
  done
  DOCPIR=(--no-doc-pir); [[ ${DOC_PIR:-1} == 1 ]] && DOCPIR=()
  [[ ${PAIRED_PIR:-1} == 1 ]] && for pv in spiral frodo; do
    design pir_stateless_${pv}_frodo --stateless --b 100 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
    design pir_log_${pv}_frodo --policy rebuild --r 100000 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
  done
  if [[ ${DOC_SPIRAL:-0} == 1 ]]; then
    design pir_stateless_spiral_spiral --stateless --b 100 --pir --val-pir spiral --doc-pir spiral \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
  fi
  done_stage paired
fi

# k copies of the corpus at equal log budgets
if named scale; then
  CUR=scale
  if [[ $QUICK == 1 ]]; then
    KS=${SCALE_KS:-"2 1"}; AS=${SCALE_AS:-"5000 20000"}; ASD=${SCALE_AS_DISJOINT:-"5000 20000"}; SB=${SCALE_B:-1000}; SAPP=${SCALE_APPENDS:-0}
  else
    KS=${SCALE_KS:-"16 4 1"}; AS=${SCALE_AS:-"100000 300000 1000000"}; ASD=${SCALE_AS_DISJOINT:-"100000 1000000"}; SB=${SCALE_B:-10000}; SAPP=${SCALE_APPENDS:-3000000}
  fi
  MODES=${SCALE_MODES:-"shared disjoint"}
  SUEL=${SCALE_UOKVS_EL:-A}
  MEM_GB=$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 8)
  SCALE_JOBS=${SCALE_JOBS:-$(( MEM_GB / 90 > 1 ? MEM_GB / 90 : 1 ))}
  SPRE=(); (( SAPP > 0 )) && SPRE=(--max-appends $SAPP)
  mkdir -p "$OUT/scale"; FAILS="$OUT/logs/scale_failed.txt"; : > "$FAILS"
  scale_job() { # static|uokvs mode k A
    local kind=$1 mode=$2 k=$3 A=$4 l="scale_$1_$2_k$3_A$4"
    local SUM=(--summary "$OUT/scale/summary_$l.csv")
    rm -f "$OUT/scale/summary_$l.csv" "$OUT/scale/summary_${l}_clients.csv"
    if [[ $kind == static ]]; then
      design "$l" --scale "$k" --scale-mode "$mode" --policy batched --k 0 --major-every "$A" --log-every "$A" "${SPRE[@]}" "${SUM[@]}" || return 1
    else
      local EL=$A; [[ $SUEL != A ]] && EL=$SUEL
      local U=(--scale "$k" --scale-mode "$mode" --policy batched --b "$SB" --age-appends "$A" --log-every "$EL" --major-every 0 "${SPRE[@]}")
      local c; c=$(dry_caps "$l" KP "${U[@]}") || return 1
      design "$l" "${U[@]}" --k "${c% *}" --pend-cap "${c#* }" "${SUM[@]}" || return 1
    fi
  }
  log "scale: modes {$MODES}, k in {$KS}, A in {$AS} (disjoint {$ASD}), B $SB, appends ${SAPP:-all}, $SCALE_JOBS jobs at a time"
  for k in $KS; do
    for mode in $MODES; do
      As=$AS; [[ $mode == disjoint ]] && As=$ASD
      for A in $As; do
        for kind in static uokvs; do
          while (( $(jobs -rp | wc -l) >= SCALE_JOBS )); do sleep 2; done
          ( scale_job $kind "$mode" "$k" "$A" || echo "scale_${kind}_${mode}_k${k}_A${A}" >> "$FAILS" ) &
        done
      done
    done
  done
  wait
  for suf in "" _clients; do
    tgt="$OUT/design_summary$suf.csv"
    for f in "$OUT"/scale/summary_scale_*"$suf".csv; do
      [[ $suf == "" && $f == *_clients.csv ]] && continue
      [[ -s $f ]] || continue
      if [[ ! -s $tgt ]]; then cat "$f" > "$tgt"
      elif [[ $(head -1 "$tgt") == "$(head -1 "$f")" ]]; then tail -n +2 "$f" >> "$tgt"
      else echo "note: $tgt has another column layout; scale rows left in $OUT/scale/"; break; fi
    done
  done
  if [[ -s $FAILS ]]; then echo "scale: FAILED runs: $(tr '\n' ' ' < "$FAILS")"; exit 1; fi
fi

if want report; then
  $B report --dir "$OUT"
fi
log "done; see $OUT/REPORT.md"
