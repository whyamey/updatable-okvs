#!/usr/bin/env bash
# The whole evaluation plan in one script (see EXPERIMENTS.md for what each
# stage measures and which paper item it answers). Run from the repository root:
#
#   DATA=/home/bfuller/MKSE/okvs_python/data/data ./scripts/eval_all.sh            # everything
#   ./scripts/eval_all.sh write pir report                                          # some stages
#   QUICK=1 ./scripts/eval_all.sh                                                   # small sizes, minutes
#   STASH_JOBS=16 PIR_APPENDS=200000 UPSTREAM_REBUILD_ALL=0 PAIRED_PIR=1             # knobs (defaults shown/derived)
#
# Stages: selftest stash churn write writepir pir compare ypir spiral paired report   (default: all); named only: oppoint scale
#         scale   (not part of `all`: ./scripts/eval_all.sh scale report; see stage 8)
# Restarting: with the default (all), finished work is skipped: design runs whose log
# has its "done" line, stages that left $OUT/logs/stage_<name>.done, and stash runs
# with output. Naming a stage reruns it; FORCE=1 reruns everything but the stash runs
# (delete $OUT/stash/*.csv to redo those). Rerun design stages after changing the cost model.
# Outputs: $OUT/ (CSV per run, design_summary*.csv, REPORT.md). Logs: $OUT/logs/.
set -euo pipefail
DATA=${DATA:-/home/bfuller/MKSE/okvs_python/data/data}
OUT=${OUT:-results}
QUICK=${QUICK:-0}
STAGES=${*:-all}
export RUSTC_BOOTSTRAP=1                       # frodo-pir uses portable_simd (or use nightly)
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
named() { [[ " $STAGES " == *" $1 "* ]]; }
rerun() { [[ ${FORCE:-0} == 1 ]] || named "$1"; }
# want STAGE: run it (and remember it as the current stage); with `all`, a stage
# that finished before (marker file) is skipped unless FORCE=1
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
  QPIR=(--synthetic 1000,5000,20)   # smaller replays for the real-PIR runs
  # the synthetic corpus has many one-chunk keywords: its live chunks reach ~0.47 mu
  # at a rebuild, above the default --val-cap of 0.46 mu (Enron: <= 0.44 mu);
  # 0.5 mu always suffices at --mu-factor 2 (mu counts the full corpus + 10%)
  QCAP=(--val-cap-frac 0.5)
  DEC_EL=20000                      # log compaction period of main_decoupled (the QUICK replay has ~140K appends)
else
  if [[ ! -f $DATA/keyword_file_map.json ]]; then echo "no $DATA/keyword_file_map.json (set DATA or QUICK=1)"; exit 1; fi
  SRC=(--data "$DATA"); N=24000000; MFS="2.0 1.5 1.333 1.111"; PIR_ROWS="131072 262144 524288"; DOC_ROWS="2097152 4194304"; TRIALS=2
  QPIR=(); QCAP=()
  DEC_EL=1000000
fi
CLIENTS=100,1000,10000,100000,1000000
# real-PIR replays measure server time per batch and per search; a prefix of
# the replay is enough (PIR_APPENDS=0 replays everything)
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
  # explicit status check: errexit is off when a caller runs this inside `|| ...`
  if ! $B design "${SRC[@]}" "${COMMON[@]}" --label "$l" --out-prefix "$OUT/design/$l" "$@" > "$OUT/logs/design_$l.log" 2>&1; then
    echo "design $l FAILED; see $OUT/logs/design_$l.log" >&2; grep -m1 "panicked" -A1 "$OUT/logs/design_$l.log" >&2; return 1
  fi
  tail -10 "$OUT/logs/design_$l.log"
}
# Dry runs that size public caps from the data, for a run given by its label and args:
#   K  (--k): the dry run with --k measure serves all due work in every batch;
#      the real run uses K = ceil(K_HEADROOM x most operations a batch needed) + 2
#      (K_HEADROOM default 1.1). --age-appends phases the deadlines per keyword, so
#      there is no burst at the first deadline after setup; K_FROM=steady (not the
#      default) sizes K from the most after the first two age windows instead (the
#      excess then overflows,
#      counted in k_overflow_batches; P̂ then comes from a second dry run with that K,
#      which has the real run's trajectory). Default K_FROM=max (no overflow);
#   P̂ (--pend-cap), for runs whose log compactions carry pending postings
#      (E_L < E_V or E_V = 0): the dry run with --pend-cap measure writes unpadded
#      snapshots; the real run uses ceil(PEND_HEADROOM x the most pending postings
#      a compaction carried) + 64 (PEND_HEADROOM default 1.1).
# Batches and snapshot padding have their own randomness, so the dry run and the
# real run follow the same trajectory (with K >= the dry-run maximum: no K-overflow).
# The summary reports both the caps and what was needed (k_max_ops_per_batch,
# pend_max_at_compaction). dry_caps LABEL K|P|KP ARGS... prints "K P" ("-" = not
# measured); cached in $OUT/logs/caps_<label>.txt, redone when the stage is named.
dry_caps() {
  local l=$1 what=$2; shift 2
  local f="$OUT/logs/caps_$l.txt"
  if rerun "$CUR" || [[ ! -s $f ]]; then
    mkdir -p "$OUT/dry"
    # one dry run: dry_run SUFFIX EXTRA-ARGS... (extra args come last and win)
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
      # with K below the dry run's maximum the real run overflows and its pending
      # postings differ: measure P̂ again with that K (same trajectory as the real run)
      if [[ $what == *P* && ${K_FROM:-max} == steady ]]; then
        sfx=_k; dry_run "$sfx" --k "$K" --pend-cap measure || return 1
      fi
    fi
    if [[ $what == *P* ]]; then
      pm=$(val "$sfx" pend_max_at_compaction); [[ -n $pm ]] || { echo "dry run for $l: no pend_max_at_compaction line" >&2; return 1; }
      P=$(up "$pm" "${PEND_HEADROOM:-1.1}" 64)
    fi
    echo "$K $P" > "$f"
    log "dry run for $l: K $K (from ${km:--}), P̂ $P (from ${pm:--} pending postings at a log compaction)" >&2
  fi
  cat "$f"
}
pend_cap() { local l=$1; shift; local c; c=$(dry_caps "$l" P "$@") || return 1; echo "${c#* }"; }
k_cap() { local l=$1; shift; local c; c=$(dry_caps "$l" K "$@") || return 1; echo "${c% *}"; }

# ---------------------------------------------------------------- 0. correctness
if want selftest; then
  log "selftest"; $B selftest | tee "$OUT/logs/selftest.log" | tail -3
  done_stage selftest
fi

# ---------------------------------------------------------------- 1. stash vs load (E5, Reviewer A)
# Paper model p = 1, s = 4, mu = mu_factor * n; n = 24M matches Enron. The
# "paper" discipline is the analyzed one; "legacy" is the original code and the
# Python simulations behind the current figures; "noinv" drops invalidation.
if want stash; then
  # the runs are independent and single-threaded: run STASH_JOBS at once
  # (default: min(cores, RAM / 10 GB); a 24M run at mu = 2n needs up to ~8 GB)
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
  for ev in "rw 10" "bfs 20"; do set -- $ev   # bucketized layout of the original implementation
    for d in paper legacy; do
      f="$OUT/stash/p256_${1}a${2}_${d}.csv"; [[ -s $f ]] && continue
      echo "$B stash --n $N --p 256 --mu-factor 2.0 --evict $1 --alpha $2 --disc $d --checkpoints 50 --out $f" >> "$JOBS"
    done
  done
  log "stash: $(wc -l < "$JOBS") runs, $STASH_JOBS at a time"
  # a run writes its CSV incrementally; remove partial files of failed runs so a restart redoes them
  xargs -P "$STASH_JOBS" -I{} bash -c '{} > /dev/null || { echo "FAILED: {}"; f=$(echo "{}" | sed "s/.*--out //"); rm -f "$f"; }' < "$JOBS"
  done_stage stash
fi

# ---------------------------------------------------------------- 1b. stash under churn (no theorem covers this)
# Fill to n live keys, then 4n operations at constant load. `replace` frees a key
# and inserts a new one (the versioned-key pattern of mkse_log); `update` rewrites
# the same key with an invalidation-mode update (a fresh state of that key each
# time). Targets uniform or Zipf(1); rebuild never or every n operations.
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

# ---------------------------------------------------------------- 2. write path, accounting only (E1, E3, Reviewer B)
# Replays 20% of the corpus (inserts, posting deletes, document edits and
# deletes) on top of an 80% setup and accounts every byte a client downloads
# (refresh on search) and every row the owner writes. No PIR here, so it is fast.
if want write; then
  # the recommended schedule (defaults): B = 10^4 appends per batch, flush a
  # keyword once a full chunk (theta = C = 203) is pending, BFS alpha = 20 with
  # sigma = 16, major compaction every 10^6 appends. Every MKSE run below uses
  # alpha = 20 (the default) except main_a10, which varies alpha. A rebuild
  # encodes exactly --val-cap items (default ceil(0.46 mu) of the reuse-mode
  # table), a DocMap repack leaves rows - --doc-live-cap never-written rows,
  # and a log snapshot has kw_pad + --ver-cap records (see `design --help`).
  # Schedules: log compaction every E_L = --log-every appends, ValMap rebuild
  # every E_V = --major-every appends (a multiple of E_L). In the runs below
  # E_L = E_V (the default), so every log compaction is a ValMap rebuild and
  # snapshots carry no pending postings (P̂ = 0); main_decoupled separates them.
  design main           --policy batched --log-every 1000000 --major-every 1000000
  # decoupled: log compaction every DEC_EL (10^6) appends, no ValMap rebuild in
  # the replay (E_V = 0): snapshots carry every pending posting (P̂ from a dry run)
  DEC=(--policy batched --log-every $DEC_EL --major-every 0 --mu-factor 2.2)   # no rebuild: tombstones accumulate; 10% larger table keeps live chunks below the cap
  PC=$(pend_cap main_decoupled "${DEC[@]}")
  design main_decoupled "${DEC[@]}" --pend-cap "$PC"
  # log budget 10^5 appends (phased age rule): every posting is committed within 10^5
  # appends, the uncommitted log tail of static rebuild_100000; K from a dry run;
  # E_L = E_V = 10^6 as in main
  AG=(--policy batched --age-appends 100000 --log-every 1000000 --major-every 1000000)
  KC=$(k_cap main_age_1e5 "${AG[@]}")
  design main_age_1e5 "${AG[@]}" --k "$KC"
  design main_a10       --policy batched --alpha 10
  # baselines: append-only log, no ValMap writes until a rebuild
  design logonly        --policy rebuild --r 0
  for r in 100000 300000 1000000; do design rebuild_$r --policy rebuild --r $r; done
  # the original design: one ValMap operation per insertion, flush at half a chunk
  design perop          --policy perop --theta 101 --major-every 0 --mu-factor 2.2 --search-every 100000
  # sensitivity around the recommended schedule
  for b in 1000 3000 30000; do design main_b$b --policy batched --b $b; done
  for th in 50 101; do design main_t$th --policy batched --theta $th; done
  design main_major300000 --policy batched --major-every 300000
  design main_major0 --policy batched --major-every 0 --mu-factor 2.2
  # random walk: its stash holds ~3% of the chunks (thousands), so sigma = 16
  # would halt at setup; this run uses a stash large enough to complete
  design main_rw        --policy batched --evict rw --sigma 8192
  design main_age       --policy batched --age 64       # adds an age bound on pending postings
  # superseded chunks kept until compaction, filler insertions instead of dummies
  # (every operation is an insertion: Theorem 1 covers owners that share with
  # corrupted clients without leaking the update pattern)
  design main_noreuse   --policy batched --no-reuse --mu-factor 2.6
  # DocMap: row-addressed (default) vs a Cuckoo UOKVS keyed (id, ver, j)
  design docmap_cuckoo  --policy batched --doc-layout cuckoo
  design docmap_rows_4k --policy batched --doc-row 4096
  # half-built corpus
  design logonly_bf05   --policy rebuild --r 0 --build-frac 0.5
  design main_bf05      --policy batched --build-frac 0.5
  done_stage write
fi

# ---------------------------------------------------------------- 3b. operating point (named stage)
# The recommended configuration: phased age rule with log budget A, log compaction
# every 10^6 appends, and no ValMap rebuild within the replay (the certified rebuild
# period, ~1.8e7 appends, exceeds it). K and P-hat from one dry run. A 10% larger
# table (mu-factor 2.2) keeps live chunks (tombstones are dropped only at a rebuild)
# below the cap. Run with: ./scripts/eval_all.sh oppoint report
if named oppoint; then
  CUR=oppoint
  for A in 100000 1000000; do
    OP=(--policy batched --age-appends $A --log-every 1000000 --major-every 0 --mu-factor 2.2)
    C=$(dry_caps main_op_$A KP "${OP[@]}") || exit 1
    design main_op_$A "${OP[@]}" --k "${C% *}" --pend-cap "${C#* }" || exit 1
  done
  done_stage oppoint
fi

# ---------------------------------------------------------------- 3. write path with real FrodoPIR (E2, E3)
# Same replays with the PIR databases maintained: per-batch row + hint updates
# (the fork's incremental hint update), FrodoPIR setup at every rebuild, and
# measured PIR searches (batch width 32). DOC_PIR=1 also keeps the DocMap PIR
# database (tens of GB on the full corpus).
if want writepir; then
  DOCPIR=(--no-doc-pir); [[ ${DOC_PIR:-0} == 1 ]] && DOCPIR=()
  design pir_main       --policy batched --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  design pir_rebuild    --policy rebuild --r 100000 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  design pir_logonly    --policy rebuild --r 0 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000 "${PREFIX[@]}"
  done_stage writepir
fi

# ---------------------------------------------------------------- 4. FrodoPIR fork microbenchmarks (E2c)
# ValMap shape (8 KB rows) and DocMap shape (1059-byte rows = 1 KB + nonce, meta, check):
# setup, hint size, row updates of one batch (k(α+1) rows), and query batches at
# widths 8..64 (one pass over the database per width-sized group).
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

# ---------------------------------------------------------------- 5. fork vs upstream FrodoPIR (our FrodoPIR improvements)
if want compare; then
  ( cd pir_compare && cargo build --release )
  : > "$OUT/pir_compare.txt"
  # upstream FrodoPIR has no hint update, so every update size is a full
  # rebuild (~35 min at 524K rows if single-threaded); measure the rebuilds
  # only at the smallest size and reuse its setup time above it
  SMALLEST=$(echo $PIR_ROWS | tr ' ' '\n' | sort -n | head -1)
  for rows in $PIR_ROWS; do
    SKIP=(); [[ $rows -gt $SMALLEST && ${UPSTREAM_REBUILD_ALL:-0} != 1 ]] && SKIP=(--skip-upstream-rebuild)
    log "fork vs upstream $rows ${SKIP[*]}"
    ./pir_compare/target/release/pir_compare --rows $rows --elem-bytes 8192 --update-rows 11,110,210 \
       --batches 4,16,64,256 "${SKIP[@]}" --out "$OUT/pir/fork_vs_upstream_${rows}.csv" | tee -a "$OUT/pir_compare.txt"
  done
  done_stage compare
fi

# ---------------------------------------------------------------- 6. YPIR+SP: reference code vs our faster server (ypir_fast)
# ValMap and DocMap shapes: records per row, batched/prepared answers, row updates, shared keys
if want ypir; then
  log "ypir"; QUICK=$QUICK ./scripts/run_ypir.sh "$OUT/pir/ypir" 2>&1 | grep -E "^(impl|ok|ypir_fast)" | tee "$OUT/ypir.txt"
  done_stage ypir
fi

# ---------------------------------------------------------------- 6b. Spiral (hintless PIR): Blyss server vs our faster server
# Byte-identical responses are checked first. Shapes: ValMap 8 KB records
# (n = 2) at 2^14..2^17, DocMap 2 KB records (n = 1, two 1 KB chunks), and the
# full-size DocMap (SPIRAL_DOC=0 skips it).
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
  # DocMap at full size (2.3M rows of 1059 B): one row per 2 KB record (n = 1, 2^21 records after
  # packing two rows per record) or seven rows per 8 KB record (n = 2, 2^19 records); ~34 GB each.
  # Our server only (the Blyss server would take hours); answers are decoded and checked.
  if [[ $QUICK != 1 && ${SPIRAL_DOC:-1} == 1 ]]; then
    for s in 1,9,12 2,9,10; do
      log "spiral docmap $s"
      $SB bench $s --impl fast --batches 1,16,64 --trials 1 | grep -E "^impl" | sed 's/^impl=fast/impl=fast_docmap/' | tee -a "$OUT/spiral.txt" \
        || echo "spiral docmap $s failed (see above)" | tee -a "$OUT/spiral.txt"
    done
  fi
  done_stage spiral
fi

# ---------------------------------------------------------------- 6c. paired evaluation: client models x PIR per map
# Every run reports, from one replay, the client download for all four
# pairings (ValMap PIR, DocMap PIR) in {FrodoPIR, Spiral}^2:
#   log clients       copy the append-only log (rebuild or batched schedules)
#   stateless clients download VolMap + the pending stash; every insertion
#                     reaches ValMap through the eviction walk (batches of B,
#                     theta = 1), DocMap is a UOKVS with versioned keys
#                     (id, (epoch, u, j)) re-encoded at every rebuild, plus a
#                     padded version map (--vmap-pad) downloaded with every search
# The real-PIR runs then measure server time for the main pairings,
# including Spiral for ValMap with FrodoPIR for DocMap.
if want paired; then
  design paired_log_only     --policy rebuild --r 0
  design paired_log_rebuild  --policy rebuild --r 100000
  design paired_log_batched  --policy batched
  for b in 1 10 100 1000; do
    design paired_stateless_b$b --stateless --b $b --search-every 100000
  done
  DOCPIR=(--no-doc-pir); [[ ${DOC_PIR:-1} == 1 ]] && DOCPIR=()
  # PAIRED_PIR=0 skips the real-PIR runs (server time) and reruns only the cost model
  [[ ${PAIRED_PIR:-1} == 1 ]] && for pv in spiral frodo; do
    design pir_stateless_${pv}_frodo --stateless --b 100 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
    design pir_log_${pv}_frodo --policy rebuild --r 100000 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
  done
  if [[ ${DOC_SPIRAL:-0} == 1 ]]; then   # Spiral DocMap: ~16 KB of server memory per 2 KB row
    design pir_stateless_spiral_spiral --stateless --b 100 --pir --val-pir spiral --doc-pir spiral \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" "${PREFIX[@]}"
  fi
  done_stage paired
fi

# ---------------------------------------------------------------- 8. scale at equal log budgets (not in `all`)
# k copies of the corpus (--scale k: id -> id + c(max id + 1)), replayed with the copies'
# operation streams interleaved round-robin by document, in two modes (SCALE_MODES):
#   shared:   keywords kept: kappa fixed, every keyword's volume (chunks, mu) k-fold,
#             the sublinear-vocabulary case (Heaps' law: kappa_A grows slower than N);
#   disjoint: keyword w -> w~c: kappa, mu, DocMap and every cap k-fold with the same
#             per-keyword statistics (vocabulary linear in N: the worst case).
# For each log budget A (every posting committed within A appends):
#   scale_static_{mode}_k{k}_A{A}: append + rebuild: K = 0, ValMap rebuild and log compaction every A.
#   scale_uokvs_{mode}_k{k}_A{A}:  batched B = SCALE_B (10^4), theta = C, phased age rule A
#                           (--age-appends A), K from a dry run (no K-overflow), no ValMap
#                           rebuild in the replay (E_V = 0), log compaction every
#                           E_L = SCALE_UOKVS_EL (default A: the same log length bound and
#                           snapshot schedule as the static run, so the two differ only in
#                           batches vs rebuilds), P̂ from the same dry run.
# The ValMap rebuild term of UOKVS for any certified period E_V is mu x row / E_V
# per append (summary column val_rebuild_bytes = mu x row).
# Replays stop after SCALE_APPENDS (3 x 10^6) appends: >= 3 windows of the largest A.
# A in SCALE_AS (10^5, 3 x 10^5, 10^6) for shared and SCALE_AS_DISJOINT (10^5, 10^6:
# 3 x 10^5 dropped to keep the stage near 3 h on the server) for disjoint.
# Runs go SCALE_JOBS at a time (default RAM / 90 GB: a k = 16 run needs up to ~70 GB),
# largest first; each job writes its own summary, merged into $OUT/design_summary*.csv.
# Run: ./scripts/eval_all.sh scale report      (knobs: SCALE_MODES SCALE_KS SCALE_AS SCALE_AS_DISJOINT SCALE_B
#      SCALE_APPENDS SCALE_UOKVS_EL SCALE_JOBS K_HEADROOM K_FROM PEND_HEADROOM)
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
  # merge the per-run summaries (same column layout as this binary's design_summary.csv)
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

# ---------------------------------------------------------------- 7. one report
if want report; then
  $B report --dir "$OUT"
fi
log "done; see $OUT/REPORT.md"
