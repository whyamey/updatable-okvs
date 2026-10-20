#!/usr/bin/env bash
# The whole evaluation plan in one script (see EXPERIMENTS.md for what each
# stage measures and which paper item it answers). Run from the repository root:
#
#   DATA=/home/bfuller/MKSE/okvs_python/data/data ./scripts/eval_all.sh            # everything
#   ./scripts/eval_all.sh write pir report                                          # some stages
#   QUICK=1 ./scripts/eval_all.sh                                                   # small sizes, minutes
#
# Stages: selftest stash write writepir pir compare ypir spiral paired report   (default: all)
# Outputs: $OUT/ (CSV per run, design_summary*.csv, REPORT.md). Logs: $OUT/logs/.
set -euo pipefail
DATA=${DATA:-/home/bfuller/MKSE/okvs_python/data/data}
OUT=${OUT:-results}
QUICK=${QUICK:-0}
STAGES=${*:-all}
export RUSTC_BOOTSTRAP=1                       # frodo-pir uses portable_simd (or use nightly)
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
want() { [[ " $STAGES " == *" all "* || " $STAGES " == *" $1 "* ]]; }
mkdir -p "$OUT/logs" "$OUT/design" "$OUT/stash" "$OUT/pir"
cargo build --release --bin uokvs_bench
B=./target/release/uokvs_bench
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [[ $QUICK == 1 ]]; then
  SRC=(--synthetic 3000,20000,40); N=1000000; MFS="2.0 1.333"; PIR_ROWS="16384"; DOC_ROWS="65536"; TRIALS=1
  QPIR=(--synthetic 1000,5000,20)   # smaller replays for the real-PIR runs
else
  if [[ ! -f $DATA/keyword_file_map.json ]]; then echo "no $DATA/keyword_file_map.json (set DATA or QUICK=1)"; exit 1; fi
  SRC=(--data "$DATA"); N=24000000; MFS="2.0 1.5 1.333 1.111"; PIR_ROWS="131072 262144 524288"; DOC_ROWS="2097152 4194304"; TRIALS=2
  QPIR=()
fi
CLIENTS=100,1000,10000,100000,1000000
COMMON=(--build-frac 0.8 --order fileid --del-frac 0.02 --upd-frac 0.02 --docdel-frac 0.002
        --clients $CLIENTS --search-every 20000 --search-sample 20 --snap-every 10000
        --summary "$OUT/design_summary.csv")
design() { # label, extra args...
  local l=$1; shift
  log "design $l"
  $B design "${SRC[@]}" "${COMMON[@]}" --label "$l" --out-prefix "$OUT/design/$l" "$@" > "$OUT/logs/design_$l.log" 2>&1
  tail -7 "$OUT/logs/design_$l.log"
}

# ---------------------------------------------------------------- 0. correctness
if want selftest; then
  log "selftest"; $B selftest | tee "$OUT/logs/selftest.log" | tail -3
fi

# ---------------------------------------------------------------- 1. stash vs load (E5, Reviewer A)
# Paper model p = 1, s = 4, mu = mu_factor * n; n = 24M matches Enron. The
# "paper" discipline is the analyzed one; "legacy" is the original code and the
# Python simulations behind the current figures; "noinv" drops invalidation.
if want stash; then
  for mf in $MFS; do
    for ev in "rw 10" "bfs 20" "bfs 10"; do set -- $ev
      for d in paper legacy noinv; do
        for jf in 0.0 0.8; do
          f="$OUT/stash/${1}a${2}_${d}_mf${mf}_jf${jf}.csv"
          [[ -s $f ]] && continue
          log "stash $1 α=$2 $d mf=$mf joint=$jf"
          $B stash --n $N --mu-factor $mf --evict $1 --alpha $2 --disc $d --joint-frac $jf \
             --checkpoints 50 --branch 2 --depth 64 --out "$f" > /dev/null
        done
      done
    done
  done
  for ev in "rw 10" "bfs 20"; do set -- $ev   # bucketized layout of the original implementation
    for d in paper legacy; do
      f="$OUT/stash/p256_${1}a${2}_${d}.csv"; [[ -s $f ]] && continue
      log "stash p=256 $1 $d"
      $B stash --n $N --p 256 --mu-factor 2.0 --evict $1 --alpha $2 --disc $d --checkpoints 50 --out "$f" > /dev/null
    done
  done
fi

# ---------------------------------------------------------------- 2. write path, accounting only (E1, E3, Reviewer B)
# Replays 20% of the corpus (inserts, posting deletes, document edits and
# deletes) on top of an 80% setup and accounts every byte a client downloads
# (refresh on search) and every row the owner writes. No PIR here, so it is fast.
if want write; then
  # the recommended schedule (defaults): B = 10^4 appends per batch, flush a
  # keyword once a full chunk (theta = C = 203) is pending, BFS alpha = 10,
  # major compaction every 10^6 appends
  design main           --policy batched
  # baselines: append-only log, no ValMap writes until a rebuild
  design logonly        --policy rebuild --r 0
  for r in 100000 300000 1000000; do design rebuild_$r --policy rebuild --r $r; done
  # the original design: one ValMap operation per insertion, flush at half a chunk
  design perop          --policy perop --theta 101 --major-every 0 --search-every 100000
  # sensitivity around the recommended schedule
  for b in 1000 3000 30000; do design main_b$b --policy batched --b $b; done
  for th in 50 101; do design main_t$th --policy batched --theta $th; done
  design main_a20       --policy batched --alpha 20
  for e in 300000 0; do design main_major$e --policy batched --major-every $e; done
  design main_rw        --policy batched --evict rw
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
fi

# ---------------------------------------------------------------- 3. write path with real FrodoPIR (E2, E3)
# Same replays with the PIR databases maintained: per-batch row + hint updates
# (the fork's incremental hint update), FrodoPIR setup at every rebuild, and
# measured PIR searches (batch width 32). DOC_PIR=1 also keeps the DocMap PIR
# database (tens of GB on the full corpus).
if want writepir; then
  DOCPIR=(--no-doc-pir); [[ ${DOC_PIR:-0} == 1 ]] && DOCPIR=()
  design pir_main       --policy batched --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000
  design pir_rebuild    --policy rebuild --r 100000 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000
  design pir_logonly    --policy rebuild --r 0 --pir "${DOCPIR[@]}" --pir-width 32 --pir-search 3 --search-every 50000
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
fi

# ---------------------------------------------------------------- 5. fork vs upstream FrodoPIR (our FrodoPIR improvements)
if want compare; then
  ( cd pir_compare && cargo build --release )
  for rows in $PIR_ROWS; do
    log "fork vs upstream $rows"
    ./pir_compare/target/release/pir_compare --rows $rows --elem-bytes 8192 --update-rows 11,110,210 \
       --batches 4,16,64,256 --out "$OUT/pir/fork_vs_upstream_${rows}.csv" | tee -a "$OUT/pir_compare.txt"
  done
fi

# ---------------------------------------------------------------- 6. YPIR+SP on the same shapes (PIR requirements table)
if want ypir; then
  log "ypir"; ./scripts/run_ypir.sh "$OUT/pir/ypir" 2>&1 | tail -20 | tee "$OUT/ypir.txt"
fi

# ---------------------------------------------------------------- 6b. Spiral (hintless PIR): Blyss server vs our faster server
# Byte-identical responses are checked first. Shapes: ValMap 8 KB records
# (n = 2) at 2^14..2^17, DocMap 2 KB records (n = 1, two 1 KB chunks).
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
fi

# ---------------------------------------------------------------- 6c. paired evaluation: client models x PIR per map
# Every run reports, from one replay, the client download for all four
# pairings (ValMap PIR, DocMap PIR) in {FrodoPIR, Spiral}^2:
#   log clients       copy the append-only log (rebuild or batched schedules)
#   stateless clients download VolMap + the pending stash; every insertion
#                     reaches ValMap through the eviction walk (batches of B,
#                     theta = 1), DocMap is a UOKVS with invalidation updates
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
  for pv in spiral frodo; do
    design pir_stateless_${pv}_frodo --stateless --b 100 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}"
    design pir_log_${pv}_frodo --policy rebuild --r 100000 --pir --val-pir $pv --doc-pir frodo "${DOCPIR[@]}" \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" 
  done
  if [[ ${DOC_SPIRAL:-0} == 1 ]]; then   # Spiral DocMap: ~16 KB of server memory per 2 KB row
    design pir_stateless_spiral_spiral --stateless --b 100 --pir --val-pir spiral --doc-pir spiral \
       --pir-width 32 --pir-search 3 --search-every 50000 "${QPIR[@]}" 
  fi
fi

# ---------------------------------------------------------------- 7. one report
if want report; then
  $B report --dir "$OUT"
fi
log "done; see $OUT/REPORT.md"
