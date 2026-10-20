#!/usr/bin/env bash
# Revision experiments (see EXPERIMENTS.md). Run from the repository root:
#   DATA=/home/bfuller/MKSE/okvs_python/data/data OUT=results ./scripts/run_experiments.sh [stash|ops|pir|all]
set -euo pipefail
DATA=${DATA:-/home/bfuller/MKSE/okvs_python/data/data}
OUT=${OUT:-results}
WHAT=${1:-all}
export RUSTC_BOOTSTRAP=1                      # frodo-pir needs portable_simd; or use nightly
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
cargo build --release --bin uokvs_bench
B=./target/release/uokvs_bench
mkdir -p "$OUT"
$B selftest

if [[ $WHAT == stash || $WHAT == all ]]; then
  # E5: stash vs load. No data needed (lookup keys are pseudorandom).
  # Paper model: p=1, s=4, mu = mu_factor * n. n = 24M matches Enron.
  N=${N:-24000000}
  for mf in 2.0 1.5 1.333 1.111; do
    for ev in "rw 10" "bfs 20"; do
      set -- $ev
      for d in paper legacy noinv; do
        for jf in 0.0 0.8; do
          $B stash --n $N --mu-factor $mf --evict $1 --alpha $2 --disc $d \
             --joint-frac $jf --checkpoints 50 --branch 2 --depth 64 \
             --out "$OUT/stash/${1}_${d}_mf${mf}_jf${jf}.csv"
        done
      done
    done
  done
  # bucketized table as in the implementation (p = 256, mu = 2n/p)
  for ev in "rw 10" "bfs 20"; do set -- $ev
    for d in paper legacy; do
      $B stash --n $N --p 256 --mu-factor 2.0 --evict $1 --alpha $2 --disc $d \
         --checkpoints 50 --out "$OUT/stash/p256_${1}_${d}.csv"
    done
  done
fi

if [[ $WHAT == ops || $WHAT == all ]]; then
  # E3/E1: build 80% (or 50%), replay the rest in file-id order with deletions
  # and document updates; snapshots and sampled searches every 10k operations.
  for ev in bfs rw; do
    for bf in 0.8 0.5; do
      $B ops --data "$DATA" --build-frac $bf --order fileid --evict $ev --disc paper \
         --update-mode inval --del-frac 0.02 --upd-frac 0.02 --docdel-frac 0.002 \
         --search-every 10000 --search-sample 50 --encrypt \
         --out-prefix "$OUT/ops/${ev}_bf${bf}_paper"
    done
  done
  # pending-stash budget sweep (stash download per search vs. partial chunks)
  for budget in none 100000 20000 5000; do
    $B ops --data "$DATA" --build-frac 0.8 --evict bfs --disc paper --pend-budget $budget \
       --search-every 10000 --search-sample 20 \
       --out-prefix "$OUT/ops/bfs_budget_${budget}"
  done
  # in-place updates (leak the update pattern) for the cost comparison
  $B ops --data "$DATA" --build-frac 0.8 --evict bfs --disc paper --update-mode inplace \
     --upd-frac 0.05 --search-every 10000 --out-prefix "$OUT/ops/bfs_inplace"
  # with FrodoPIR: real row + hint updates and PIR searches (memory heavy)
  $B ops --data "$DATA" --build-frac 0.8 --evict bfs --disc paper --pir \
     --search-every 20000 --search-sample 20 --pir-search 5 \
     --out-prefix "$OUT/ops/bfs_pir"
fi

if [[ $WHAT == pir || $WHAT == all ]]; then
  # FrodoPIR fork microbenchmarks at ValMap sizes (8 KB rows) and DocMap sizes (1 KB rows)
  for rows in 65536 262144 524288; do
    $B pir --rows $rows --elem-bytes 8192 --batches 4,8,16,64,256,784 --update-rows 10,21 \
       --trials 3 --out "$OUT/pir/frodo_fork_${rows}x8k.csv"
  done
  $B pir --rows 4194304 --elem-bytes 1024 --batches 8,64,512 --update-rows 10,21 \
     --trials 3 --out "$OUT/pir/frodo_fork_4Mx1k.csv"
  # fork vs upstream FrodoPIR (the paper's FrodoPIR improvements)
  (cd pir_compare && cargo build --release && \
   for rows in 16384 65536 262144; do
     ./target/release/pir_compare --rows $rows --elem-bytes 8192 --update-rows 10,21 \
        --batches 4,16,64,256 --out "../$OUT/pir/fork_vs_upstream_${rows}.csv"
   done)
  # YPIR (+SP) on the same shapes, through its own Rust CLI
  ./scripts/run_ypir.sh "$OUT/pir/ypir"
fi
echo "results in $OUT/"
