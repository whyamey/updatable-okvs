#!/usr/bin/env bash
# YPIR+SP (Menon-Wu, USENIX Security 2024) at the UOKVS table shapes.
#
#  1. The reference code (github.com/menonsamir/ypir, a73e550) through its CLI: one record per
#     row, single-threaded, one query per pass, no row updates (an update is the offline rebuild).
#  2. ypir_fast (ours, ./ypir_fast): several records per row, row updates (hint delta for the
#     touched row blocks + parallel packing precomputation), batched and parallel online pass,
#     packing prepared before the query arrives, and packing keys shared over public seeds.
#     `--ref` also times the reference code paths (offline and one query) at the same shape.
#
#   ./scripts/run_ypir.sh results/pir/ypir
# Knobs: QUICK=1 (small shapes), YPIR_SEEDS (default 8), YPIR_REF_CLI=0 (skip part 1),
#        VAL_RECORDS / DOC_RECORDS / DOC_BYTES (table shapes of the main replay).
set -euo pipefail
OUTDIR=${1:-results/pir/ypir}
mkdir -p "$OUTDIR"
QUICK=${QUICK:-0}
SEEDS=${YPIR_SEEDS:-8}; [[ $QUICK == 1 ]] && SEEDS=${YPIR_SEEDS:-2}
export RUSTC_BOOTSTRAP=1
if [[ $QUICK == 1 ]]; then
  VAL_RECORDS=${VAL_RECORDS:-20000}; DOC_RECORDS=${DOC_RECORDS:-100000}
  CLI_SHAPES=("65536 65536"); TR=1
else
  VAL_RECORDS=${VAL_RECORDS:-316214}; DOC_RECORDS=${DOC_RECORDS:-2318593}
  CLI_SHAPES=("65536 65536" "262144 65536" "524288 65536" "4194304 8192"); TR=2
fi
DOC_BYTES=${DOC_BYTES:-1059}

# ---------------------------------------------------------------- 1. reference CLI
if [[ ${YPIR_REF_CLI:-1} == 1 ]]; then
  YPIR_DIR=${YPIR_DIR:-../ypir}
  if [[ ! -d $YPIR_DIR ]]; then git clone https://github.com/menonsamir/ypir.git "$YPIR_DIR"; fi
  ( cd "$YPIR_DIR" && RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable} \
      RUSTFLAGS="-C target-cpu=native" cargo build --release )
  RUN="$YPIR_DIR/target/release/run"
  # rows x item bits; YPIR+SP needs >= 28672-bit items: pack four 1 KB rows per item
  for shape in "${CLI_SHAPES[@]}"; do
    set -- $shape
    if [[ $2 -ge 28672 ]]; then
      $RUN $1 $2 1 3 "$OUTDIR/ypirsp_${1}x${2}.json" --is-simplepir > /dev/null
    else
      $RUN $(( $1 / 4 )) 32768 1 3 "$OUTDIR/ypirsp_${1}x${2}.json" --is-simplepir > /dev/null
    fi
    python3 - "$OUTDIR/ypirsp_${1}x${2}.json" "$1" "$2" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); o, n = d["offline"], d["online"]
print(f"impl=ypir_cli,records={sys.argv[2]},item_bits={sys.argv[3]},offline_ms={o['serverTimeMs']},"
      f"upload_bytes={n['uploadBytes']},download_bytes={n['downloadBytes']},server_ms={n['serverTimeMs']},"
      f"first_ms={n['firstPassTimeMs']},pack_ms={n['ringPackingTimeMs']},client_query_ms={n['clientQueryGenTimeMs']}")
EOF
  done
fi

# ---------------------------------------------------------------- 2. ypir_fast
( cd ypir_fast && RUSTFLAGS="-C target-cpu=native" cargo build --release )
YB=./ypir_fast/target/release/ypir_bench
$YB check | tail -1
run() { $YB bench "$@" --trials $TR | grep -E "^impl"; }
# ValMap (8 KB records): balanced rows (auto), with the reference code paths at the same shape
run --records $VAL_RECORDS --record-bytes 8192 --per-row auto --batches 1,16,32 --updates 1,100,1000,10000 --ref
# one record per row (the reference layout): smallest response, largest query
run --records $VAL_RECORDS --record-bytes 8192 --per-row 1 --batches 1,16 --updates 1,1000
# shared packing keys: a search's queries use one client key over $SEEDS public seeds
run --records $VAL_RECORDS --record-bytes 8192 --per-row auto --seeds $SEEDS --batches 16,32 --updates 1000
# DocMap (1 KB records)
run --records $DOC_RECORDS --record-bytes $DOC_BYTES --per-row auto --batches 1,64,256 --updates 1,1000,10000
echo "YPIR reports in $OUTDIR"
