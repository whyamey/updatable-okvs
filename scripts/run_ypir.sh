#!/usr/bin/env bash
# YPIR / YPIR+SP (Menon-Wu, USENIX Security 2024) on database shapes matching
# ValMap (8 KB rows) and DocMap (1 KB rows). YPIR has no client hint; its API
# does not expose row updates, so the offline preprocessing time is the cost of
# a rebuild (an incremental update of the SimplePIR hint is linear, but its
# packing precomputation is not implemented incrementally).
#   ./scripts/run_ypir.sh results/pir/ypir
set -euo pipefail
OUTDIR=${1:-results/pir/ypir}
mkdir -p "$OUTDIR"
YPIR_DIR=${YPIR_DIR:-../ypir}
if [[ ! -d $YPIR_DIR ]]; then git clone https://github.com/menonsamir/ypir.git "$YPIR_DIR"; fi
( cd "$YPIR_DIR" && RUSTC_BOOTSTRAP=1 RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable} \
    RUSTFLAGS="-C target-cpu=native" cargo build --release )
RUN="$YPIR_DIR/target/release/run"
# rows x item bits. YPIR+SP (needed for KB-sized items) supports a single
# client/query per pass, so a batch of k queries costs k single queries; we
# record k = 1 and scale.
for shape in "65536 65536" "262144 65536" "524288 65536" "4194304 8192"; do
  set -- $shape
  if [[ $2 -ge 28672 ]]; then
    $RUN $1 $2 1 3 "$OUTDIR/ypirsp_${1}x${2}.json" --is-simplepir
  else
    # YPIR+SP needs >= 28672-bit items: pack four 1 KB rows (8192 bits) per item
    $RUN $(( $1 / 4 )) 32768 1 3 "$OUTDIR/ypirsp_${1}x${2}.json" --is-simplepir
  fi
done
echo "YPIR reports in $OUTDIR"
