# UOKVS / MKSE

Code for the paper on updatable oblivious key-value stores (UOKVS) and the multi-key searchable encryption (MKSE) scheme built on them. The UOKVS is Cuckoo hashing with random walk or BFS eviction; the MKSE stores VolMap, ValMap and DocMap and reads them over PIR. The benchmarks for the evaluation are here too.

This is research code, written to get the numbers for the paper and never audited. Don't use it in production.

## Layout

```
src/main.rs, dip.rs, bench.rs,   original static benchmark (binary `mkse`)
  generate_mmap.rs
src/uokvs.rs                     UOKVS: RW/BFS eviction with invalidation, stash, encryption
src/mkse_log.rs                  dynamic MKSE: append-only log, batched ValMap flushes,
                                 compaction/rebuild schedules, DocMap layouts, client cost model
src/mkse_dyn.rs                  earlier dynamic design (one ValMap op per insertion), used by `ops`
src/pir_backend.rs               FrodoPIR and Spiral behind one interface
src/bin/uokvs_bench.rs           benchmark binary for the dynamic experiments
frodo-pir/                       our FrodoPIR fork (batched answers, SIMD, hint updates)
spiral_fast/                     faster Spiral server, compatible with the Blyss client
ypir_fast/                       faster YPIR+SP server (records per row, updates, batching)
pir_compare/                     FrodoPIR fork vs. upstream FrodoPIR
scripts/eval_all.sh              runs the dynamic experiments
scripts/run_ypir.sh              YPIR+SP benchmarks (called from eval_all.sh)
data/                            dataset scripts (Enron parsing, synthetic data)
Makefile                         static benchmarks (setup/query table, sharding, subsets)
```

## Requirements

- Rust (tested with 1.97). FrodoPIR uses `portable_simd`, so either use nightly or set `RUSTC_BOOTSTRAP=1` (the scripts do this).
- x86-64 with AVX-512 for `spiral_fast` and `ypir_fast` (the upstream YPIR code needs it). Everything else, including `selftest`, also builds on ARM.
- Python 3 with `nltk numpy tqdm brotli` for the data scripts.

```bash
python3 -m venv env && source env/bin/activate
pip install nltk numpy tqdm brotli
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('words')"
```

The full runs in the paper were done on a 96-core AMD Threadripper PRO 7995WX with 768 GB of RAM (Ubuntu 22.04). Some stages need a lot of memory (see Resources below).

## Data

Download the Enron corpus (May 2015 version, https://www.cs.cmu.edu/~enron/) and extract it so that `data/maildir` exists. Then:

```bash
cd data
python3 parse_enron.py              # writes data/data/{keyword_file_map.json, keywords.txt, files.txt, <doc ids>}
python3 compress.py data data_br    # Brotli-compressed documents, only needed for DocMap retrieval in `mkse`
cd ..
```

`generate_power.py` and `generate_gaussian.py` generate synthetic multimaps instead. The dynamic experiments can also run on a built-in synthetic corpus (`QUICK=1` or `--synthetic`), which needs no data at all.

## Building

```bash
export RUSTC_BOOTSTRAP=1 RUSTFLAGS="-C target-cpu=native"
cargo build --release
./target/release/uokvs_bench selftest
```

`selftest` checks every eviction variant against a reference map, the encryption and PIR round trip (including after row updates), and the dynamic MKSE designs against a reference multimap under inserts, deletes, edits and removals. It takes a few seconds. `cargo test --release --lib` runs the unit tests.

## Static benchmarks

These measure setup time, query time and bandwidth for the static maps (encoded once, never updated) on the full and partial Enron corpus. They use the original `mkse` binary and the Makefile, and the dynamic experiments below don't depend on them. `DATA_DIR` is the directory produced by `parse_enron.py`.

```bash
DATA_DIR=data/data USE_EXISTING=false make all                      # all of the below
DATA_DIR=data/data USE_EXISTING=true make subsets                   # queries for keywords with 100/500/1000 docs
DATA_DIR=data/data USE_EXISTING=true make sharding
DATA_DIR=data/data USE_EXISTING=true make sharding-with-keywords
DATA_DIR=data/data USE_EXISTING=true make test-docmap               # DocMap retrieval
DATA_DIR=data/data USE_EXISTING=true make test-inserts              # single insertions into an 80% build
```

Results go to `output/`. The binary can also be run directly: `./target/release/mkse --data-dir data/data [--use-bfs] [--read-keywords FILE]`.

## Dynamic experiments

Run `scripts/eval_all.sh` from the repo root:

```bash
DATA=data/data ./scripts/eval_all.sh            # every stage in `all`, on Enron
./scripts/eval_all.sh write paired report       # only some stages
QUICK=1 ./scripts/eval_all.sh                   # synthetic corpus and small sizes
```

Results go to `results/` (set `OUT` to change it) and logs to `results/logs/`. The `report` stage collects every CSV into `results/REPORT.md`. If the script stops you can just start it again: with `all` it skips stages and runs that already finished. A stage named on the command line always reruns, and `FORCE=1` reruns everything except stash runs that already have output.

| Stage | What it runs |
|---|---|
| `selftest` | correctness checks (see above) |
| `stash` | stash size vs. load for RW (α=10) and BFS (α=10, 20), with the analyzed invalidation rule (`paper`), the original code (`legacy`) and no invalidation, built from scratch and from an 80% joint build; n = 24M, s = 4, p = 1, plus the p = 256 layout |
| `churn` | stash under churn at constant load: fill to 10^6 keys, then 4n replace or update operations, uniform or Zipf(1) targets, with and without periodic rebuilds |
| `write` | the write path on Enron without PIR: 80% setup, then a replay of the remaining 20% with 2% posting deletes, 2% edits and 0.2% document removals. Compares the append-only log, rebuild every 10^5/3·10^5/10^6 appends, one ValMap operation per insertion, and batched flushes over B, θ, α, the age rule, RW vs BFS, row vs Cuckoo DocMap, and a 50% build |
| `writepir` | the main `write` variants with real FrodoPIR databases kept up to date (row and hint updates per batch), with measured search times |
| `pir` | FrodoPIR fork microbenchmarks at the ValMap (8 KB rows) and DocMap (1059 B rows) shapes: setup, hint size, row updates, query batches of width 8 to 64 |
| `compare` | FrodoPIR fork vs. upstream FrodoPIR (setup, hint refresh after k changed rows, batched answers, client preprocessing) |
| `ypir` | YPIR+SP: the reference CLI and `ypir_fast` at the ValMap and DocMap shapes |
| `spiral` | Spiral: the Blyss server vs. `spiral_fast` (responses are checked to be byte-identical first), then `spiral_fast` alone at the full DocMap size |
| `paired` | client download per search for log clients and stateless clients, for each pairing of PIR schemes on ValMap and DocMap, plus real-PIR runs for server time |
| `report` | writes `REPORT.md` |
| `oppoint` | not in `all`. The configuration we recommend: age rule with log budget A = 10^5 and 10^6, log compaction every 10^6 appends, no ValMap rebuild during the replay |
| `scale` | not in `all`. 1, 4 and 16 copies of Enron (shared or disjoint keywords) at equal log budgets: append + rebuild vs. UOKVS batches |

To run the two extra stages: `./scripts/eval_all.sh oppoint scale report`.

### Options

Environment variables read by `eval_all.sh`:

- `DATA`: Enron directory (default `data/data`). Not needed with `QUICK=1`.
- `OUT`: output directory (default `results`).
- `QUICK=1`: synthetic corpus and smaller sizes.
- `STASH_JOBS`: stash/churn runs in parallel (default min(cores, RAM/10 GB)).
- `PIR_APPENDS`: real-PIR replays stop after this many appends (default 200000, 0 = full replay).
- `DOC_PIR=1`: also keep the DocMap in PIR in `writepir`.
- `PAIRED_PIR=0`: skip the real-PIR runs in `paired` and only compute the cost model. `DOC_SPIRAL=1` adds a run with Spiral for both maps.
- `UPSTREAM_REBUILD_ALL=1`: in `compare`, time upstream full rebuilds at every size, not only the smallest.
- `SPIRAL_DOC=0`: skip the full-size DocMap in `spiral`.
- `YPIR_SEEDS`: public seeds for shared packing keys in `ypir` (default 8).
- `scale` stage: `SCALE_MODES` ("shared disjoint"), `SCALE_KS` ("16 4 1"), `SCALE_AS` ("100000 300000 1000000"), `SCALE_AS_DISJOINT` ("100000 1000000"), `SCALE_B` (10000), `SCALE_APPENDS` (3000000, 0 = full replay), `SCALE_UOKVS_EL`, `SCALE_JOBS` (default RAM/90 GB).
- `K_HEADROOM`, `PEND_HEADROOM`: headroom over the dry-run maximum for K and P̂ (default 1.1).

### Resources

On the full corpus:

- `write` runs keep only owner state and take a few minutes each.
- `stash` is 76 runs at n = 24M, a few GB each.
- The ValMap FrodoPIR database at 2^18 rows is about 7.6 GB plus the hint computation. The DocMap PIR database is about 15 GB, which is why `writepir` skips it unless `DOC_PIR=1`.
- The full-size Spiral DocMap runs need about 34 GB each. `ypir` with 8 seeds needs about 25 GB.
- A `scale` run with k = 16 needs up to about 70 GB.

### Running things by hand

Each stage is just a list of `uokvs_bench` calls, so you can run them yourself. Running `./target/release/uokvs_bench` with no arguments prints all the options. For example:

```bash
B=./target/release/uokvs_bench
$B stash --n 1000000 --mu-factor 2.0 --evict bfs --alpha 20 --disc paper --out results/stash/bfs.csv
$B churn --n 100000 --mu-factor 1.5 --evict rw --alpha 10 --disc paper --mode replace --out results/churn/rw.csv
$B design --data data/data --policy batched --label main --out-prefix results/design/main --summary results/design_summary.csv
$B design --data data/data --policy rebuild --r 100000 --label rebuild_100000 --out-prefix results/design/rebuild_100000 --summary results/design_summary.csv
$B pir --rows 131072 --elem-bytes 8192 --batches 16,64 --widths 8,32 --update-rows 11,110 --out results/pir/val.csv
$B report --dir results
```

The PIR benchmarks build separately:

```bash
(cd pir_compare && cargo build --release) && ./pir_compare/target/release/pir_compare --rows 16384 --elem-bytes 8192
(cd spiral_fast && cargo build --release) && ./spiral_fast/target/release/spiral_bench check
QUICK=1 ./scripts/run_ypir.sh results/pir/ypir
```

## License

GPLv3. `frodo-pir/` keeps the license of the original FrodoPIR.
