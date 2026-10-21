# Revision experiments

All new code sits next to the original benchmark (`src/main.rs`, `dip.rs`, `bench.rs`, `generate_mmap.rs` are unchanged).

| File | What it is |
|---|---|
| `src/uokvs.rs` | UOKVS core: Cuckoo hashing with the paper's invalidation discipline (Fig. 5–7, Alg. 1), plus the old behaviour for ablations; `remove` frees a superseded versioned key in owner state only |
| `src/mkse_log.rs` | **The full dynamic MKSE design (revised §4)**: append-only log with PRF tags and epochs, ValMap chunks with versioned keys `(w, i, len)`, batches of exactly K padded UOKVS operations, major compaction, row-addressed DocMap (or a Cuckoo DocMap for comparison), and the client refresh-on-search cost model |
| `src/mkse_dyn.rs` | The previous dynamic design (pending postings in the ValMap stash, one operation per insertion); kept for reference |
| `src/pir_backend.rs` | FrodoPIR (our fork) behind one interface: setup, row updates with hint refresh, batched queries at widths 8/16/32/64 (`query_batch_w`) |
| `src/bin/uokvs_bench.rs` | `selftest`, `stash`, `design`, `pir`, `report` (and the old `ops`) |
| `pir_compare/` | Our FrodoPIR fork vs. upstream FrodoPIR (the commit the fork branched from) |
| `spiral_fast/` | Faster Spiral server (dense database, batched AVX-512 first dimension, parallel, row updates) and `spiral_bench` (check against the Blyss server, benchmarks) |
| `scripts/eval_all.sh` | **The whole evaluation plan in one script** |
| `ypir_fast/` | Faster YPIR+SP server (several records per row, row updates, batched and parallel online pass, packing prepared before the query, packing keys shared over public seeds) and `ypir_bench` (check against the reference code paths, benchmarks) |
| `scripts/run_ypir.sh` | YPIR+SP: the reference CLI and `ypir_fast` at the ValMap and DocMap shapes (called by `eval_all.sh`) |
| `scripts/run_experiments.sh` | The earlier script for the old design (superseded) |

## Run

```
DATA=/home/bfuller/MKSE/okvs_python/data/data ./scripts/eval_all.sh          # everything
./scripts/eval_all.sh write report                                           # selected stages
QUICK=1 ./scripts/eval_all.sh                                                # synthetic data, small sizes
```

Knobs (environment variables):
- `STASH_JOBS`: stash runs at once. The default is min(cores, RAM / 10 GB).
- `PIR_APPENDS`: the real-PIR replays (stages `writepir` and `paired`) stop after this many appends; the default is 200000, and 0 replays everything.
- `UPSTREAM_REBUILD_ALL=1`: also time upstream FrodoPIR's full rebuilds at the larger sizes in `compare`. By default they are timed only at the smallest size.
- `DOC_PIR`, `DOC_SPIRAL`: see the stages below.
- `design --max-appends N` replays only a prefix of the workload.

frodo-pir uses `portable_simd`; the script sets `RUSTC_BOOTSTRAP=1` (or use nightly) and `-C target-cpu=native`. Results go to `$OUT` (default `results/`): one CSV set per run, `design_summary.csv`, `design_summary_clients.csv`, and `REPORT.md` (all tables, written by `uokvs_bench report`). Logs are in `$OUT/logs/`. Stash runs that already have output are skipped, so the script can be restarted.

| Stage | What it measures | Paper / review item |
|---|---|---|
| `selftest` | Every eviction variant against a reference map; fixed access counts; encryption + PIR round trip; the old and the new MKSE designs against a reference multimap under inserts, deletes, edits, removals; batches make exactly K(α+1) accesses and K flush records; compacted logs have the public size | correctness |
| `stash` | Stash vs load, RW / BFS (α = 10, 20) × {paper discipline, legacy code, no invalidation} × {from scratch, 80% joint build}, n = 24M, plus the p = 256 layout | E5, Reviewer A (the current figures come from the Python simulations, which do not use the analyzed rule) |
| `churn` | Stash under churn at constant load (no theorem covers it): fill to n = 10^6 live keys (`CHURN_N`), then 4n operations; `replace` (free a key, insert a new one: the versioned keys of `mkse_log`) or `update` (invalidation-mode update of the same key); uniform or Zipf(1) targets; rebuild never or every n operations; RW α=10 (discipline and original code) and BFS α=10/20, loads 0.5 and 0.67 | stash beyond the theorems; why BFS and versioned keys |
| `write` | Enron: 80% setup, 20% replay with 2% posting deletes, 2% edits, 0.2% removals. Variants: log only (K = 0, no rebuild), rebuild every 10^5 / 3·10^5 / 10^6 appends, one operation per insertion (the original design), batched flushes over B × θ × α, RW, an age bound, DocMap rows vs Cuckoo, a 50% build | E1, E3, Reviewer B (insertions, deletions, updates) |
| `writepir` | The main variants with the PIR databases maintained: per-batch row + hint updates with the fork, FrodoPIR setup at each rebuild, measured PIR searches at width 32 (`DOC_PIR=1` also keeps DocMap in PIR) | E2, E3 |
| `pir` | Fork microbenchmarks at ValMap shape (8 KB rows) and DocMap shape (1059-byte rows): setup, hint size, update of 11–420 rows, query batches at widths 8–64 | E2c, DocMap speed |
| `compare` | Fork vs upstream FrodoPIR: setup, hint refresh after k changed rows (upstream must rebuild), batched answers, client preparation | our FrodoPIR improvements |
| `ypir` | YPIR+SP: the reference CLI, and `ypir_fast` vs the reference code paths at the ValMap (8 KB) and DocMap (1 KB) shapes: records per row, batches 1–256, prepared packing, updates of 1–10^4 records, shared packing keys (`YPIR_SEEDS`, default 8) | PIR requirements table; hintless PIR with low server time for the keyword lookup |
| `spiral` | Spiral: Blyss server vs our `spiral_fast` server (byte-identical responses checked first), ValMap and DocMap shapes, batches 1–128; then our server alone at the full DocMap size (2.3M rows: 2^21 × 2 KB and 2^19 × 8 KB records, ≈ 34 GB each, answers decoded and checked; `SPIRAL_DOC=0` skips) | hintless PIR for the paired evaluation; our Spiral improvements |
| `paired` | Log and stateless clients × {FrodoPIR, Spiral} per map: download per search and per append at every search interval, plus real-PIR runs for server time (Spiral ValMap + FrodoPIR DocMap, FrodoPIR for both; `DOC_SPIRAL=1` adds Spiral for both) | direct updates, PIR choice |
| `report` | `REPORT.md` with every table | |

## The design in `mkse_log.rs`

- **ValMap.** A bucket (p = 1, 8 KB) holds a chunk of up to C = 203 postings of 40 bytes. A keyword's committed postings are canonical: full chunks, then one partial chunk. Chunk keys carry their length, `(w, i, len)`, so committing more postings inserts a fresh key and frees the old copy in owner state (its bucket becomes stale = free). A fixed key could only be rewritten s − 1 times under invalidation.
- **Log.** Same-size records (16-byte tag + nonce + 40-byte body + 16-byte check = 84 bytes). Tags are PRF outputs of the keyword's (or document's) delegation key on `(epoch, type, counter)`: posting records `post n`, flush records `flush m` (the committed volume), document version records `ver m part` (version, chunk count, row numbers). Clients copy the log incrementally and find everything by tag.
- **DocMap.** `--doc-layout rows` (default): chunks go to never-written rows chosen uniformly; row numbers live in version records; one PIR query per chunk; freed rows are reused only after a repack. `--doc-layout cuckoo`: a UOKVS keyed `(id, ver, j)`, s queries per chunk.
- **Schedule.** Every `--b` appends, exactly `--k` ValMap operations (dummies pad) and `--k` flush records (dummies pad). A keyword is due at `--theta` pending postings (and, with `--age A`, when its oldest pending posting is A batches old); first-due-first; leftover due work is a K-overflow. `--k auto` = ⌈1.5·b/θ⌉ + 2. Major compaction every `--major-every` appends (and when DocMap runs out of rows): ValMap rebuilt from all postings with tombstones applied, log restarted in a new epoch. `--policy rebuild --r R` is K = 0 with a major compaction every R appends (`--r 0`: never). `--policy perop` is B = K = 1.
- **Reuse of freed chunks.** By default a superseded chunk is freed at once and its bucket reused; hiding which earlier write it came from relies on the invalidation discipline (Theorem `thm:oblivious ops`, owners with no corrupted clients). `--no-reuse` keeps superseded chunks until the next major compaction and pads batches with filler insertions, so every operation is an insertion (Theorem 1 for any corruption set), at the cost of load (0.66 instead of 0.46 in the synthetic run; size ValMap with `--mu-factor`).
- **Client cost model.** Profiles that search every 10^2 … 10^6 appends (`--clients`) and refresh only then: log records since the last sync (or the compacted log), the ValMap stash if a batch ran, and for each PIR database the packed deltas of the changed rows or a fresh hint, whichever is smaller. The report also shows the same numbers with hint traffic removed (a hintless PIR).

## Preliminary numbers (2-core cloud VM; rerun on the server)

### Write path (synthetic: 3K keywords, 20K documents, 40 keywords/document, 80% setup, ≈140K appends)

Download per append for a client that searches every 100 appends (every 10^5 appends in brackets):

| variant | per append |
|---|---|
| log only (K = 0, no rebuild) | 0.32 KB (0.10 KB) |
| rebuild every 10^5 appends | 0.60 KB (0.32 KB) |
| batched, B = 1000, θ = 101, BFS α = 10 | 2.1 KB (0.37 KB) |
| batched, BFS α = 20 | 3.6 KB (0.37 KB) |
| one ValMap operation per insertion (the original design) | 175 KB (0.43 KB) |
| batched with a Cuckoo DocMap | 8.2 KB (0.36 KB) |

- With FrodoPIR's hint, a changed 8 KB row costs every searching client about 9.1 KB, while a log record costs 84 bytes. Flushing postings into ValMap therefore never saves client bandwidth: it rewrites α+1 rows to commit about θ postings that clients already hold in the log. What it buys is a bounded log (client storage, join cost, compaction size). Rare searchers pay about one full hint per search in every variant.
- The cheapest batched setting in the QUICK sweep is B = 10^4, θ = C = 203, α = 10: 1.0 KB per append for a client searching every 100 appends (3× the log-only cost).
- The original one-operation-per-insertion design is about 80× worse than batching and 550× worse than the log for frequent searchers.
- RW under the discipline reached a ValMap stash of 137 chunks at load 0.45; BFS stayed at 0–1.
- Row-addressed DocMap: 4× fewer DocMap queries, and about 4.6 KB less download per append than the Cuckoo DocMap.

### Stash vs load

n = 10^6, p = 1, s = 4; RW α = 10; BFS α = 20, branch 2, depth 64; built from scratch:

| final load | RW paper | RW legacy | BFS paper | BFS legacy |
|---|---|---|---|---|
| 0.50 | 2.5e-2 | 2e-6 | 0 | 0 |
| 0.75 | 7.5e-2 | 6.1e-4 | **1.7e-4** | 7.5e-4 |
| 0.90 | 1.2e-1 | 1.4e-2 | **6.4e-3** | 1.5e-2 |

### Stash under churn (QUICK, n = 10^5)

Largest stash over 4n operations, as a fraction of n (replace, uniform / Zipf(1); no rebuild → rebuild every n):
- BFS α=20, load 0.5: 4e-5 / 2e-5 → 1e-5 / 1e-5. Load 0.67: 1.5e-2 / 1.8e-2 → 1.5e-3 / 4.1e-4.
- RW α=10 (discipline), load 0.5: 0.11 / 0.37 → 0.08 / 0.14. Load 0.67: 0.19 / 0.50 → 0.16 / 0.25 (already 2.4% / 5.4% after the fill).
- RW α=10, original first-fit code without invalidation (secure for insertions only), load 0.67: 1.3e-3 / 2.1e-3 → 9.7e-4 / 1.1e-3.
- `update` (same key): 75–86% of keys end in the stash for every disciplined variant (each update uses one of the key's s = 4 states); 18–41% at most between rebuilds every n; the original code stays near 1e-4.

### FrodoPIR fork

Fork vs upstream, 16384 rows × 8 KB (2 cores): setup 26.2 s vs 62.9 s; hint refresh after 11 / 110 / 210 rows 13.5 / 84 / 244 ms vs ≈ 54 s (upstream rebuilds); answers for 64 / 256 queries 0.70 / 2.45 s vs 3.64 / 15.1 s; client prep for 256 queries 1.75 s vs 5.77 s.

Fork vs upstream, 4096 rows × 8 KB: setup 10.5 s vs 16.8 s; hint refresh after 10 rows 14.6 ms vs 17.1 s (rebuild); 16 answers 54 ms vs 343 ms; client prep for 16 queries 58 ms vs 212 ms.

Fork, 16384 rows × 8 KB: hint update 11 rows 8 ms, 110 rows 71 ms, 210 rows 173 ms, 420 rows 437 ms. Server time per batch of queries by width (16 / 64 / 256 queries): width 8: 161 / 596 / 2296 ms; width 16: 109 / 468 / 1893 ms; width 32: 271 / 426 / 1853 ms; width 64: 594 / 592 / 2400 ms. On 2 cores the evaluation is compute-bound, so wider groups gain little; on the server, where a pass over a large database is memory-bound, they should gain more.

FrodoPIR fork vs YPIR+SP, 16384 rows × 8 KB: client hint 37.5 MB vs none; query 64 KB vs 588 KB; response 29 KB vs 37 KB; server, 1 query 107 ms vs 148 ms; 4 queries 117 ms (batched) vs ≈ 590 ms; updates: incremental vs a ≈ 5 s rebuild.

### Spiral (hintless PIR): Blyss server vs `spiral_fast`

`spiral_fast/` is a faster Spiral server with the Blyss client, parameters, and wire format; `spiral_bench check` verifies byte-identical responses against the Blyss server (`blyssprivacy/sdk`, `lib/server`) and that row updates are visible. Changes: a dense, coefficient-major database; several queries per database pass; an AVX-512 kernel with register-resident accumulators and lazy reduction; parallel first dimension (128 tasks per array instead of 4); expansion and folding in parallel across queries; and `update_item` for one record (0.1–0.6 ms).

Server time per query, 2-core VM (Blyss "concurrent" = its best throughput, with queries run in parallel):

| records | Blyss, one at a time | Blyss, 8 concurrent | ours, batch 8 | ours, batch 32 |
|---|---|---|---|---|
| 16K × 8 KB | 575–690 ms | 463 ms | 430–442 ms | 401–414 ms |
| 32K × 8 KB | 819–1011 ms | 703–786 ms | 557–568 ms | 509–516 ms |
| 64K × 8 KB | 1416–2206 ms | 1351–1399 ms | 835–935 ms | 757 ms (batch 16) |
| 64K × 2 KB (DocMap) | 815–943 ms | 499–504 ms | 438–457 ms | 407–425 ms |

- The first dimension is about 4× faster per query (64K × 8 KB: 157 ms per query in a batch on 2 cores, vs 1254 ms single-threaded in the Blyss server).
- What remains is per-query work that cannot be shared across queries: query expansion (≈ 250 ms per query on 2 cores) and folding (70–280 ms, growing with the table). The gain therefore grows with table size: 1.15× at 16K records, 1.8× at 64K. Enron's ValMap (2^17–2^18 rows) should be at the upper end.
- The next lever is the NTT inside expansion and folding (AVX2 in spiral-rs today, per-call allocations).
- Memory: a query in flight holds ≈ 85 MB (8 KB records, nu1 = 9), so batch size is limited by RAM; batch 32 at 64K records did not fit in the 7 GB VM.
- Client: 16 KB query, 21.5 KB response (8 KB records), 1.7 MB public parameters once, no hint.

### YPIR+SP: reference code vs `ypir_fast`

YPIR+SP (Menon–Wu, USENIX Security 2024) is SimplePIR with the hint kept by the server: the query is an RLWE encoding of a one-hot vector over the rows (7 bytes per row), the server multiplies it with the database and ring-packs the resulting LWE ciphertexts, and the hint-dependent part of the packing is precomputed offline. The reference code (`menonsamir/ypir`, a73e550) is single-threaded, answers one query per pass, stores one record per row and has no update path (an update is the offline rebuild). `ypir_fast/` builds on its kernels and client:

- **Records per row.** With no client hint the row width is free. Wider rows shrink the query (7 B per row) and grow the response (12,288 B per 3,584 B of row); `--per-row auto` minimizes the sum. ValMap (316K × 8 KB): 10 records per row, 32,768 rows, 23 outputs, query 229 KB + response 283 KB, against 3.7 MB + 37 KB with one record per row (524,288 rows after padding). DocMap (2.3M × 1059 B): 71 per row, query 229 KB + response 258 KB.
- **Row updates.** The hint is linear in the database, so an update adds `a_ρ · Δ` for the touched 2048-row blocks only (checked against a full recomputation). The packing precomputation is not linear (gadget decomposition) and is redone for every output, in parallel across outputs and across the independent merges of each level (checked equal to the reference `precompute_pack`). Precomputed values are stored compactly (half the reference's memory) and the automorphism tables once rather than per output.
- **Batched, parallel online pass.** All queries of a batch share the database pass (the reference AVX-512 kernel on column tiles, 8 queries per pass), and packing runs in parallel over (query, output).
- **Prepared packing.** The expensive part of online packing depends only on the client's packing keys and the precomputation, not on the query. A client can send its keys ahead (e.g. right after its previous search); the server prepares, and the online answer is the database pass plus adding the LWE bodies (`prepare_many` + `answer_prepared`; valid until the next update).
- **Shared packing keys.** Two queries under one client key and the same public matrix leak the difference of their selection vectors, so the reference needs fresh keys (473 KB) per query. With `--seeds S` the server keeps S public matrices (S hints and precomputations) and up to S queries of a search share one key.

2-core VM, QUICK shapes (rerun on the server):

| shape | reference: 1 query | ours: batch 16, per query | ours: prepared, online, batch 16 | update (1 / 1000 records): reference rebuild | ours |
|---|---|---|---|---|---|
| 20K × 8 KB, 3 per row (8192 rows, 7 outputs) | 276 ms | 160 ms | 8.6 ms | 8.8 s | 4.1 / 5.4 s |
| 20K × 8 KB, 1 per row (32768 rows, 3 outputs) | — | 72 ms | 15 ms | — | 1.5 / 3.0 s |
| 100K × 1 KB, 13 per row (8192 rows, 4 outputs) | — | 74 ms (batch 64) | 4.7 ms (batch 64) | — | 1.9 / 2.9 s |

- On 2 cores the parallel gains are at most 2×; packing (≈ 30 ms of CPU per query and output) dominates, and the database pass is small. Prepared packing removes it from the online path.
- Updating the database rows is one pass over each touched column (parallel), not a strided walk per row; on the server the earlier version spent 1.3 s / 10.5 s copying rows for 10^3 / 10^4 updates.
- An update costs one packing precomputation per output (≈ 0.9 s of CPU each), independent of how many records changed, so updates should be batched (one per flush); the hint delta grows with the touched blocks.
- State per seed: hint plus precomputation, ≈ 0.94 GB for the 7-output shape; at the ValMap shape about 3 GB, so `--seeds 8` needs ≈ 25 GB.
- Cost model (`PirKind::Ypir`): per query `rows × 7 + outputs × 12,288` bytes for the auto shape of the table, plus 473 KB of packing keys per `YPIR_SEEDS` queries (1 by default, i.e. per query); no join download and no hint refresh. Pairings costed: YPIR ValMap with Spiral DocMap, and YPIR for both.

## Paired evaluation (stage `paired`)

Two client models, each costed for six pairings (ValMap PIR, DocMap PIR) from one replay: {FrodoPIR, Spiral}², plus YPIR+SP for ValMap with Spiral for DocMap, and YPIR+SP for both:
- **log clients** copy the append-only log. This covers the rebuild schedules and batched flushes above.
- **stateless clients** (`--stateless`) keep no log. A search downloads the stash-only VolMap (one record per keyword, refreshed by the owner after each batch) and the pending stash (padded to 2B).
  - Every insertion reaches ValMap through the eviction walk, in batches of B with θ = 1: everything pending is committed in each batch, using K = B + B/8 operations.
  - DocMap is a UOKVS keyed (id, j). Edits and removals are invalidation-mode updates, decode keeps the newest copy, and an edit keeps the chunk count.

Per search, a client pays:
- the sync: log records, or VolMap plus the pending stash;
- the stashes;
- for FrodoPIR only, hint deltas or a fresh hint, whichever is smaller;
- the queries and responses of the search, per PIR:
  - FrodoPIR: query = 4 bytes × rows; response = 4 × width;
  - Spiral: query 16,416 B; response 21,504 B for 8 KB rows, 8,192 B for 2 KB rows; no hint, and 1.7 MB of public parameters once.
  - YPIR+SP: query 7 B × rows and response 12,288 B × outputs for the auto shape; 473 KB of packing keys per client key (per query unless the server keeps several seeds); no hint.

`--pir --val-pir frodo|spiral --doc-pir frodo|spiral` runs real databases: Spiral through `spiral_fast` with row updates, FrodoPIR through the fork. The sampled searches then record measured server time. `selftest` checks retrieval through both PIRs before and after an update, and the stateless design against the reference multimap.

The report splits each search into two parts:
- **the keyword lookup**: sync, ValMap stash, ValMap hint refresh, and ValMap queries. This is the part compared with prior MKSE.
- **document retrieval**: DocMap stash, DocMap hint refresh, and DocMap queries.

DocMap queries per document: a document costs one query per 1 KB chunk. The mean over all documents is ≈ 4.6 chunks, but a search's matches are length-biased (a long email contains more keywords, so it matches more searches), and the mean over matched documents is ≈ 13 chunks. This is not padding.

Document retrieval is reported per document fetched, and capped at `--max-docs-fetched` (default 100) per search. Full retrieval of every matching document is shown for reference only, since Enron keywords match up to 10^5 documents. Medians are shown next to means.

The report also has the cheapest configuration per search interval, query counts, and the one-time cost of joining. `design_summary*.csv` files with an older column layout are moved aside automatically.

First small run (synthetic, 1K keywords, 5K documents, 17K appends; client searching every 1,000 appends):

| clients | ValMap / DocMap | per search |
|---|---|---|
| log | FrodoPIR / FrodoPIR | 20.5 MB |
| log | Spiral / Spiral | 4.8 MB |
| stateless | Spiral / FrodoPIR | 98 MB |
| stateless | Spiral / Spiral | 13.7 MB |

- FrodoPIR's query grows with the number of rows (4 bytes per row). Searches that fetch many documents therefore pay heavily for a FrodoPIR DocMap: 1,417 DocMap queries × 170 KB in the stateless run.
- Spiral answers are slower on the server: 2.3 s for about 5 ValMap queries at this small size, against 44 ms for FrodoPIR. So bandwidth and server time favour different pairings, and the paired tables show both.
- The stateless model sends s = 4 DocMap queries per chunk, against 1 for the row-addressed DocMap of log clients.

## Memory and time on the full corpus (rough)

- `write` runs keep only owner state (no PIR) and take minutes each.
- ValMap PIR: rows × 7282 u32 (9-bit plaintexts): 2^18 rows ≈ 7.6 GB of database plus the hint computation. DocMap PIR at 2^22 rows of 1059 bytes is ≈ 15 GB; `writepir` skips it unless `DOC_PIR=1`.
- `stash` at n = 24M needs a few GB per run; 76 runs.
- FrodoPIR client preprocessing is O(m · 1288) per group of queries, so searches with hundreds of queries take seconds on the client.
