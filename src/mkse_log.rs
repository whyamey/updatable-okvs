//! The full dynamic MKSE design (revised Section 4).
//!
//! Per owner:
//! * **ValMap**: UOKVS (Cuckoo, p = 1). A bucket is a chunk of up to C
//!   postings. Chunk keys are versioned by their length: `(w, i, len)`, so a
//!   rewrite of the partial last chunk is an insertion of a fresh key and the
//!   old copy is freed in owner state only (it becomes stale = free).
//! * **Log**: append-only, stash-only map that clients copy incrementally.
//!   Every record has the same size: a PRF tag and an encrypted body.
//!     - posting record, tag F(k_w, "post", epoch, n): w's n-th posting
//!       (or a tombstone)
//!     - flush record, tag F(k_w, "flush", epoch, m): w's committed volume v_c
//!     - document version record, tag F(cap_id, "ver", epoch, m, part)
//!   Every log compaction starts a new log epoch ε_L, so tags never repeat
//!   across epochs. ValMap chunk keys carry the ValMap epoch ε_V, a separate
//!   counter that only a ValMap rebuild advances.
//! * **DocMap**: either rows addressed by number (`DocLayout::Rows`; the row
//!   numbers live in the document version records, chunks are written to
//!   never-used rows, freed rows are reused only after a DocMap compaction)
//!   or a UOKVS keyed by `(id, ver, j)` (`DocLayout::Cuckoo`, for comparison).
//!
//! Schedules (all public; they depend only on the number of log appends):
//! * `Policy::Batched{b, k, theta, age}`: every `b` appends the owner runs
//!   exactly `k` ValMap operations (dummies pad), executed in a uniformly
//!   random order, and appends exactly `k` flush records (dummies pad), also
//!   in a uniformly random order. Keyword w is due once it has `theta` pending
//!   postings or its oldest pending posting is `age` batches old; due keywords
//!   are served first-due-first. Whether w is due depends only on w's own
//!   history; serving more than `k` operations in one batch is a K-overflow
//!   (counted) and delays the rest. Log budget: a posting appended after
//!   batch j is due at batch j + `age` at the latest, i.e. within `age * b`
//!   appends, so with no K-overflow every posting is committed (and the
//!   uncommitted log tail spans) fewer than `age * b` appends.
//!   Phased (`age_phased`, the bench's `--age-appends`): keyword w has a
//!   phase φ(w) in [0, age) from a PRF under an owner-only key (never
//!   delegated) of its keyword index, and is due by age at
//!   batch t if it has pending postings and (t + φ(w)) mod age = 0. Every
//!   pending posting is still committed within `age * b` appends, and the
//!   deadlines of different keywords are spread over the batches, so there is
//!   no synchronized burst (e.g. right after setup). Whether w is due depends
//!   only on w's own history and its hidden phase.
//!   `b = 1, k = 1` is the old one-operation-per-insertion design.
//! * `Policy::Rebuild{r}`: no flushes; a ValMap rebuild every `r` appends
//!   (the static-PIR-plus-rebuild baseline; the same as `Batched` with
//!   `k = 0` and `major_every = r`).
//! * Two public schedules on the append counter t (absolute: a forced
//!   compaction does not shift them):
//!     - **ValMap rebuild** when t is a multiple of E_V (`major_every`, or `r`;
//!       0 = never). It commits every pending posting, re-encodes ValMap
//!       under epoch ε_V + 1, and is always accompanied by a log compaction.
//!     - **Log compaction** when t is a multiple of E_L (`log_every`, 0 = only
//!       with ValMap rebuilds). E_V must be a multiple of E_L, so a scheduled
//!       ValMap rebuild coincides with a scheduled log compaction.
//!   Otherwise only on a public counter: the DocMap has fewer never-written
//!   rows than a write needs (rows layout: an early log compaction with a
//!   repack; with `pend_cap = 0`, which keeps pending postings out of
//!   snapshots, it is a full rebuild as well), or the version map of the
//!   stateless DocMap has no slot left for an edit or removal (fixed layout:
//!   a full rebuild). A stash overflow never triggers a rebuild: it is a
//!   public failure and the structure halts.
//!   Compactions reveal nothing beyond this schedule:
//!     - ValMap (rebuilds only): exactly `val_cap` items are encoded (live chunks plus filler
//!       items under fresh random keys, in one uniformly random order); reuse
//!       mode frees the fillers right after the Encode, keep mode keeps them.
//!       Chunk keys carry the epoch, so every lookup key is fresh per epoch.
//!     - DocMap (rows; at log compactions only, since only version records
//!       carry row numbers): a repack re-places live chunks at random rows and
//!       reserves `doc_live_cap - live` rows, so exactly `doc_rows -
//!       doc_live_cap` never-written rows remain. A repack happens when a
//!       write lacks never-written rows, or at a scheduled compaction when
//!       fewer than `doc_repack_below` remain.
//!     - DocMap (fixed, stateless; at ValMap rebuilds): re-encoded under the
//!       new epoch with exactly `doc_live_cap` items (fillers as for ValMap).
//!     - Log: a snapshot of exactly `kw_pad` flush records (each keyword's
//!       committed count v^c), `pend_cap` posting records (the postings not
//!       yet committed to ValMap, under epoch-(ε_L + 1) tags with their old
//!       posting numbers n in [v^c, v), so a search is unchanged; dummies
//!       pad) and `ver_cap` version records (current versions of live
//!       documents, dummies pad), in one uniformly random order.

use crate::pir_backend::{frodo_hint_bytes, frodo_row_delta_bytes};
use crate::uokvs::{derive, Key, OpOut, OwnerKeys, Params, Table, CHECK, K32, META, NONCE};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

pub const TAG: usize = 16;
pub type Tag = [u8; TAG];

#[derive(Clone, Debug)]
pub enum Policy {
    Batched { b: usize, k: usize, theta: usize, age: usize },
    Rebuild { r: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocLayout {
    Rows,
    Cuckoo,
    /// UOKVS keyed `(id, (epoch, u, j))` without version records in the log
    /// (the stateless client model): version u = 0 of every live document is
    /// encoded at each rebuild; an edit inserts the new version's chunks under
    /// fresh keys (reuse mode frees the old ones), a removal frees them, and
    /// both record (id, u, n_id | deleted) in a padded stash-only version map
    /// that clients download with every search.
    Fixed,
}

#[derive(Clone, Debug)]
pub struct LogCfg {
    /// ValMap table (p = 1; val_len = chunk plaintext bytes)
    pub val: Params,
    pub posting_bytes: usize,
    pub policy: Policy,
    /// E_V: a ValMap rebuild when the append counter is a multiple of this
    /// (0 = never; `Policy::Rebuild{r}` uses r instead). A rebuild commits all
    /// postings (tombstones applied) and comes with a log compaction.
    pub major_every: usize,
    /// E_L: a log compaction when the append counter is a multiple of this
    /// (0 = only together with ValMap rebuilds). E_V must be a multiple of E_L.
    pub log_every: usize,
    /// P̂: posting records in every log snapshot (pending postings, dummies
    /// pad); more pending postings at a compaction is a capacity failure.
    /// 0 is allowed only when every compaction is a ValMap rebuild (E_L = E_V
    /// or E_L = 0); forced compactions then rebuild ValMap too.
    pub pend_cap: usize,
    /// dry runs only: snapshots carry the pending postings without padding
    /// (data-dependent size) so that `pend_snap_max` can size `pend_cap`
    pub pend_measure: bool,
    /// dry runs only: batches serve all due work without padding (K is then
    /// not public); `max_batch_ops` sizes K for the real run
    pub k_measure: bool,
    /// age rule with per-keyword phases (see the module docs)
    pub age_phased: bool,
    pub doc_layout: DocLayout,
    /// DocMap table for `DocLayout::Cuckoo` (val_len = doc row bytes)
    pub doc: Params,
    /// plaintext bytes per DocMap row
    pub doc_row_bytes: usize,
    /// rows of the row-addressed DocMap
    pub doc_rows: usize,
    /// bytes of a log record body (>= posting_bytes, >= 16)
    pub record_body: usize,
    /// public padding of a compacted log: number of keywords
    pub kw_pad: usize,
    /// true: a superseded chunk is freed at once (its bucket can be reused;
    /// needs the invalidation discipline, Theorem `thm:oblivious ops`).
    /// false: superseded chunks stay until the next major compaction and
    /// batches pad with filler insertions instead of dummy operations, so
    /// every ValMap operation is an insertion of a fresh key (Theorem 1).
    pub reuse_freed: bool,
    /// Stateless clients: no synced log. A search downloads the stash-only
    /// VolMap (`kw_pad` entries, refreshed by the owner after each batch) and
    /// the pending-posting stash (padded to `pend_pad`). DocMap is `Fixed`.
    pub stateless: bool,
    pub pend_pad: usize,
    /// public number of ValMap items encoded at every rebuild (live chunks +
    /// fillers); more live chunks than this is a capacity failure (panic)
    pub val_cap: usize,
    /// public cap on live DocMap chunks at a repack/rebuild (rows layout: the
    /// repack leaves `doc_rows - doc_live_cap` never-written rows; fixed
    /// layout: the DocMap Encode has exactly this many items)
    pub doc_live_cap: usize,
    /// rows layout: a scheduled rebuild also repacks the DocMap when fewer
    /// never-written rows than this remain (0 = only when a write needs it)
    pub doc_repack_below: usize,
    /// public number of document version records in a log snapshot
    pub ver_cap: usize,
    /// fixed layout: slots of the padded version map (one per edit or removal
    /// in an epoch); a rebuild runs when an edit or removal finds it full
    pub vmap_pad: usize,
}

impl LogCfg {
    /// E_V, the effective ValMap rebuild period (0 = never)
    pub fn val_every(&self) -> usize {
        match self.policy {
            Policy::Rebuild { r } => r,
            Policy::Batched { .. } => self.major_every,
        }
    }
    /// every scheduled log compaction is a ValMap rebuild
    pub fn coupled(&self) -> bool {
        self.log_every == 0 || self.log_every == self.val_every()
    }
    /// forced compactions (DocMap repacks) are log compactions, not rebuilds
    pub fn pending_in_snapshots(&self) -> bool {
        self.pend_cap > 0 || self.pend_measure
    }
    /// public size of a log snapshot, in records (not in measure mode)
    pub fn snapshot_records(&self) -> usize {
        self.kw_pad + self.pend_cap + self.ver_cap
    }
    pub fn chunk_cap(&self) -> usize {
        self.val.val_len / self.posting_bytes
    }
    pub fn record_bytes(&self) -> usize {
        TAG + NONCE + self.record_body + CHECK
    }
    /// row numbers carried by one document version record
    pub fn rows_per_record(&self) -> usize {
        ((self.record_body - 8) / 4).max(1)
    }
    pub fn doc_records(&self, nch: usize) -> usize {
        match self.doc_layout {
            DocLayout::Rows => ((nch + self.rows_per_record() - 1) / self.rows_per_record()).max(1),
            DocLayout::Cuckoo => 1,
            DocLayout::Fixed => 0,
        }
    }
    pub fn val_row_bytes(&self) -> usize {
        self.val.bucket_bytes()
    }
    pub fn doc_row_cipher_bytes(&self) -> usize {
        match self.doc_layout {
            DocLayout::Rows => NONCE + self.doc_row_bytes + META + CHECK,
            DocLayout::Cuckoo | DocLayout::Fixed => self.doc.bucket_bytes(),
        }
    }
    /// What a stateless client downloads besides PIR answers: VolMap and the
    /// pending stash (each record has the log record size).
    pub fn volmap_bytes(&self) -> usize {
        self.kw_pad * self.record_bytes()
    }
    pub fn pending_stash_bytes(&self) -> usize {
        self.pend_pad * self.record_bytes()
    }
    /// The padded version map of the fixed DocMap layout (one record per slot),
    /// downloaded with every search.
    pub fn vmap_bytes(&self) -> usize {
        if self.doc_layout == DocLayout::Fixed {
            self.vmap_pad * self.record_bytes()
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Posting {
    pub id: u32,
    pub nch: u16,
    pub tomb: bool,
}

#[derive(Clone, Debug)]
enum Body {
    Post(Posting),
    Flush { vc: u32 },
    DocVer { ver: u32, nch: u16, deleted: bool, rows: Vec<u32> },
    Dummy,
}

/// One planned ValMap chunk write of a batch: an insertion of a fresh key
/// and, in reuse mode, the owner-local Free of the chunk it supersedes.
struct ChunkWrite {
    free: Option<Key>,
    key: Key,
    keys: (K32, K32),
    val: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Rec {
    tag: Tag,
    body: Body,
}

struct Kw {
    name: String,
    cap_val: (K32, K32),
    cap_log: K32,
    /// committed postings, in order (owner copy of the chunks)
    committed: Vec<Posting>,
    /// pending postings with the batch in which each was appended
    pend: Vec<(Posting, u64)>,
    /// posting records appended in total (= volume v, tombstones included)
    n: u32,
    /// flush records in the current epoch
    m: u32,
    due_since: Option<u64>,
    /// phase of the phased age rule: owner-only PRF of the keyword index, in [0, 2^32)
    phase: u32,
}

pub struct Doc {
    cap: (K32, K32),
    pub ver: u32,
    pub nch: u16,
    rows: Vec<u32>,
    pub deleted: bool,
    /// version records in the current epoch
    m: u32,
    keywords: Vec<u32>,
    /// fixed layout: the DocMap keys of the current version are
    /// `(id, (dk_ep, u, j))`; `u` counts versions within epoch `dk_ep`
    u: u32,
    dk_ep: u32,
}

/// What one owner action did, for accounting.
#[derive(Clone, Debug, Default)]
pub struct Action {
    pub appends: usize,
    pub val_ops: Vec<OpOut>,
    pub doc_rows_written: Vec<usize>,
    pub doc_ops: Vec<OpOut>,
    pub batches: Vec<BatchOut>,
    /// log compactions (with or without a ValMap rebuild)
    pub compactions: usize,
    /// ValMap rebuilds (owner ms each); each comes with a log compaction
    pub rebuilds: Vec<f64>,
    /// owner ms of log compactions without a ValMap rebuild
    pub log_only_ms: Vec<f64>,
    pub doc_compactions: usize,
    /// fixed layout: refreshes of the padded version map (edits, removals)
    pub vmap_refreshes: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BatchOut {
    pub index: u64,
    pub real_ops: usize,
    pub dummy_ops: usize,
    /// flush records appended (K, or the keywords served in measure mode)
    pub records: usize,
    pub keywords_served: usize,
    pub postings_flushed: usize,
    pub due: usize,
    pub deferred: usize,
    pub accessed: Vec<usize>,
    pub owner_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct SearchOut {
    pub ids: Vec<u32>,
    pub volume: usize,
    pub committed: usize,
    pub pending: usize,
    pub chunks_queried: usize,
    pub val_queries: Vec<usize>,
    pub doc_queries: Vec<usize>,
    /// DocMap queries of each returned document, in result order
    pub doc_queries_per_doc: Vec<usize>,
    pub tag_probes: usize,
}

impl SearchOut {
    /// DocMap queries needed to fetch only the first `cap` documents.
    pub fn doc_queries_capped(&self, cap: usize) -> usize {
        self.doc_queries_per_doc.iter().take(cap).sum()
    }
}

pub struct MkseLog {
    pub cfg: LogCfg,
    ki: OwnerKeys,
    kl: OwnerKeys,
    /// owner-only PRF key of the age phases
    kphase: K32,
    kd: OwnerKeys,
    pub val: Table,
    pub val_epoch: u32,
    kws: Vec<Kw>,
    kw_ix: HashMap<String, u32>,
    pending_kws: BTreeSet<u32>,
    pub docs: HashMap<u32, Doc>,
    // DocMap (rows layout)
    doc_rows: Vec<Option<(u32, u16)>>,
    virgin: Vec<u32>,
    pub doc_freed: usize,
    // DocMap (cuckoo layout)
    pub doc_table: Option<Table>,
    // log
    log: Vec<Rec>,
    tag_ix: HashMap<Tag, usize>,
    pub epoch: u32,
    pub log_live_since_compaction: usize,
    pub compacted_records: usize,
    // schedule
    pub batch: u64,
    since_batch: usize,
    /// randomness of snapshot padding and order only, so that the rest of a
    /// run (and its random workload) does not depend on `pend_cap`: a dry run
    /// with `pend_measure` sees exactly the pending counts of the real run
    srng: StdRng,
    /// randomness of batch execution (see `run_batch`)
    brng: StdRng,
    /// most real ValMap operations a batch needed (to size K)
    pub max_batch_ops: usize,
    /// the same over batches after the first two age windows (age rule):
    /// leaves out the burst when the postings of the first batch interval
    /// after setup all come due together
    pub max_batch_ops_steady: usize,
    /// kappa_A statistics: windows of this many appends (0 = off; set by the
    /// caller): distinct keywords that got a posting record, and that were
    /// committed by a batch, per complete window
    pub kappa_window: usize,
    win_active: HashSet<u32>,
    win_committed: HashSet<u32>,
    pub kappa_active: Vec<usize>,
    pub kappa_committed: Vec<usize>,
    // counters
    pub log_compactions: usize,
    pub val_rebuilds: usize,
    /// largest number of pending postings a log snapshot had to carry
    pub pend_snap_max: usize,
    pub total_appends: usize,
    pub k_overflow_batches: usize,
    pub deferred_total: usize,
    pub pend_total: usize,
    pub pend_peak: usize,
    pub max_pend_age: u64,
    /// stateless: batches after which more postings were pending than the
    /// padded pending stash holds (the pad would have to grow: a leak)
    pub pend_pad_overflow: usize,
    /// filler insertions and superseded chunks kept (reuse_freed = false)
    pub fillers: u64,
    pub superseded: u64,
    /// fixed layout: version map (id -> (u, n_id, deleted)) of the documents
    /// edited or removed in the current epoch, and the slots used (one per
    /// edit or removal, a public count)
    vmap: HashMap<u32, (u32, u16, bool)>,
    pub vmap_ops: usize,
    pub vmap_peak: usize,
    /// a public failure (stash overflow of a table, or more pending postings
    /// than the padded pending stash holds) halted the structure
    pub halted: bool,
    pub halt_reason: String,
    /// largest number of live items a rebuild had to encode / place / write
    /// (to check the public caps val_cap, doc_live_cap, ver_cap)
    pub val_live_max: usize,
    /// live (committed, current) ValMap chunks now: sum over keywords of
    /// ceil(committed / C); capped by val_cap at all times
    pub val_live: usize,
    pub doc_live_max: usize,
    pub ver_max: usize,
    /// items passed to the last padded Encode (= its public cap)
    pub last_encode: usize,
}

fn tag_of(k: &K32, dom: &[u8], epoch: u32, a: u64, b: u32) -> Tag {
    let mut h = blake3::Hasher::new_keyed(k);
    h.update(dom);
    h.update(&epoch.to_le_bytes());
    h.update(&a.to_le_bytes());
    h.update(&b.to_le_bytes());
    let mut t = [0u8; TAG];
    t.copy_from_slice(&h.finalize().as_bytes()[..TAG]);
    t
}

pub fn enc_postings(ps: &[Posting], pb: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(ps.len() * pb);
    for p in ps {
        let start = v.len();
        v.extend_from_slice(&p.id.to_le_bytes());
        v.extend_from_slice(&p.nch.to_le_bytes());
        v.push(p.tomb as u8);
        // the 32-byte DocMap capability: its bytes only matter for size
        v.resize(start + pb, 0);
    }
    v
}

pub fn dec_postings(b: &[u8], pb: usize) -> Vec<Posting> {
    b.chunks(pb)
        .filter(|c| c.len() >= 7)
        .map(|c| Posting {
            id: u32::from_le_bytes(c[0..4].try_into().unwrap()),
            nch: u16::from_le_bytes(c[4..6].try_into().unwrap()),
            tomb: c[6] != 0,
        })
        .collect()
}

impl MkseLog {
    // ------------------------------------------------------------ keys

    fn chunk_key(&self, w: &str, i: usize, len: usize) -> Key {
        Key::sub(w.as_bytes(), ((self.val_epoch as u64) << 48) | ((i as u64) << 16) | len as u64)
    }
    fn chunk_keys(&self, kw: &Kw, key: &Key) -> (K32, K32) {
        derive(&kw.cap_val, key.idx)
    }
    fn doc_key(id: u32, ver: u32, j: u16) -> Key {
        Key::sub(&id.to_le_bytes(), ((ver as u64) << 16) | j as u64)
    }
    /// fixed layout: chunk j of version u (within epoch ep) of document id
    fn doc_key_fixed(id: u32, ep: u32, u: u32, j: u16) -> Key {
        assert!(ep < 1 << 16, "too many epochs for the key encoding");
        Key::sub(&id.to_le_bytes(), ((ep as u64) << 48) | ((u as u64) << 16) | j as u64)
    }
    fn halt(&mut self, why: String) {
        if !self.halted {
            self.halted = true;
            self.halt_reason = why;
        }
    }
    /// A table whose stash overflowed has halted; so does the whole structure.
    fn check_tables(&mut self) {
        if self.val.halted {
            self.halt(format!("ValMap stash overflow (more than sigma = {} items) after {} appends", self.val.prm.sigma, self.total_appends));
        }
        if self.doc_table.as_ref().map_or(false, |t| t.halted) {
            let sg = self.doc_table.as_ref().unwrap().prm.sigma;
            self.halt(format!("DocMap stash overflow (more than sigma = {sg} items) after {} appends", self.total_appends));
        }
    }
    fn assert_live(&self) {
        assert!(!self.halted, "the structure halted after a public failure: {}", self.halt_reason);
    }
    /// Fresh random filler item (fresh random keys; never looked up).
    fn filler<R: Rng>(&mut self, rng: &mut R) -> (Key, (K32, K32), Vec<u8>) {
        self.fillers += 1;
        let key = Key::sub(b"\0filler", self.fillers);
        let mut lk = [0u8; 32];
        let mut dk = [0u8; 32];
        rng.fill(&mut lk);
        rng.fill(&mut dk);
        (key, (lk, dk), vec![])
    }
    /// Encode of a rebuilt table with exactly `cap` items: the live entries
    /// padded with fillers, all in one uniformly random order. Reuse mode
    /// frees the fillers (owner-local) right after the Encode; keep mode keeps
    /// them, so the occupancy stays public all epoch.
    fn encode_padded<R: Rng>(&mut self, prm: Params, mut entries: Vec<(Key, (K32, K32), Vec<u8>)>, cap: usize, what: &str, rng: &mut R) -> Table {
        let live = entries.len();
        assert!(live <= cap, "{what} capacity exceeded: {live} live items at a rebuild > public cap {cap}");
        let mut fill = Vec::with_capacity(cap - live);
        for _ in live..cap {
            let f = self.filler(rng);
            fill.push(f.0.clone());
            entries.push(f);
        }
        entries.shuffle(rng);
        self.last_encode = entries.len();
        let mut t = Table::new(prm, rng);
        t.encode(entries, rng);
        if self.cfg.reuse_freed {
            for k in &fill {
                t.remove(k);
            }
        }
        t
    }
    pub fn doc_nch(&self, len: usize) -> u16 {
        ((len + self.cfg.doc_row_bytes - 1) / self.cfg.doc_row_bytes).clamp(1, u16::MAX as usize) as u16
    }

    // ------------------------------------------------------------ log

    fn push_rec(&mut self, tag: Tag, body: Body) {
        let i = self.log.len();
        self.tag_ix.insert(tag, i);
        self.log.push(Rec { tag, body });
        self.log_live_since_compaction += 1;
    }

    fn append_flush(&mut self, w: u32) {
        let (cap, ep, m, vc) = {
            let k = &self.kws[w as usize];
            (k.cap_log, self.epoch, k.m, k.committed.len() as u32)
        };
        let t = tag_of(&cap, b"flush", ep, m as u64, 0);
        self.push_rec(t, Body::Flush { vc });
        self.kws[w as usize].m += 1;
    }

    fn append_dummy<R: Rng>(&mut self, rng: &mut R) {
        let mut t = [0u8; TAG];
        rng.fill(&mut t);
        self.log.push(Rec { tag: t, body: Body::Dummy });
        self.log_live_since_compaction += 1;
    }

    fn append_doc_version(&mut self, id: u32) {
        if self.cfg.doc_layout == DocLayout::Fixed {
            return;
        }
        let (cap, ver, nch, deleted, rows, m) = {
            let d = &self.docs[&id];
            (d.cap.0, d.ver, d.nch, d.deleted, d.rows.clone(), d.m)
        };
        let parts = self.cfg.doc_records(nch as usize);
        let rpr = self.cfg.rows_per_record();
        for part in 0..parts {
            let slice: Vec<u32> = if self.cfg.doc_layout == DocLayout::Rows && !deleted {
                rows.iter().skip(part * rpr).take(rpr).copied().collect()
            } else {
                vec![]
            };
            let t = tag_of(&cap, b"ver", self.epoch, m as u64, part as u32);
            self.push_rec(t, Body::DocVer { ver, nch, deleted, rows: slice });
        }
        self.docs.get_mut(&id).unwrap().m += 1;
    }

    // ------------------------------------------------------------ setup

    /// `mm`: keyword -> ids at setup; `doc_len`: id -> bytes.
    pub fn setup<R: Rng>(
        cfg: LogCfg,
        mm: &[(String, Vec<u32>)],
        all_keywords: &[String],
        doc_len: &HashMap<u32, usize>,
        default_doc_len: usize,
        rng: &mut R,
    ) -> Self {
        let (ve, le) = (cfg.val_every(), cfg.log_every);
        assert!(ve == 0 || le == 0 || ve % le == 0, "the ValMap rebuild period E_V = {ve} must be a multiple of the log compaction period E_L = {le}");
        assert!(
            cfg.coupled() || cfg.pending_in_snapshots(),
            "log compactions between ValMap rebuilds (E_L = {le}, E_V = {ve}) carry pending postings: pend_cap must be > 0"
        );
        let ki = OwnerKeys::random(rng);
        let kl = OwnerKeys::random(rng);
        let kd = OwnerKeys::random(rng);
        let mut kphase = [0u8; 32];
        rng.fill(&mut kphase);
        let val = Table::new(cfg.val.clone(), rng);
        let doc_table = match cfg.doc_layout {
            DocLayout::Cuckoo | DocLayout::Fixed => Some(Table::new(cfg.doc.clone(), rng)),
            DocLayout::Rows => None,
        };
        let mut s = Self {
            ki,
            kl,
            kphase,
            kd,
            val,
            val_epoch: 0,
            kws: Vec::new(),
            kw_ix: HashMap::new(),
            pending_kws: BTreeSet::new(),
            docs: HashMap::new(),
            doc_rows: vec![None; if cfg.doc_layout == DocLayout::Rows { cfg.doc_rows } else { 0 }],
            virgin: Vec::new(),
            doc_freed: 0,
            doc_table,
            log: Vec::new(),
            tag_ix: HashMap::new(),
            epoch: 0,
            log_live_since_compaction: 0,
            compacted_records: 0,
            batch: 0,
            since_batch: 0,
            srng: StdRng::seed_from_u64(rng.gen()),
            brng: StdRng::seed_from_u64(rng.gen()),
            max_batch_ops: 0,
            max_batch_ops_steady: 0,
            kappa_window: 0,
            win_active: HashSet::new(),
            win_committed: HashSet::new(),
            kappa_active: Vec::new(),
            kappa_committed: Vec::new(),
            log_compactions: 0,
            val_rebuilds: 0,
            pend_snap_max: 0,
            total_appends: 0,
            k_overflow_batches: 0,
            deferred_total: 0,
            pend_total: 0,
            pend_peak: 0,
            max_pend_age: 0,
            pend_pad_overflow: 0,
            fillers: 0,
            superseded: 0,
            vmap: HashMap::new(),
            vmap_ops: 0,
            vmap_peak: 0,
            halted: false,
            halt_reason: String::new(),
            val_live_max: 0,
            val_live: 0,
            doc_live_max: 0,
            ver_max: 0,
            last_encode: 0,
            cfg,
        };
        if s.cfg.doc_layout == DocLayout::Rows {
            s.virgin = (0..s.cfg.doc_rows as u32).collect();
            s.virgin.shuffle(rng);
        }
        for w in all_keywords {
            s.kw_index(w);
        }
        // documents
        let mut doc_kws: HashMap<u32, Vec<u32>> = HashMap::new();
        for (w, ids) in mm {
            let wi = s.kw_index(w);
            for &id in ids {
                doc_kws.entry(id).or_default().push(wi);
            }
        }
        let mut ids: Vec<u32> = doc_kws.keys().copied().collect();
        ids.sort_unstable();
        let mut cuckoo_entries = Vec::new();
        for &id in &ids {
            let len = *doc_len.get(&id).unwrap_or(&default_doc_len);
            let nch = s.doc_nch(len);
            let cap = s.kd.cap(&id.to_le_bytes());
            let mut d = Doc { cap, ver: 0, nch, rows: vec![], deleted: false, m: 0, keywords: doc_kws.remove(&id).unwrap(), u: 0, dk_ep: 0 };
            match s.cfg.doc_layout {
                DocLayout::Rows => {
                    for j in 0..nch {
                        let r = s.virgin.pop().expect("DocMap rows exhausted at setup; raise --doc-spare");
                        s.doc_rows[r as usize] = Some((id, j));
                        d.rows.push(r);
                    }
                }
                DocLayout::Cuckoo | DocLayout::Fixed => {
                    for j in 0..nch {
                        let k = if s.cfg.doc_layout == DocLayout::Fixed { Self::doc_key_fixed(id, 0, 0, j) } else { Self::doc_key(id, 0, j) };
                        let keys = derive(&cap, k.idx);
                        cuckoo_entries.push((k, keys, vec![0u8; 1]));
                    }
                }
            }
            s.docs.insert(id, d);
        }
        match s.cfg.doc_layout {
            DocLayout::Cuckoo => s.doc_table.as_mut().unwrap().encode(cuckoo_entries, rng),
            DocLayout::Fixed => {
                // the same padded Encode as at every rebuild
                s.doc_live_max = s.doc_live_max.max(cuckoo_entries.len());
                let (prm, cap) = (s.cfg.doc.clone(), s.cfg.doc_live_cap);
                s.doc_table = Some(s.encode_padded(prm, cuckoo_entries, cap, "DocMap (--doc-live-cap)", rng));
            }
            DocLayout::Rows => {}
        }
        // ValMap: canonical chunks (full chunks, then one partial chunk)
        let c = s.cfg.chunk_cap();
        let mut entries = Vec::new();
        for (w, ids) in mm {
            let wi = s.kw_index(w) as usize;
            let ps: Vec<Posting> = ids.iter().map(|&id| Posting { id, nch: s.docs[&id].nch, tomb: false }).collect();
            for (i, ch) in ps.chunks(c).enumerate() {
                let key = s.chunk_key(w, i, ch.len());
                let keys = s.chunk_keys(&s.kws[wi], &key);
                entries.push((key, keys, enc_postings(ch, s.cfg.posting_bytes)));
            }
            let kw = &mut s.kws[wi];
            kw.n = ps.len() as u32;
            kw.committed = ps;
        }
        // the same padded Encode (exactly val_cap items) as at every rebuild
        s.val_live_max = s.val_live_max.max(entries.len());
        s.val_live = entries.len();
        let (prm, cap) = (s.cfg.val.clone(), s.cfg.val_cap);
        s.val = s.encode_padded(prm, entries, cap, "ValMap (--val-cap)", rng);
        s.check_tables();
        // initial log = epoch-0 snapshot
        s.write_snapshot();
        s
    }

    fn kw_index(&mut self, w: &str) -> u32 {
        if let Some(&i) = self.kw_ix.get(w) {
            return i;
        }
        let i = self.kws.len() as u32;
        let cap_val = self.ki.cap(w.as_bytes());
        let cap_log = self.kl.cap(w.as_bytes()).0;
        // owner-only PRF key (not derived from any delegated key)
        let ph = tag_of(&self.kphase, b"phase", 0, i as u64, 0);
        let phase = u32::from_le_bytes(ph[..4].try_into().unwrap());
        self.kws.push(Kw {
            name: w.to_string(),
            cap_val,
            cap_log,
            committed: vec![],
            pend: vec![],
            n: 0,
            m: 0,
            due_since: None,
            phase,
        });
        self.kw_ix.insert(w.to_string(), i);
        i
    }

    /// Snapshot padding: a dummy record with a random tag (from `srng`).
    fn snap_dummies(&mut self, n: usize) {
        for _ in 0..n {
            let mut t = [0u8; TAG];
            self.srng.fill(&mut t);
            self.log.push(Rec { tag: t, body: Body::Dummy });
        }
    }

    /// Writes a fresh log for the current epoch: exactly `kw_pad` flush
    /// records (one per keyword with postings, holding its committed count
    /// v^c; dummies pad), exactly `pend_cap` posting records (the pending
    /// postings v^c, ..., v - 1 of every keyword under the new epoch's tags
    /// with their old posting numbers; dummies pad; none right after a ValMap
    /// rebuild) and exactly `ver_cap` version records (the current version of
    /// every live document; removed documents are dropped; dummies pad), all
    /// in a uniformly random order. Its size is public.
    fn write_snapshot(&mut self) {
        self.log.clear();
        self.tag_ix.clear();
        let mut kw_records = 0usize;
        let mut pend_records = 0usize;
        for w in 0..self.kws.len() as u32 {
            self.kws[w as usize].m = 0;
            if self.kws[w as usize].n == 0 {
                continue;
            }
            self.append_flush(w);
            kw_records += 1;
            // pending postings keep their numbers n in [v^c, v)
            let (cap, vc, ps) = {
                let k = &self.kws[w as usize];
                debug_assert_eq!(k.committed.len() + k.pend.len(), k.n as usize);
                (k.cap_log, k.committed.len(), k.pend.iter().map(|x| x.0).collect::<Vec<_>>())
            };
            for (j, p) in ps.into_iter().enumerate() {
                let t = tag_of(&cap, b"post", self.epoch, (vc + j) as u64, 0);
                self.push_rec(t, Body::Post(p));
                pend_records += 1;
            }
        }
        debug_assert_eq!(pend_records, self.pend_total);
        assert!(kw_records <= self.cfg.kw_pad, "snapshot needs {kw_records} flush records > kw_pad = {}", self.cfg.kw_pad);
        self.snap_dummies(self.cfg.kw_pad - kw_records);
        self.pend_snap_max = self.pend_snap_max.max(pend_records);
        if !self.cfg.pend_measure {
            assert!(
                pend_records <= self.cfg.pend_cap,
                "log snapshot capacity exceeded: {pend_records} pending postings at a log compaction > pend_cap = {}; raise --pend-cap",
                self.cfg.pend_cap
            );
            self.snap_dummies(self.cfg.pend_cap - pend_records);
        }
        let mut ids: Vec<u32> = self.docs.keys().copied().collect();
        ids.sort_unstable();
        let mut ver_records = 0usize;
        for id in ids {
            let d = self.docs.get_mut(&id).unwrap();
            d.m = 0;
            if d.deleted {
                continue;
            }
            ver_records += self.cfg.doc_records(d.nch as usize);
            self.append_doc_version(id);
        }
        self.ver_max = self.ver_max.max(ver_records);
        assert!(
            ver_records <= self.cfg.ver_cap,
            "log snapshot capacity exceeded: {ver_records} version records of live documents > ver_cap = {}; raise --ver-cap",
            self.cfg.ver_cap
        );
        self.snap_dummies(self.cfg.ver_cap - ver_records);
        // one uniformly random order; clients find records by tag
        self.log.shuffle(&mut self.srng);
        self.tag_ix.clear();
        for (i, r) in self.log.iter().enumerate() {
            if !matches!(r.body, Body::Dummy) {
                self.tag_ix.insert(r.tag, i);
            }
        }
        if !self.cfg.pend_measure {
            assert_eq!(self.log.len(), self.cfg.snapshot_records(), "the snapshot size must be public");
        }
        self.compacted_records = self.log.len();
        self.log_live_since_compaction = 0;
    }

    // ------------------------------------------------------------ owner ops

    fn tick<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        self.total_appends += 1;
        self.since_batch += 1;
        act.appends += 1;
        if self.halted {
            return;
        }
        if let Policy::Batched { b, k, .. } = self.cfg.policy {
            if k > 0 && self.since_batch >= b {
                self.since_batch = 0;
                let bo = self.run_batch();
                act.batches.push(bo);
                if self.halted {
                    return;
                }
            }
        }
        // kappa_A: distinct keywords with a posting record / committed per window
        if self.kappa_window > 0 && self.total_appends % self.kappa_window == 0 {
            self.kappa_active.push(self.win_active.len());
            self.kappa_committed.push(self.win_committed.len());
            self.win_active.clear();
            self.win_committed.clear();
        }
        // the two public schedules, on the absolute append counter
        let (ve, le, t) = (self.cfg.val_every(), self.cfg.log_every, self.total_appends);
        if ve > 0 && t % ve == 0 {
            self.major(act, rng);
        } else if le > 0 && t % le == 0 {
            self.log_compaction(act, rng);
        }
    }

    /// ValMap rebuild (with a log compaction): rebuild ValMap (all postings,
    /// tombstones applied, padded to `val_cap`), repack the DocMap if fewer
    /// than `doc_repack_below` never-written rows remain (rows layout) or
    /// re-encode it (fixed layout), and rewrite the log in a new epoch.
    pub fn major<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        self.major_inner(false, act, rng);
    }

    /// `repack`: the DocMap lacks never-written rows for a write (public).
    fn major_inner<R: Rng>(&mut self, repack: bool, act: &mut Action, rng: &mut R) {
        let t = Instant::now();
        if self.cfg.doc_layout == DocLayout::Rows && (repack || self.virgin.len() < self.cfg.doc_repack_below) {
            self.compact_docmap(rng);
            act.doc_compactions += 1;
        }
        self.rebuild_val(rng);
        if self.cfg.doc_layout == DocLayout::Fixed {
            self.rebuild_doc(rng);
            act.doc_compactions += 1;
        }
        self.compact(rng);
        self.check_tables();
        self.val_rebuilds += 1;
        act.rebuilds.push(t.elapsed().as_secs_f64() * 1e3);
        act.compactions += 1;
    }

    /// Log compaction without a ValMap rebuild: repack the DocMap if fewer
    /// than `doc_repack_below` never-written rows remain (rows layout), and
    /// write a snapshot in a new log epoch that carries the pending postings.
    pub fn log_compaction<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        self.log_compaction_inner(false, act, rng);
    }

    fn log_compaction_inner<R: Rng>(&mut self, repack: bool, act: &mut Action, rng: &mut R) {
        let t = Instant::now();
        if self.cfg.doc_layout == DocLayout::Rows && (repack || self.virgin.len() < self.cfg.doc_repack_below) {
            self.compact_docmap(rng);
            act.doc_compactions += 1;
        }
        self.compact(rng);
        act.log_only_ms.push(t.elapsed().as_secs_f64() * 1e3);
        act.compactions += 1;
    }

    /// Rows layout: before a write of `n` chunks, compact the log early (with
    /// a DocMap repack) if fewer than `n` never-written rows remain. Both
    /// counts are public: every write reveals its chunk count, and a repack
    /// resets the never-written rows to `doc_rows - doc_live_cap`. With
    /// `pend_cap = 0` (snapshots never carry pending postings) the forced
    /// compaction also rebuilds ValMap.
    fn ensure_doc_rows<R: Rng>(&mut self, n: usize, act: &mut Action, rng: &mut R) {
        if self.cfg.doc_layout == DocLayout::Rows && self.virgin.len() < n {
            if self.cfg.pending_in_snapshots() {
                self.log_compaction_inner(true, act, rng);
            } else {
                self.major_inner(true, act, rng);
            }
            assert!(
                self.virgin.len() >= n,
                "DocMap too small even after a repack ({} never-written rows, {n} needed); raise --doc-spare or lower --doc-live-cap",
                self.virgin.len()
            );
        }
    }

    /// Fixed layout: an edit or removal takes a version-map slot; rebuild
    /// first if none is left (the slot count is public).
    fn ensure_vmap_slot<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        if self.cfg.doc_layout == DocLayout::Fixed {
            if self.vmap_ops >= self.cfg.vmap_pad {
                self.major_inner(false, act, rng);
            }
            assert!(self.vmap_ops < self.cfg.vmap_pad, "--vmap-pad must be at least 1");
            self.vmap_ops += 1;
        }
    }

    fn set_vmap(&mut self, id: u32, e: (u32, u16, bool), act: &mut Action) {
        self.vmap.insert(id, e);
        assert!(self.vmap.len() <= self.vmap_ops.max(1), "version map holds more entries than slots used");
        self.vmap_peak = self.vmap_peak.max(self.vmap.len());
        act.vmap_refreshes += 1;
    }

    fn add_posting<R: Rng>(&mut self, w: u32, p: Posting, act: &mut Action, rng: &mut R) {
        let (cap, n) = {
            let k = &self.kws[w as usize];
            (k.cap_log, k.n)
        };
        let t = tag_of(&cap, b"post", self.epoch, n as u64, 0);
        self.push_rec(t, Body::Post(p));
        let b = self.batch;
        let k = &mut self.kws[w as usize];
        k.n += 1;
        k.pend.push((p, b));
        self.pending_kws.insert(w);
        if self.kappa_window > 0 {
            self.win_active.insert(w);
        }
        self.pend_total += 1;
        self.pend_peak = self.pend_peak.max(self.pend_total);
        self.tick(act, rng);
    }

    /// Inserts posting (w, id); a new id also inserts the document.
    pub fn insert<R: Rng>(&mut self, w: &str, id: u32, doc_len: usize, rng: &mut R) -> Action {
        self.assert_live();
        let mut act = Action::default();
        let wi = self.kw_index(w);
        if !self.docs.contains_key(&id) || self.docs[&id].deleted {
            self.insert_doc(id, doc_len, &mut act, rng);
        }
        let d = self.docs.get_mut(&id).unwrap();
        d.keywords.push(wi);
        let nch = d.nch;
        self.add_posting(wi, Posting { id, nch, tomb: false }, &mut act, rng);
        self.check_tables();
        act
    }

    pub fn delete_posting<R: Rng>(&mut self, w: &str, id: u32, rng: &mut R) -> Action {
        self.assert_live();
        let mut act = Action::default();
        let wi = self.kw_index(w);
        if let Some(d) = self.docs.get_mut(&id) {
            d.keywords.retain(|&x| x != wi);
        }
        self.add_posting(wi, Posting { id, nch: 0, tomb: true }, &mut act, rng);
        self.check_tables();
        act
    }

    fn alloc_rows(&mut self, n: usize) -> Vec<u32> {
        assert!(self.virgin.len() >= n, "DocMap write without enough never-written rows (ensure_doc_rows not called)");
        (0..n).map(|_| self.virgin.pop().unwrap()).collect()
    }

    fn write_doc_chunks<R: Rng>(&mut self, id: u32, act: &mut Action, rng: &mut R) {
        let (nch, ver, cap) = {
            let d = &self.docs[&id];
            (d.nch, d.ver, d.cap)
        };
        match self.cfg.doc_layout {
            DocLayout::Rows => {
                let rows = self.alloc_rows(nch as usize);
                for (j, &r) in rows.iter().enumerate() {
                    self.doc_rows[r as usize] = Some((id, j as u16));
                    act.doc_rows_written.push(r as usize);
                }
                self.docs.get_mut(&id).unwrap().rows = rows;
            }
            DocLayout::Cuckoo => {
                let t = self.doc_table.as_mut().unwrap();
                for j in 0..nch {
                    let k = Self::doc_key(id, ver, j);
                    let (lk, dk) = derive(&cap, k.idx);
                    act.doc_ops.push(t.insert(&k, lk, dk, vec![0u8; 1], rng));
                }
            }
            DocLayout::Fixed => {
                // a fresh version u of the current epoch: fresh keys (id, (ep, u, j))
                let ep = self.val_epoch;
                let u = {
                    let d = self.docs.get_mut(&id).unwrap();
                    d.u = if d.dk_ep == ep { d.u + 1 } else { 0 };
                    d.dk_ep = ep;
                    d.u
                };
                let t = self.doc_table.as_mut().unwrap();
                for j in 0..nch {
                    let k = Self::doc_key_fixed(id, ep, u, j);
                    let (lk, dk) = derive(&cap, k.idx);
                    act.doc_ops.push(t.insert(&k, lk, dk, vec![(ver % 251) as u8; 1], rng));
                }
                // version 0 of an epoch is the default; later ones (edits, a
                // re-insertion after a removal in this epoch) are announced
                if u > 0 {
                    self.set_vmap(id, (u, nch, false), act);
                }
            }
        }
    }

    fn free_doc_chunks(&mut self, id: u32) {
        let (nch, ver, rows, u, ep) = {
            let d = &self.docs[&id];
            (d.nch, d.ver, d.rows.clone(), d.u, d.dk_ep)
        };
        match self.cfg.doc_layout {
            DocLayout::Rows => {
                for r in rows {
                    self.doc_rows[r as usize] = None;
                    self.doc_freed += 1;
                }
            }
            DocLayout::Cuckoo => {
                let t = self.doc_table.as_mut().unwrap();
                for j in 0..nch {
                    t.remove(&Self::doc_key(id, ver, j));
                }
            }
            // owner-local Free of the old version (reuse mode); keep mode
            // leaves it until the next rebuild
            DocLayout::Fixed => {
                if self.cfg.reuse_freed {
                    let t = self.doc_table.as_mut().unwrap();
                    for j in 0..nch {
                        let removed = t.remove(&Self::doc_key_fixed(id, ep, u, j));
                        debug_assert!(removed, "missing DocMap chunk {id}/{ep}/{u}/{j}");
                    }
                } else {
                    self.superseded += nch as u64;
                }
            }
        }
        self.docs.get_mut(&id).unwrap().rows.clear();
    }

    fn insert_doc<R: Rng>(&mut self, id: u32, len: usize, act: &mut Action, rng: &mut R) {
        let nch = self.doc_nch(len);
        self.ensure_doc_rows(nch as usize, act, rng);
        let prev = self.docs.get(&id).map(|d| (d.ver, d.m, d.u, d.dk_ep));
        let cap = self.kd.cap(&id.to_le_bytes());
        let (ver, m, u, dk_ep) = prev.map(|(v, m, u, e)| (v + 1, m, u, e)).unwrap_or((0, 0, 0, u32::MAX));
        self.docs.insert(id, Doc { cap, ver, nch, rows: vec![], deleted: false, m, keywords: vec![], u, dk_ep });
        self.write_doc_chunks(id, act, rng);
        self.append_doc_version(id);
        for _ in 0..self.cfg.doc_records(nch as usize) {
            self.tick(act, rng);
        }
    }

    /// Replaces a document's content (new length `len`).
    pub fn update_doc<R: Rng>(&mut self, id: u32, len: usize, rng: &mut R) -> Action {
        self.assert_live();
        let mut act = Action::default();
        if self.docs.get(&id).map_or(true, |d| d.deleted) {
            return act;
        }
        let nch = self.doc_nch(len);
        self.ensure_doc_rows(nch as usize, &mut act, rng);
        self.ensure_vmap_slot(&mut act, rng);
        self.free_doc_chunks(id);
        {
            let d = self.docs.get_mut(&id).unwrap();
            d.ver += 1;
            d.nch = nch;
        }
        self.write_doc_chunks(id, &mut act, rng);
        self.append_doc_version(id);
        for _ in 0..self.cfg.doc_records(nch as usize) {
            self.tick(&mut act, rng);
        }
        self.check_tables();
        act
    }

    /// Deletes a document: a version record (or, fixed layout, a version-map
    /// entry) marks it deleted, its postings get tombstones, its chunks are
    /// freed (no server-visible write).
    pub fn delete_doc<R: Rng>(&mut self, id: u32, rng: &mut R) -> Action {
        self.assert_live();
        let mut act = Action::default();
        if self.docs.get(&id).map_or(true, |d| d.deleted) {
            return act;
        }
        self.ensure_vmap_slot(&mut act, rng);
        self.free_doc_chunks(id);
        let kws = {
            let d = self.docs.get_mut(&id).unwrap();
            d.deleted = true;
            std::mem::take(&mut d.keywords)
        };
        if self.cfg.doc_layout == DocLayout::Fixed {
            let (u, nch) = (self.docs[&id].u, self.docs[&id].nch);
            self.set_vmap(id, (u, nch, true), &mut act);
        }
        let nch = self.docs[&id].nch as usize;
        self.append_doc_version(id);
        for _ in 0..self.cfg.doc_records(nch) {
            self.tick(&mut act, rng);
        }
        for w in kws {
            self.add_posting(w, Posting { id, nch: 0, tomb: true }, &mut act, rng);
        }
        self.check_tables();
        act
    }

    // ------------------------------------------------------------ batches

    /// One batch. Its randomness (operation order, dummy and filler
    /// operations, eviction walks, flush-record order) comes from `brng`, so
    /// the rest of the run, including the random workload of the bench, does
    /// not depend on K: a dry run with `k_measure` serves exactly the due work
    /// of the real run as long as the real K is at least its maximum.
    fn run_batch(&mut self) -> BatchOut {
        let t = Instant::now();
        self.batch += 1;
        let (k, theta, age) = match self.cfg.policy {
            Policy::Batched { k, theta, age, .. } => (k, theta, age as u64),
            _ => unreachable!(),
        };
        let phased = self.cfg.age_phased && age < u64::MAX / 8;
        // measure mode: serve all due work, no padding
        let budget = if self.cfg.k_measure { usize::MAX } else { k };
        let now = self.batch;
        let mut due: Vec<(u64, u32)> = Vec::new();
        for &w in &self.pending_kws {
            let kw = &mut self.kws[w as usize];
            let oldest = kw.pend.first().map(|x| x.1).unwrap_or(now);
            self.max_pend_age = self.max_pend_age.max(now - oldest);
            let by_age = if phased {
                !kw.pend.is_empty() && (now + kw.phase as u64 % age) % age == 0
            } else {
                now - oldest >= age
            };
            if kw.pend.len() >= theta || by_age {
                if kw.due_since.is_none() {
                    kw.due_since = Some(now);
                }
                due.push((kw.due_since.unwrap(), w));
            }
        }
        due.sort_unstable();
        let mut out = BatchOut { index: now, due: due.len(), ..Default::default() };
        // 1. plan the K operations (owner state advances; nothing is executed yet)
        let mut plan: Vec<Option<ChunkWrite>> = Vec::with_capacity(k.min(1 << 20));
        let mut served: Vec<u32> = Vec::new();
        for &(_, w) in &due {
            let left = budget - plan.len();
            if left == 0 {
                out.deferred += 1;
                continue;
            }
            let (ops, moved) = self.flush_kw(w, left, &mut plan);
            out.real_ops += ops;
            out.postings_flushed += moved;
            if ops > 0 {
                served.push(w);
                if self.kappa_window > 0 {
                    self.win_committed.insert(w);
                }
            }
            if !self.kws[w as usize].pend.is_empty() {
                out.deferred += 1;
            }
        }
        self.max_batch_ops = self.max_batch_ops.max(out.real_ops);
        if age < u64::MAX / 8 && now > 2 * age {
            self.max_batch_ops_steady = self.max_batch_ops_steady.max(out.real_ops);
        }
        // padding: Dummy (reuse) or a filler insertion (keep)
        let width = if self.cfg.k_measure { plan.len() } else { k };
        while plan.len() < width {
            plan.push(None);
            out.dummy_ops += 1;
        }
        // 2. execute them in a uniformly random order
        let mut br = std::mem::replace(&mut self.brng, StdRng::seed_from_u64(0));
        plan.shuffle(&mut br);
        for op in plan {
            let o = match op {
                Some(c) => {
                    if let Some(old) = c.free {
                        let removed = self.val.remove(&old);
                        debug_assert!(removed, "missing chunk {old:?}");
                    }
                    self.val.insert(&c.key, c.keys.0, c.keys.1, c.val, &mut br)
                }
                None if self.cfg.reuse_freed => self.val.dummy(&mut br),
                None => {
                    // filler insertion of a fresh random key, dropped at compaction
                    let (key, (lk, dk), v) = self.filler(&mut br);
                    self.val.insert(&key, lk, dk, v, &mut br)
                }
            };
            out.accessed.extend_from_slice(&o.accessed);
        }
        // 3. exactly K flush records (dummies pad), in a uniformly random order
        out.keywords_served = served.len();
        let nrec = if self.cfg.k_measure { served.len() } else { k };
        assert!(served.len() <= nrec);
        out.records = nrec;
        let mut recs: Vec<Option<u32>> = served.iter().map(|&w| Some(w)).collect();
        recs.resize(nrec, None);
        recs.shuffle(&mut br);
        for r in recs {
            match r {
                Some(w) => self.append_flush(w),
                None => self.append_dummy(&mut br),
            }
        }
        self.brng = br;
        if self.cfg.stateless && self.pend_total > self.cfg.pend_pad {
            // the padded pending stash cannot grow: a public failure
            self.pend_pad_overflow += 1;
            self.halt(format!(
                "pending stash overflow ({} pending postings > pend_pad = {}) after {} appends",
                self.pend_total, self.cfg.pend_pad, self.total_appends
            ));
        }
        // the stash bound assumes at most val_cap live chunks all epoch
        self.val_live_max = self.val_live_max.max(self.val_live);
        if self.val_live > self.cfg.val_cap {
            self.halt(format!(
                "ValMap capacity exceeded: {} live chunks > val_cap = {} after {} appends",
                self.val_live, self.cfg.val_cap, self.total_appends
            ));
        }
        if out.deferred > 0 {
            self.k_overflow_batches += 1;
            self.deferred_total += out.deferred;
        }
        assert_eq!(out.accessed.len(), width * self.cfg.val.accesses_per_op(), "a batch must make exactly k(alpha+1) accesses");
        self.check_tables();
        out.owner_ms = t.elapsed().as_secs_f64() * 1e3;
        out
    }

    /// Plans the commit of w's pending postings with at most `budget` ValMap
    /// operations (appended to `plan`): first the partial last chunk (a fresh
    /// key (w, i, len'); reuse mode frees the old one), then new chunks.
    /// Returns (operations, postings committed).
    fn flush_kw(&mut self, w: u32, budget: usize, plan: &mut Vec<Option<ChunkWrite>>) -> (usize, usize) {
        let c = self.cfg.chunk_cap();
        let mut ops = 0;
        let mut moved = 0;
        let chunks_before = (self.kws[w as usize].committed.len() + c - 1) / c;
        while ops < budget && !self.kws[w as usize].pend.is_empty() {
            let (name, vc, npend) = {
                let k = &self.kws[w as usize];
                (k.name.clone(), k.committed.len(), k.pend.len())
            };
            let i = vc / c;
            let have = vc % c;
            let take = (c - have).min(npend);
            let mut free = None;
            if have > 0 && self.cfg.reuse_freed {
                free = Some(self.chunk_key(&name, i, have));
            } else if have > 0 {
                self.superseded += 1;
            }
            let mut content: Vec<Posting> = self.kws[w as usize].committed[i * c..].to_vec();
            let drained: Vec<(Posting, u64)> = self.kws[w as usize].pend.drain(..take).collect();
            content.extend(drained.iter().map(|x| x.0));
            let key = self.chunk_key(&name, i, have + take);
            let keys = self.chunk_keys(&self.kws[w as usize], &key);
            plan.push(Some(ChunkWrite { free, key, keys, val: enc_postings(&content, self.cfg.posting_bytes) }));
            let kw = &mut self.kws[w as usize];
            kw.committed.extend(drained.iter().map(|x| x.0));
            ops += 1;
            moved += take;
        }
        self.pend_total -= moved;
        self.val_live += (self.kws[w as usize].committed.len() + c - 1) / c - chunks_before;
        let kw = &mut self.kws[w as usize];
        if kw.pend.is_empty() {
            kw.due_since = None;
            self.pending_kws.remove(&w);
        }
        (ops, moved)
    }

    // ------------------------------------------------------------ compaction

    /// Starts log epoch ε_L + 1 with a snapshot (see `write_snapshot`).
    pub fn compact<R: Rng>(&mut self, _rng: &mut R) {
        self.epoch += 1;
        self.log_compactions += 1;
        self.write_snapshot();
    }

    /// Repack (only inside a log compaction): re-places every live document chunk at
    /// uniformly random rows, all rows are rewritten (clients take a fresh
    /// DocMap hint), and `doc_live_cap - live` further rows are reserved as
    /// unusable until the next repack, so exactly `doc_rows - doc_live_cap`
    /// never-written rows remain (a public number). The caller then writes a
    /// log snapshot so version records carry the new row numbers.
    pub fn compact_docmap<R: Rng>(&mut self, rng: &mut R) {
        let n = self.doc_rows.len();
        let cap = self.cfg.doc_live_cap;
        assert!(cap <= n, "doc_live_cap {cap} > DocMap rows {n}");
        let live: usize = self.docs.values().filter(|d| !d.deleted).map(|d| d.nch as usize).sum();
        self.doc_live_max = self.doc_live_max.max(live);
        assert!(live <= cap, "DocMap capacity exceeded: {live} live chunks at a repack > doc_live_cap = {cap}; raise --doc-live-cap");
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.shuffle(rng);
        self.doc_rows = vec![None; n];
        let mut ids: Vec<u32> = self.docs.keys().copied().collect();
        ids.sort_unstable();
        for id in ids {
            let d = self.docs.get_mut(&id).unwrap();
            if d.deleted {
                continue;
            }
            let mut rows = Vec::with_capacity(d.nch as usize);
            for j in 0..d.nch {
                let r = perm.pop().expect("DocMap full");
                self.doc_rows[r as usize] = Some((id, j));
                rows.push(r);
            }
            d.rows = rows;
        }
        // reserve cap - live rows: never written, but not usable before the next repack
        perm.truncate(n - cap);
        self.virgin = perm;
        self.doc_freed = 0;
        debug_assert_eq!(self.virgin.len(), n - cap);
    }

    /// Rebuild of ValMap: every keyword's postings, with tombstones applied,
    /// re-chunked canonically under the new epoch's chunk keys, padded with
    /// fillers to exactly `val_cap` items (one uniformly random order).
    pub fn rebuild_val<R: Rng>(&mut self, rng: &mut R) {
        self.val_epoch += 1;
        let c = self.cfg.chunk_cap();
        let mut entries = Vec::new();
        for wi in 0..self.kws.len() {
            let mut all: Vec<Posting> = self.kws[wi].committed.clone();
            all.extend(self.kws[wi].pend.iter().map(|x| x.0));
            let live = apply_tombstones(&all);
            let ps: Vec<Posting> = live
                .into_iter()
                .map(|id| Posting { id, nch: self.docs.get(&id).map_or(1, |d| d.nch), tomb: false })
                .collect();
            let name = self.kws[wi].name.clone();
            for (i, ch) in ps.chunks(c).enumerate() {
                let key = self.chunk_key(&name, i, ch.len());
                let keys = self.chunk_keys(&self.kws[wi], &key);
                entries.push((key, keys, enc_postings(ch, self.cfg.posting_bytes)));
            }
            let kw = &mut self.kws[wi];
            kw.n = ps.len() as u32;
            kw.committed = ps;
            kw.pend.clear();
            kw.due_since = None;
        }
        self.pending_kws.clear();
        self.pend_total = 0;
        self.val_live_max = self.val_live_max.max(entries.len());
        self.val_live = entries.len();
        let (prm, cap) = (self.cfg.val.clone(), self.cfg.val_cap);
        self.val = self.encode_padded(prm, entries, cap, "ValMap (--val-cap)", rng);
    }

    /// Fixed layout: re-encode the DocMap under the new epoch (call after
    /// `rebuild_val`): every live document at version u = 0 with its current
    /// chunk count, padded with fillers to exactly `doc_live_cap` items; the
    /// version map is cleared.
    fn rebuild_doc<R: Rng>(&mut self, rng: &mut R) {
        let ep = self.val_epoch;
        let mut ids: Vec<u32> = self.docs.keys().copied().collect();
        ids.sort_unstable();
        let mut entries = Vec::new();
        for id in ids {
            let d = self.docs.get_mut(&id).unwrap();
            if d.deleted {
                continue;
            }
            d.u = 0;
            d.dk_ep = ep;
            for j in 0..d.nch {
                let k = Self::doc_key_fixed(id, ep, 0, j);
                let keys = derive(&d.cap, k.idx);
                entries.push((k, keys, vec![(d.ver % 251) as u8; 1]));
            }
        }
        self.doc_live_max = self.doc_live_max.max(entries.len());
        let (prm, cap) = (self.cfg.doc.clone(), self.cfg.doc_live_cap);
        self.doc_table = Some(self.encode_padded(prm, entries, cap, "DocMap (--doc-live-cap)", rng));
        self.vmap.clear();
        self.vmap_ops = 0;
    }

    // ------------------------------------------------------------ client

    /// Search for w by a client holding w's delegation keys and a synced copy
    /// of the log. Uses only tags it can derive, the log, and the tables'
    /// public lookup procedure.
    pub fn search(&self, w: &str) -> SearchOut {
        let mut out = SearchOut::default();
        let Some(&wi) = self.kw_ix.get(w) else { return out };
        let kw = &self.kws[wi as usize];
        // 1. latest flush record -> committed volume
        let mut vc = 0u32;
        let mut m = 0u64;
        loop {
            out.tag_probes += 1;
            match self.tag_ix.get(&tag_of(&kw.cap_log, b"flush", self.epoch, m, 0)) {
                Some(&i) => {
                    if let Body::Flush { vc: v } = self.log[i].body {
                        vc = v;
                    }
                    m += 1;
                }
                None => break,
            }
        }
        // 2. pending postings n >= vc
        let mut pending: Vec<Posting> = Vec::new();
        let mut n = vc as u64;
        loop {
            out.tag_probes += 1;
            match self.tag_ix.get(&tag_of(&kw.cap_log, b"post", self.epoch, n, 0)) {
                Some(&i) => {
                    if let Body::Post(p) = self.log[i].body {
                        pending.push(p);
                    }
                    n += 1;
                }
                None => break,
            }
        }
        let c = self.cfg.chunk_cap();
        let vc = vc as usize;
        let v = n as usize;
        out.volume = v;
        out.committed = vc;
        out.pending = v - vc;
        // 3. chunks: ceil(v / C) queries, the last ones dummies if not yet committed
        let mut all: Vec<Posting> = Vec::with_capacity(v);
        let committed_chunks = (vc + c - 1) / c;
        let total_chunks = (v + c - 1) / c;
        for i in 0..committed_chunks {
            let len = if i + 1 < committed_chunks || vc % c == 0 { c } else { vc % c };
            let key = self.chunk_key(w, i, len);
            let (lk, _) = self.chunk_keys(kw, &key);
            out.val_queries.extend(self.val.candidates(&lk));
            let (val, _) = self.val.lookup(&key, &lk).expect("committed chunk not found");
            all.extend(dec_postings(&val, self.cfg.posting_bytes));
        }
        for _ in committed_chunks..total_chunks {
            // padding query: s uniformly random positions
            out.val_queries.extend((0..self.val.prm.s).map(|j| (j * 7919 + v) % self.val.prm.mu.max(1)));
        }
        out.chunks_queried = total_chunks;
        all.extend(pending);
        out.ids = apply_tombstones(&all);
        // chunk count carried by each live posting (the latest one wins)
        let mut nch_of: HashMap<u32, u16> = HashMap::new();
        for p in &all {
            if !p.tomb {
                nch_of.insert(p.id, p.nch);
            }
        }
        // 4. documents: latest version record from the log
        for &id in &out.ids {
            let d = &self.docs[&id];
            let before = out.doc_queries.len();
            if self.cfg.doc_layout == DocLayout::Fixed {
                // the version map (downloaded with the search) names the
                // version of a document edited or removed in this epoch;
                // otherwise it is version 0 with the posting's chunk count
                let (u, n) = match self.vmap.get(&id) {
                    Some(&(u, n, del)) => {
                        assert!(!del, "search returned a removed document");
                        (u, n)
                    }
                    None => (0, nch_of[&id]),
                };
                assert_eq!(n, d.nch, "stale chunk count for document {id}");
                let t = self.doc_table.as_ref().unwrap();
                for j in 0..n {
                    let k = Self::doc_key_fixed(id, self.val_epoch, u, j);
                    let (lk, _) = derive(&d.cap, k.idx);
                    let (_, tomb) = t.lookup(&k, &lk).expect("document chunk missing");
                    assert!(!tomb, "search returned a removed document");
                    out.doc_queries.extend(t.candidates(&lk));
                }
                out.doc_queries_per_doc.push(out.doc_queries.len() - before);
                continue;
            }
            let mut latest: Option<(u32, u16, bool, Vec<u32>)> = None;
            let mut m = 0u64;
            loop {
                out.tag_probes += 1;
                let Some(&i0) = self.tag_ix.get(&tag_of(&d.cap.0, b"ver", self.epoch, m, 0)) else { break };
                let mut rows = Vec::new();
                let parts = match &self.log[i0].body {
                    Body::DocVer { nch, .. } => self.cfg.doc_records(*nch as usize),
                    _ => 1,
                };
                let mut head = None;
                for part in 0..parts {
                    if let Some(&ip) = self.tag_ix.get(&tag_of(&d.cap.0, b"ver", self.epoch, m, part as u32)) {
                        if let Body::DocVer { ver, nch, deleted, rows: r } = &self.log[ip].body {
                            rows.extend_from_slice(r);
                            head = Some((*ver, *nch, *deleted));
                        }
                    }
                }
                if let Some((ver, nch, del)) = head {
                    latest = Some((ver, nch, del, rows));
                }
                m += 1;
            }
            let (ver, nch, deleted, rows) = latest.expect("no version record for a live document");
            assert!(!deleted, "search returned a deleted document");
            match self.cfg.doc_layout {
                DocLayout::Rows => {
                    assert_eq!(rows.len(), nch as usize);
                    for (j, &r) in rows.iter().enumerate() {
                        assert_eq!(self.doc_rows[r as usize], Some((id, j as u16)), "row pointer is stale");
                        out.doc_queries.push(r as usize);
                    }
                }
                DocLayout::Fixed => unreachable!("handled above"),
                DocLayout::Cuckoo => {
                    let t = self.doc_table.as_ref().unwrap();
                    for j in 0..nch {
                        let k = Self::doc_key(id, ver, j);
                        let (lk, _) = derive(&d.cap, k.idx);
                        assert!(t.lookup(&k, &lk).is_some(), "document chunk missing");
                        out.doc_queries.extend(t.candidates(&lk));
                    }
                }
            }
            out.doc_queries_per_doc.push(out.doc_queries.len() - before);
        }
        out
    }

    // ------------------------------------------------------------ sizes

    pub fn log_bytes(&self) -> usize {
        self.log.len() * self.cfg.record_bytes()
    }
    pub fn compacted_bytes(&self) -> usize {
        self.compacted_records * self.cfg.record_bytes()
    }
    pub fn doc_mu(&self) -> usize {
        match self.cfg.doc_layout {
            DocLayout::Rows => self.doc_rows.len(),
            DocLayout::Cuckoo | DocLayout::Fixed => self.doc_table.as_ref().unwrap().prm.mu,
        }
    }
    pub fn doc_load(&self) -> f64 {
        match self.cfg.doc_layout {
            DocLayout::Rows => self.doc_rows.iter().filter(|x| x.is_some()).count() as f64 / self.doc_rows.len().max(1) as f64,
            DocLayout::Cuckoo | DocLayout::Fixed => self.doc_table.as_ref().unwrap().load(),
        }
    }
    pub fn virgin_rows(&self) -> usize {
        self.virgin.len()
    }
    pub fn pending_keywords(&self) -> usize {
        self.pending_kws.len()
    }
    pub fn keywords(&self) -> usize {
        self.kws.len()
    }
    pub fn val_hint_bytes(&self) -> usize {
        frodo_hint_bytes(self.cfg.val_row_bytes())
    }
    pub fn doc_hint_bytes(&self) -> usize {
        frodo_hint_bytes(self.cfg.doc_row_cipher_bytes())
    }
    pub fn val_delta_bytes(&self, packed: bool) -> usize {
        frodo_row_delta_bytes(self.cfg.val_row_bytes(), packed)
    }
    pub fn doc_delta_bytes(&self, packed: bool) -> usize {
        frodo_row_delta_bytes(self.cfg.doc_row_cipher_bytes(), packed)
    }
    /// Bytes of the ValMap stash a client downloads (padded to sigma).
    pub fn val_stash_bytes(&self) -> usize {
        self.cfg.val.stash_bytes()
    }
    /// Encrypted DocMap row (rows layout) for PIR.
    pub fn doc_row_cipher<R: Rng>(&self, r: usize, rng: &mut R) -> Vec<u8> {
        let mut v = vec![0u8; self.cfg.doc_row_cipher_bytes()];
        rng.fill(&mut v[..]);
        if let Some((id, j)) = self.doc_rows[r] {
            v[NONCE..NONCE + 4].copy_from_slice(&id.to_le_bytes());
            v[NONCE + 4..NONCE + 6].copy_from_slice(&j.to_le_bytes());
        }
        v
    }
}

/// Live ids of a posting sequence: the latest record for an id wins.
pub fn apply_tombstones(ps: &[Posting]) -> Vec<u32> {
    let mut order: Vec<u32> = Vec::new();
    let mut live: HashSet<u32> = HashSet::new();
    for p in ps {
        if p.tomb {
            live.remove(&p.id);
        } else if live.insert(p.id) {
            order.push(p.id);
        }
    }
    let mut seen = HashSet::new();
    order.retain(|id| live.contains(id) && seen.insert(*id));
    order
}

// ------------------------------------------------------------ client profiles

/// The PIR used for one map in the paired evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PirKind {
    /// FrodoPIR (our fork): client hint, refreshed by row deltas or anew
    Frodo,
    /// Spiral (our fast server): no hint; public parameters once
    Spiral,
    /// YPIR+SP (our fast server, ypir_fast): no hint; packing keys per client key,
    /// several records per row (`ypir_shape`)
    Ypir,
}

pub const N_COMBOS: usize = 6;
pub const PIR_COMBOS: [(PirKind, PirKind); N_COMBOS] = [
    (PirKind::Frodo, PirKind::Frodo),
    (PirKind::Spiral, PirKind::Frodo),
    (PirKind::Frodo, PirKind::Spiral),
    (PirKind::Spiral, PirKind::Spiral),
    (PirKind::Ypir, PirKind::Spiral),
    (PirKind::Ypir, PirKind::Ypir),
];

pub fn pir_name(k: PirKind) -> &'static str {
    match k {
        PirKind::Frodo => "frodo",
        PirKind::Spiral => "spiral",
        PirKind::Ypir => "ypir",
    }
}

/// YPIR+SP packing keys (11 x 3 polynomials x 2048 x 7 bytes), one set per client key.
pub const YPIR_PP_BYTES: usize = 473_088;
/// Queries that share one client key (the server keeps this many public seeds;
/// ypir_fast `--seeds`). 1 = a fresh key, and packing keys, with every query.
pub const YPIR_SEEDS: usize = 1;

/// ypir_fast's `Shape::auto`: records per row minimizing query (7 B per row, rows a
/// power of two >= 2048) plus response (12,288 B per 3,584 B of row). Returns
/// (rows, instances).
pub fn ypir_shape(records: usize, record_bytes: usize) -> (usize, usize) {
    (1..=256usize)
        .map(|k| {
            let rows = ((records + k - 1) / k).next_power_of_two().max(2048);
            let inst = (k * record_bytes * 8 + 2048 * 14 - 1) / (2048 * 14);
            (rows, inst)
        })
        .min_by_key(|&(r, i)| r * 7 + i * 12_288)
        .unwrap()
}

/// Packing-key bytes for `queries` YPIR queries of one search.
pub fn ypir_pp_bytes(queries: usize) -> usize {
    (queries + YPIR_SEEDS - 1) / YPIR_SEEDS * YPIR_PP_BYTES
}

/// Spiral query size (bytes): one seeded ciphertext, independent of the table
/// (measured with the Blyss client at nu1 = 9).
pub const SPIRAL_QUERY_BYTES: usize = 16_416;
/// Spiral public parameters (expansion and conversion keys), downloaded once.
pub const SPIRAL_PP_BYTES: usize = 1_700_000;

/// Spiral response size for records of `row_bytes` (n = 1 holds 2 KB, n = 2
/// holds 8 KB): instances * (q2_bits * n + q1_bits * n^2) * 2048 bits with
/// q2_bits = 22 and q1_bits = 10.
pub fn spiral_response_bytes(row_bytes: usize) -> usize {
    let n = if row_bytes <= 2048 { 1 } else { 2 };
    let inst = (row_bytes + n * n * 2048 - 1) / (n * n * 2048);
    inst * (22 * n + 10 * n * n) * 2048 / 8
}

/// Per-query sizes of one map under each PIR.
#[derive(Clone, Copy, Debug)]
pub struct MapPir {
    pub frodo_query: usize,
    pub frodo_response: usize,
    pub spiral_query: usize,
    pub spiral_response: usize,
    pub ypir_query: usize,
    pub ypir_response: usize,
}

impl MapPir {
    pub fn new(rows: usize, row_bytes: usize) -> Self {
        Self {
            frodo_query: rows * 4,
            frodo_response: crate::pir_backend::frodo_width(row_bytes) * 4,
            spiral_query: SPIRAL_QUERY_BYTES,
            spiral_response: spiral_response_bytes(row_bytes),
            ypir_query: ypir_shape(rows, row_bytes).0 * 7,
            ypir_response: ypir_shape(rows, row_bytes).1 * 12_288,
        }
    }
    pub fn per_query(&self, k: PirKind) -> usize {
        match k {
            PirKind::Frodo => self.frodo_query + self.frodo_response,
            PirKind::Spiral => self.spiral_query + self.spiral_response,
            // packing keys are added per search (ypir_pp_bytes)
            PirKind::Ypir => self.ypir_query + self.ypir_response,
        }
    }
}

impl MkseLog {
    pub fn val_pir(&self) -> MapPir {
        MapPir::new(self.val.prm.mu, self.cfg.val_row_bytes())
    }
    pub fn doc_pir(&self) -> MapPir {
        MapPir::new(self.doc_mu(), self.cfg.doc_row_cipher_bytes())
    }
    /// Bytes a new client downloads before its first search, per PIR choice.
    pub fn join_bytes(&self, val: PirKind, doc: PirKind) -> usize {
        let state = if self.cfg.stateless { 0 } else { self.compacted_bytes() };
        let h = |k: PirKind, hint: usize| match k {
            PirKind::Frodo => hint,
            PirKind::Spiral => SPIRAL_PP_BYTES,
            PirKind::Ypir => 0,
        };
        state + h(val, self.val_hint_bytes()) + h(doc, self.doc_hint_bytes())
    }
}

/// A client that searches every `interval` appends and refreshes its state
/// only then. Costs are kept in two parts so that every pairing of PIRs
/// (ValMap, DocMap) can be costed from one run:
/// * the **keyword lookup**: sync (new log records or the compacted log, or for
///   stateless clients VolMap plus the pending stash), the ValMap stash, the
///   ValMap hint refresh (FrodoPIR only), and the ValMap queries and answers;
/// * **document retrieval**: the DocMap stash, the DocMap hint refresh
///   (FrodoPIR only), and the DocMap queries and answers, both for fetching
///   every matching document and for fetching at most `doc_cap` of them.
#[derive(Clone, Debug)]
pub struct Profile {
    pub interval: usize,
    pub doc_cap: usize,
    since: usize,
    val_rows: HashSet<u32>,
    val_full: bool,
    doc_rows: HashSet<u32>,
    doc_full: bool,
    log_bytes: usize,
    dirty: bool,
    doc_dirty: bool,
    pub searches: usize,
    pub hint_val: usize,
    pub hint_doc: usize,
    pub sync: usize,
    pub stash_val: usize,
    pub stash_doc: usize,
    pub val_queries: usize,
    pub doc_queries: usize,
    pub doc_queries_capped: usize,
    pub docs: usize,
    pub docs_capped: usize,
    pub full_hint_refreshes: usize,
    /// per pairing in PIR_COMBOS order (lookup depends on the ValMap PIR,
    /// documents on the DocMap PIR, but both are kept per pairing for clarity)
    pub lookup: [usize; N_COMBOS],
    pub lookup_max: [usize; N_COMBOS],
    pub lookup_hist: [Vec<usize>; N_COMBOS],
    pub docs_all: [usize; N_COMBOS],
    pub docs_capped_bytes: [usize; N_COMBOS],
    pub capped_hist: [Vec<usize>; N_COMBOS],
}

impl Profile {
    pub fn new(interval: usize, doc_cap: usize) -> Self {
        Self {
            interval,
            doc_cap,
            since: 0,
            val_rows: HashSet::new(),
            val_full: false,
            doc_rows: HashSet::new(),
            doc_full: false,
            log_bytes: 0,
            dirty: false,
            doc_dirty: false,
            searches: 0,
            hint_val: 0,
            hint_doc: 0,
            sync: 0,
            stash_val: 0,
            stash_doc: 0,
            val_queries: 0,
            doc_queries: 0,
            doc_queries_capped: 0,
            docs: 0,
            docs_capped: 0,
            full_hint_refreshes: 0,
            lookup: [0; N_COMBOS],
            lookup_max: [0; N_COMBOS],
            lookup_hist: Default::default(),
            docs_all: [0; N_COMBOS],
            docs_capped_bytes: [0; N_COMBOS],
            capped_hist: Default::default(),
        }
    }

    /// Feeds one owner action; returns true if the client searches now.
    pub fn observe(&mut self, m: &MkseLog, act: &Action) -> bool {
        let vcap = m.val_hint_bytes() / m.val_delta_bytes(true) + 1;
        let dcap = m.doc_hint_bytes() / m.doc_delta_bytes(true) + 1;
        for b in &act.batches {
            self.dirty = true;
            if !self.val_full {
                self.val_rows.extend(b.accessed.iter().map(|&x| x as u32));
                if self.val_rows.len() >= vcap {
                    self.val_full = true;
                    self.val_rows.clear();
                }
            }
        }
        if !act.rebuilds.is_empty() {
            self.val_full = true;
            self.val_rows.clear();
            self.dirty = true;
        }
        if act.doc_compactions > 0 {
            self.doc_full = true;
            self.doc_rows.clear();
        }
        if !act.doc_ops.is_empty() {
            self.doc_dirty = true;
        }
        if !self.doc_full {
            self.doc_rows.extend(act.doc_rows_written.iter().map(|&x| x as u32));
            for o in &act.doc_ops {
                self.doc_rows.extend(o.accessed.iter().map(|&x| x as u32));
            }
            if self.doc_rows.len() >= dcap {
                self.doc_full = true;
                self.doc_rows.clear();
            }
        }
        if !m.cfg.stateless {
            // log: a compaction replaces everything the client had
            let appended = act.appends + act.batches.iter().map(|b| b.records).sum::<usize>();
            if act.compactions > 0 {
                self.log_bytes = m.compacted_bytes() + m.log_live_since_compaction * m.cfg.record_bytes();
            } else {
                self.log_bytes += appended * m.cfg.record_bytes();
            }
        } else if act.appends > 0 {
            self.dirty = true;
        }
        self.since += act.appends;
        if self.since >= self.interval {
            self.since = 0;
            return true;
        }
        false
    }

    /// The client refreshes and runs the search `sr`.
    pub fn sync(&mut self, m: &MkseLog, sr: &SearchOut) {
        let qv = sr.val_queries.len();
        let qd = sr.doc_queries.len();
        let qdc = sr.doc_queries_capped(self.doc_cap);
        let nd = sr.ids.len();
        let ndc = nd.min(self.doc_cap);
        let hv = if self.val_full { self.full_hint_refreshes += 1; m.val_hint_bytes() } else { self.val_rows.len() * m.val_delta_bytes(true) };
        let hd = if self.doc_full { m.doc_hint_bytes() } else { self.doc_rows.len() * m.doc_delta_bytes(true) };
        let sv = if self.dirty { m.val_stash_bytes() } else { 0 };
        let sd = if self.doc_dirty && m.cfg.doc_layout != DocLayout::Rows { m.cfg.doc.stash_bytes() } else { 0 };
        let sync = if m.cfg.stateless {
            if self.dirty { m.cfg.volmap_bytes() + m.cfg.pending_stash_bytes() } else { 0 }
        } else {
            self.log_bytes
        } + m.cfg.vmap_bytes(); // fixed DocMap layout: the padded version map, with every search
        let (vp, dp) = (m.val_pir(), m.doc_pir());
        for (c, &(kv, kd)) in PIR_COMBOS.iter().enumerate() {
            let pp = |k: PirKind, q: usize| if k == PirKind::Ypir { ypir_pp_bytes(q) } else { 0 };
            let look = sync + sv + qv * vp.per_query(kv) + pp(kv, qv) + if kv == PirKind::Frodo { hv } else { 0 };
            let dh = sd + if kd == PirKind::Frodo { hd } else { 0 };
            let all = dh + qd * dp.per_query(kd) + pp(kd, qd);
            let capped = dh + qdc * dp.per_query(kd) + pp(kd, qdc);
            self.lookup[c] += look;
            self.lookup_max[c] = self.lookup_max[c].max(look);
            self.lookup_hist[c].push(look);
            self.docs_all[c] += all;
            self.docs_capped_bytes[c] += capped;
            self.capped_hist[c].push(look + capped);
        }
        self.hint_val += hv;
        self.hint_doc += hd;
        self.stash_val += sv;
        self.stash_doc += sd;
        self.sync += sync;
        self.val_queries += qv;
        self.doc_queries += qd;
        self.doc_queries_capped += qdc;
        self.docs += nd;
        self.docs_capped += ndc;
        self.searches += 1;
        self.val_rows.clear();
        self.doc_rows.clear();
        self.val_full = false;
        self.doc_full = false;
        self.log_bytes = 0;
        self.dirty = false;
        self.doc_dirty = false;
    }
}

/// Median of a list of byte counts.
pub fn median(v: &[usize]) -> usize {
    let mut h = v.to_vec();
    h.sort_unstable();
    h.get(h.len() / 2).copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uokvs::{Discipline, Evict};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    const DOC_ROW: usize = 512;

    fn prm(mu: usize, val_len: usize, sigma: usize) -> Params {
        Params { mu, p: 1, s: 4, alpha: 20, sigma, evict: Evict::Bfs, disc: Discipline::Paper, bfs_max_branch: 2, bfs_max_depth: 64, val_len }
    }

    /// 40 keywords over 300 documents (setup: the first 200).
    fn data(rng: &mut StdRng) -> (Vec<(String, Vec<u32>)>, Vec<(String, u32)>, Vec<String>) {
        let kws: Vec<String> = (0..40).map(|i| format!("w{i}")).collect();
        let mut mm: HashMap<String, Vec<u32>> = HashMap::new();
        let mut later = Vec::new();
        for d in 1..=300u32 {
            let mut ws: Vec<usize> = (0..40).collect();
            ws.shuffle(rng);
            for &w in ws.iter().take(rng.gen_range(2..7)) {
                if d <= 200 {
                    mm.entry(kws[w].clone()).or_default().push(d);
                } else {
                    later.push((kws[w].clone(), d));
                }
            }
        }
        let mut mm: Vec<(String, Vec<u32>)> = mm.into_iter().collect();
        mm.sort();
        (mm, later, kws)
    }

    fn cfg(layout: DocLayout, reuse: bool, stateless: bool) -> LogCfg {
        let val_len = 1024 - NONCE - META - CHECK;
        let c = val_len / 40;
        // chunks of all 300 documents (about 4.5 keywords each), with slack
        let chunks = 40 * ((300 * 9 / 2 / 40) * 3 / 2 / c + 1);
        let mu = 3 * chunks;
        let mu = if reuse { mu } else { mu * 13 / 10 };
        let policy = if stateless {
            Policy::Batched { b: 10, k: 12, theta: 1, age: usize::MAX / 4 }
        } else {
            Policy::Batched { b: 20, k: 6, theta: 10, age: 5 }
        };
        LogCfg {
            val: prm(mu, val_len, 16),
            posting_bytes: 40,
            policy,
            major_every: 400,
            log_every: 400,
            pend_cap: 0,
            pend_measure: false,
            k_measure: false,
            age_phased: false,
            doc_layout: layout,
            doc: prm(4000, DOC_ROW, 32),
            doc_row_bytes: DOC_ROW,
            doc_rows: 2600,
            record_body: 40,
            kw_pad: 40,
            reuse_freed: reuse,
            stateless,
            pend_pad: 20,
            val_cap: (0.46 * (3 * chunks) as f64).ceil() as usize,
            doc_live_cap: 1800,
            doc_repack_below: 0,
            ver_cap: if layout == DocLayout::Fixed { 0 } else { 400 },
            vmap_pad: 40,
        }
    }

    struct Run {
        m: MkseLog,
        refmm: HashMap<String, Vec<u32>>,
        live: Vec<u32>,
    }

    fn setup(c: LogCfg, seed: u64) -> (Run, Vec<(String, u32)>, StdRng) {
        let mut rng = StdRng::seed_from_u64(seed);
        let (mm, later, kws) = data(&mut rng);
        let lens: HashMap<u32, usize> = (1..=300).map(|d| (d, 300 + (d as usize * 37) % 2000)).collect();
        let m = MkseLog::setup(c, &mm, &kws, &lens, 1500, &mut rng);
        let refmm: HashMap<String, Vec<u32>> = mm.iter().cloned().collect();
        (Run { m, refmm, live: (1..=200).collect() }, later, rng)
    }

    impl Run {
        /// one insertion, then with some probability a posting deletion, an
        /// edit, or a removal
        fn step(&mut self, w: &str, d: u32, rng: &mut StdRng) {
            self.m.insert(w, d, 300 + (d as usize * 37) % 2000, rng);
            self.refmm.entry(w.to_string()).or_default().push(d);
            if !self.live.contains(&d) {
                self.live.push(d);
            }
            let r: f64 = rng.gen();
            if r < 0.1 {
                let w2 = format!("w{}", rng.gen_range(0..40));
                if let Some(list) = self.refmm.get_mut(&w2) {
                    if !list.is_empty() {
                        let id = list.remove(rng.gen_range(0..list.len()));
                        self.m.delete_posting(&w2, id, rng);
                    }
                }
            } else if r < 0.25 {
                let id = *self.live.choose(rng).unwrap();
                self.m.update_doc(id, rng.gen_range(100..3500), rng);
            } else if r < 0.3 && self.live.len() > 1 {
                let id = self.live.swap_remove(rng.gen_range(0..self.live.len()));
                self.m.delete_doc(id, rng);
                for list in self.refmm.values_mut() {
                    list.retain(|x| *x != id);
                }
            }
        }
        fn check(&self) {
            for i in 0..40 {
                let w = format!("w{i}");
                let mut got = self.m.search(&w).ids;
                let mut want = self.refmm.get(&w).cloned().unwrap_or_default();
                got.sort_unstable();
                want.sort_unstable();
                want.dedup();
                assert_eq!(got, want, "search mismatch for {w}");
            }
        }
    }

    #[test]
    fn snapshot_has_public_size() {
        for layout in [DocLayout::Rows, DocLayout::Cuckoo] {
            let c = cfg(layout, true, false);
            let want = c.kw_pad + c.ver_cap;
            let (mut r, later, mut rng) = setup(c, 3);
            assert_eq!(r.m.log.len(), want);
            assert_eq!(r.m.compacted_records, want);
            for (i, (w, d)) in later.iter().enumerate() {
                r.step(w, *d, &mut rng);
                if i % 97 == 0 {
                    let mut act = Action::default();
                    r.m.major(&mut act, &mut rng);
                    assert_eq!(r.m.log.len(), want, "{layout:?}: snapshot size depends on the data");
                    r.check();
                }
            }
            // removed documents left the snapshot: the needed records changed, the size did not
            assert!(r.m.docs.values().any(|d| d.deleted));
            r.check();
        }
    }

    #[test]
    fn rebuild_encodes_exactly_val_cap() {
        for reuse in [true, false] {
            let c = cfg(DocLayout::Rows, reuse, false);
            let cap = c.val_cap;
            let (mut r, later, mut rng) = setup(c, 5);
            assert_eq!(r.m.last_encode, cap);
            let mut rebuilds = 0;
            for (w, d) in later.iter() {
                let before = r.m.val_epoch;
                r.step(w, *d, &mut rng);
                if r.m.val_epoch != before {
                    rebuilds += 1;
                    assert_eq!(r.m.last_encode, cap);
                }
            }
            let mut act = Action::default();
            r.m.major(&mut act, &mut rng);
            assert_eq!(r.m.last_encode, cap);
            let live: usize = r.m.kws.iter().map(|k| (k.committed.len() + r.m.cfg.chunk_cap() - 1) / r.m.cfg.chunk_cap()).sum();
            if reuse {
                // fillers freed right after the Encode
                assert_eq!(r.m.val.len(), live);
            } else {
                // fillers kept: occupancy is public all epoch
                assert_eq!(r.m.val.len(), cap);
            }
            assert!(rebuilds > 0);
            r.m.val.check_invariants();
            r.check();
        }
    }

    #[test]
    #[should_panic(expected = "capacity exceeded")]
    fn rebuild_over_val_cap_panics() {
        let mut c = cfg(DocLayout::Rows, true, false);
        c.val_cap = 3;
        let _ = setup(c, 1);
    }

    #[test]
    fn repack_leaves_public_never_written_rows() {
        let mut c = cfg(DocLayout::Rows, true, false);
        // few spare rows: writes run out of never-written rows and force repacks
        c.doc_rows = 1040;
        c.doc_live_cap = 1000;
        c.major_every = 0;
        c.log_every = 0;
        let (mu_d, cap) = (c.doc_rows, c.doc_live_cap);
        let (mut r, later, mut rng) = setup(c, 9);
        let mut repacks = 0;
        for (w, d) in later.iter() {
            let before = r.m.epoch;
            let vbefore = r.m.val_epoch;
            r.step(w, *d, &mut rng);
            if r.m.epoch != before {
                // pend_cap = 0: the forced compaction is a full rebuild
                assert_ne!(r.m.val_epoch, vbefore);
                repacks += 1;
                // the write that triggered it then took its rows from the never-written ones
                assert!(r.m.virgin_rows() <= mu_d - cap);
            }
        }
        assert!(repacks > 0, "test should force a repack");
        r.check();
        // a repack inside a scheduled rebuild, with a threshold
        r.m.cfg.doc_repack_below = mu_d;
        let mut act = Action::default();
        r.m.major(&mut act, &mut rng);
        assert_eq!(act.doc_compactions, 1);
        assert_eq!(r.m.virgin_rows(), mu_d - cap);
        r.check();
        // without the threshold a scheduled rebuild does not repack
        r.m.cfg.doc_repack_below = 0;
        let v = r.m.virgin_rows();
        let mut act = Action::default();
        r.m.major(&mut act, &mut rng);
        assert_eq!((act.doc_compactions, r.m.virgin_rows()), (0, v));
    }

    #[test]
    fn stateless_docmap_versions() {
        for reuse in [true, false] {
            let mut c = cfg(DocLayout::Fixed, reuse, true);
            if !reuse {
                c.doc = prm(6000, DOC_ROW, 32);
            }
            c.vmap_pad = 25; // full every ~25 edits/removals: forces rebuilds
            let cap = c.doc_live_cap;
            let (mut r, later, mut rng) = setup(c, 13);
            let mut vmap_rebuilds = 0;
            for (i, (w, d)) in later.iter().enumerate() {
                let before = r.m.val_epoch;
                let ops = r.m.vmap_ops;
                r.step(w, *d, &mut rng);
                if r.m.val_epoch != before {
                    // the DocMap Encode (after the ValMap one) has exactly doc_live_cap items
                    assert_eq!(r.m.last_encode, cap);
                    if ops >= 24 {
                        vmap_rebuilds += 1;
                    }
                }
                assert!(r.m.vmap.len() <= r.m.vmap_ops && r.m.vmap_ops <= 25);
                if i % 50 == 0 {
                    r.check();
                }
            }
            assert!(!r.m.halted, "{}", r.m.halt_reason);
            assert!(vmap_rebuilds > 0);
            r.check();
            let mut act = Action::default();
            r.m.major(&mut act, &mut rng);
            assert_eq!(act.doc_compactions, 1);
            let live: usize = r.m.docs.values().filter(|d| !d.deleted).map(|d| d.nch as usize).sum();
            assert_eq!(r.m.doc_table.as_ref().unwrap().len(), if reuse { live } else { cap });
            assert!(r.m.vmap.is_empty() && r.m.vmap_ops == 0);
            r.check();
            r.m.doc_table.as_ref().unwrap().check_invariants();
        }
    }

    #[test]
    fn stash_overflow_halts() {
        let mut c = cfg(DocLayout::Rows, true, false);
        // a full table with no stash: the Encode overflows
        c.val = prm(c.val_cap, c.val.val_len, 0);
        c.val.alpha = 1;
        let (mut r, later, mut rng) = setup(c, 2);
        assert!(r.m.halted && r.m.val.halted && r.m.val.overflow > 0, "no overflow");
        assert!(r.m.halt_reason.contains("stash overflow"));
        let (w, d) = later[0].clone();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| r.step(&w, d, &mut rng)));
        assert!(res.is_err(), "a halted structure must refuse operations");
    }

    /// batched with a high theta (postings stay pending), log compaction every
    /// `le` appends, ValMap rebuild every `ve` (0 = never), P̂ = `cap`
    fn decoupled(layout: DocLayout, le: usize, ve: usize, cap: usize) -> LogCfg {
        let mut c = cfg(layout, true, false);
        c.policy = Policy::Batched { b: 20, k: 3, theta: 15, age: usize::MAX / 4 };
        c.log_every = le;
        c.major_every = ve;
        c.pend_cap = cap;
        c
    }

    #[test]
    fn decoupled_snapshot_carries_pending() {
        for layout in [DocLayout::Rows, DocLayout::Cuckoo] {
            let c = decoupled(layout, 60, 0, 3000);
            let want = c.kw_pad + c.pend_cap + c.ver_cap;
            assert_eq!(want, c.snapshot_records());
            let (mut r, later, mut rng) = setup(c, 21);
            assert_eq!(r.m.log.len(), want);
            let (mut seen_pending, mut manual) = (0, 0);
            for (i, (w, d)) in later.iter().enumerate() {
                r.step(w, *d, &mut rng);
                assert_eq!(r.m.compacted_records, want, "{layout:?}: snapshot size depends on the data");
                if i % 40 == 39 {
                    let (ve, le, pend) = (r.m.val_epoch, r.m.epoch, r.m.pend_total);
                    let mut act = Action::default();
                    r.m.log_compaction(&mut act, &mut rng);
                    assert_eq!((r.m.val_epoch, r.m.epoch, act.compactions, act.rebuilds.len()), (ve, le + 1, 1, 0));
                    // the snapshot alone (nothing appended yet) has exactly κ̂ + P̂ + V̂ records
                    assert_eq!(r.m.log.len(), want);
                    assert_eq!(r.m.pend_total, pend, "a log compaction commits nothing");
                    if pend > 0 {
                        manual += 1;
                    }
                    // pending postings survive under the new epoch's tags with their numbers
                    for k in 0..40 {
                        seen_pending += r.m.search(&format!("w{k}")).pending;
                    }
                    r.check();
                }
            }
            assert!(manual > 0 && seen_pending > 0, "test should compact with pending postings");
            assert_eq!(r.m.val_rebuilds, 0);
            assert!(r.m.log_compactions > 10);
            assert!(r.m.pend_snap_max > 0 && r.m.pend_snap_max <= 3000);
            r.check();
        }
    }

    #[test]
    fn val_rebuild_coincides_with_log_compaction() {
        let (le, ve) = (50, 200);
        let (mut r, later, mut rng) = setup(decoupled(DocLayout::Rows, le, ve, 3000), 23);
        let (mut rebuilds, mut logs) = (0, 0);
        for (i, (w, d)) in later.iter().enumerate() {
            let (v0, l0, t0) = (r.m.val_epoch, r.m.epoch, r.m.total_appends);
            r.step(w, *d, &mut rng);
            let t1 = r.m.total_appends;
            // absolute schedules (no forced compaction in this test)
            let due_v = (t0 + 1..=t1).filter(|t| t % ve == 0).count();
            let due_l = (t0 + 1..=t1).filter(|t| t % le == 0).count();
            assert_eq!((r.m.val_epoch - v0) as usize, due_v);
            assert_eq!((r.m.epoch - l0) as usize, due_l, "every ValMap rebuild is also a scheduled log compaction");
            rebuilds += due_v;
            logs += due_l;
            if i % 30 == 0 {
                r.check();
            }
        }
        assert!(rebuilds >= 2 && logs >= 4 * rebuilds);
        assert_eq!((r.m.val_rebuilds, r.m.log_compactions), (rebuilds, logs));
        assert!(r.m.pend_snap_max > 0);
        // a rebuild commits everything: its snapshot has only dummy posting records
        let mut act = Action::default();
        r.m.major(&mut act, &mut rng);
        assert_eq!(r.m.pend_total, 0);
        assert_eq!(r.m.log.len(), r.m.cfg.snapshot_records());
        assert!(r.m.log.iter().all(|x| !matches!(x.body, Body::Post(_))));
        r.check();
    }

    #[test]
    #[should_panic(expected = "multiple")]
    fn val_period_must_be_a_multiple_of_log_period() {
        let _ = setup(decoupled(DocLayout::Rows, 60, 200, 3000), 1);
    }

    #[test]
    #[should_panic(expected = "pend_cap must be > 0")]
    fn decoupled_needs_pend_cap() {
        let _ = setup(decoupled(DocLayout::Rows, 50, 200, 0), 1);
    }

    #[test]
    #[should_panic(expected = "capacity exceeded")]
    fn pend_cap_exceeded_panics() {
        let mut c = decoupled(DocLayout::Rows, 20, 0, 1);
        c.policy = Policy::Batched { b: 20, k: 0, theta: 15, age: usize::MAX / 4 };
        let (mut r, later, mut rng) = setup(c, 4);
        for (w, d) in later.iter() {
            r.step(w, *d, &mut rng);
        }
    }

    /// K = 0: no ValMap writes between rebuilds; the log carries everything
    #[test]
    fn static_baseline_k0() {
        for (le, ve, cap) in [(150, 150, 0), (100, 300, 300)] {
            let mut c = decoupled(DocLayout::Rows, le, ve, cap);
            c.policy = Policy::Batched { b: 20, k: 0, theta: 15, age: usize::MAX / 4 };
            let (mut r, later, mut rng) = setup(c, 29);
            for (i, (w, d)) in later.iter().enumerate() {
                let (v0, n0) = (r.m.val_epoch, r.m.val.len());
                r.step(w, *d, &mut rng);
                if r.m.val_epoch == v0 {
                    assert_eq!(r.m.val.len(), n0, "ValMap written between rebuilds");
                }
                if i % 25 == 0 {
                    r.check();
                }
            }
            assert_eq!(r.m.batch, 0, "K = 0 runs no batches");
            assert!(r.m.val_rebuilds >= 2);
            // pending postings at a log-only compaction: at most E_V - E_L
            assert!(r.m.pend_snap_max <= ve - le && (cap == 0 || r.m.pend_snap_max > 0));
            r.check();
        }
    }

    /// With pend_cap > 0 a forced compaction (DocMap repack) is a log compaction only.
    #[test]
    fn forced_compaction_keeps_valmap() {
        let mut c = decoupled(DocLayout::Rows, 0, 0, 3000);
        c.doc_rows = 1040;
        c.doc_live_cap = 1000;
        let (mu_d, cap) = (c.doc_rows, c.doc_live_cap);
        let (mut r, later, mut rng) = setup(c, 9);
        let mut repacks = 0;
        for (w, d) in later.iter() {
            let (l0, v0) = (r.m.epoch, r.m.val_epoch);
            r.step(w, *d, &mut rng);
            if r.m.epoch != l0 {
                assert_eq!(r.m.val_epoch, v0, "a forced compaction rebuilt ValMap");
                assert!(r.m.virgin_rows() <= mu_d - cap);
                repacks += 1;
                r.check();
            }
        }
        assert!(repacks > 0, "test should force a repack");
        assert!(r.m.pend_snap_max > 0);
        r.check();
    }

    /// A dry run (`pend_measure`) follows exactly the trajectory of the real
    /// run, so its largest pending count sizes `pend_cap` for that run.
    #[test]
    fn measure_mode_matches_real_run() {
        let real = decoupled(DocLayout::Rows, 60, 0, 3000);
        let mut dry = real.clone();
        dry.pend_cap = 0;
        dry.pend_measure = true;
        let (mut a, later, mut ra) = setup(real, 31);
        let (mut b, _, mut rb) = setup(dry, 31);
        for (w, d) in later.iter() {
            a.step(w, *d, &mut ra);
            b.step(w, *d, &mut rb);
            assert_eq!(a.m.pend_total, b.m.pend_total);
        }
        assert_eq!(a.m.pend_snap_max, b.m.pend_snap_max);
        assert!(b.m.pend_snap_max > 0);
        assert_eq!(a.m.compacted_records, a.m.cfg.snapshot_records());
        assert_eq!(a.refmm, b.refmm);
        a.check();
        b.check();
    }

    /// age rule (log budget): age 3 batches of 20 appends, theta high
    fn aged(k: usize, measure: bool) -> LogCfg {
        aged_p(k, measure, false)
    }
    fn aged_p(k: usize, measure: bool, phased: bool) -> LogCfg {
        let mut c = decoupled(DocLayout::Rows, 60, 0, 3000);
        c.policy = Policy::Batched { b: 20, k, theta: 50, age: 3 };
        c.k_measure = measure;
        c.age_phased = phased;
        c
    }

    /// A K dry run (`k_measure`) serves all due work; with K = its maximum the
    /// real run has no K-overflow and the same trajectory (batches have their
    /// own randomness), so every posting is committed within age * b appends.
    #[test]
    fn k_measure_sizes_k_and_age_bounds_the_tail() {
        for phased in [false, true] {
            k_measure_and_tail(phased);
        }
    }

    fn k_measure_and_tail(phased: bool) {
        let aged = |k, m| aged_p(k, m, phased);
        let (mut d, later, mut rd) = setup(aged(1, true), 41);
        d.m.kappa_window = 60;
        for (w, x) in later.iter() {
            d.step(w, *x, &mut rd);
        }
        let kmax = d.m.max_batch_ops;
        assert!(kmax > 1 && d.m.k_overflow_batches == 0);
        let (mut r, _, mut rr) = setup(aged(kmax, false), 41);
        r.m.kappa_window = 60;
        let (mut dd, _, mut rdd) = setup(aged(1, true), 41);
        for (w, x) in later.iter() {
            r.step(w, *x, &mut rr);
            dd.step(w, *x, &mut rdd);
            assert_eq!(r.m.pend_total, dd.m.pend_total, "the dry run diverged");
            // the uncommitted tail: no pending posting is older than age batches
            let now = r.m.batch;
            for k in &r.m.kws {
                if let Some(&(_, j)) = k.pend.first() {
                    assert!(now - j < 3, "a posting waited {} batches", now - j);
                }
            }
        }
        assert_eq!(r.m.k_overflow_batches, 0);
        assert_eq!(r.m.max_batch_ops, kmax);
        assert!(r.m.max_pend_age <= 3);
        assert_eq!(r.refmm, d.refmm);
        // kappa_A: committed keywords per window, at most the active ones of
        // this and the previous windows, and the same as in the dry run
        assert!(!r.m.kappa_committed.is_empty());
        assert_eq!(r.m.kappa_committed, d.m.kappa_committed);
        assert_eq!(r.m.kappa_active, d.m.kappa_active);
        assert!(r.m.kappa_committed.iter().all(|&x| x <= 40) && r.m.kappa_active.iter().any(|&x| x > 0));
        r.check();
        d.check();
    }

    /// With K below the need, batches overflow (counted) but stay exactly K wide.
    #[test]
    fn small_k_overflows_but_stays_padded() {
        let (mut r, later, mut rng) = setup(aged(1, false), 43);
        for (w, x) in later.iter() {
            let a = r.m.insert(w, *x, 1000, &mut rng);
            r.refmm.entry(w.clone()).or_default().push(*x);
            for b in &a.batches {
                assert_eq!((b.real_ops + b.dummy_ops, b.records), (1, 1));
            }
        }
        assert!(r.m.k_overflow_batches > 0);
        r.check();
    }

    /// val_live tracks the live chunks exactly; more than val_cap halts.
    #[test]
    fn live_chunks_capped_at_all_times() {
        let (mut r, later, mut rng) = setup(decoupled(DocLayout::Rows, 60, 0, 3000), 47);
        let c = r.m.cfg.chunk_cap();
        for (w, d) in later.iter() {
            r.step(w, *d, &mut rng);
            let live: usize = r.m.kws.iter().map(|k| (k.committed.len() + c - 1) / c).sum();
            assert_eq!(r.m.val_live, live);
            assert!(r.m.val_live_max >= live && r.m.val_live <= r.m.cfg.val_cap);
            assert_eq!(r.m.val.len(), live, "reuse mode: the table holds exactly the live chunks");
        }
        // a cap just above the setup chunks: the first batches that add chunks halt it
        let mut c2 = decoupled(DocLayout::Rows, 60, 0, 3000);
        let (probe, _, _) = setup(c2.clone(), 47);
        c2.val_cap = probe.m.val_live;
        let (mut r2, later2, mut rng2) = setup(c2, 47);
        for (w, d) in later2.iter() {
            if r2.m.halted {
                break;
            }
            r2.m.insert(w, *d, 1000, &mut rng2);
        }
        assert!(r2.m.halted && r2.m.halt_reason.contains("ValMap capacity exceeded"), "{}", r2.m.halt_reason);
    }
}
