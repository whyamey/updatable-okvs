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
//!   Every compaction starts a new epoch, so tags never repeat across epochs.
//! * **DocMap**: either rows addressed by number (`DocLayout::Rows`; the row
//!   numbers live in the document version records, chunks are written to
//!   never-used rows, freed rows are reused only after a DocMap compaction)
//!   or a UOKVS keyed by `(id, ver, j)` (`DocLayout::Cuckoo`, for comparison).
//!
//! Schedules (all public; they depend only on the number of log appends):
//! * `Policy::Batched{b, k, theta, age}`: every `b` appends the owner runs
//!   exactly `k` ValMap operations (dummies pad) and appends exactly `k` flush
//!   records (dummies pad). Keyword w is due once it has `theta` pending
//!   postings or its oldest pending posting is `age` batches old; due keywords
//!   are served first-due-first. Whether w is due depends only on w's own
//!   history; serving more than `k` operations in one batch is a K-overflow
//!   (counted) and delays the rest.
//!   `b = 1, k = 1` is the old one-operation-per-insertion design.
//! * `Policy::Rebuild{r}`: no flushes; a major compaction every `r` appends
//!   (the static-PIR-plus-rebuild baseline).
//! * Major compaction every `major_every` appends (and whenever the DocMap
//!   runs out of never-used rows): ValMap is rebuilt from all postings with
//!   tombstones applied (sequential disciplined Encode), and the log restarts
//!   in a new epoch with one flush record per keyword (padded to `kw_pad`)
//!   and the current version of every document. Nothing is pending after a
//!   major compaction, so the compacted log has a public size.

use crate::pir_backend::{frodo_hint_bytes, frodo_row_delta_bytes};
use crate::uokvs::{derive, Key, OpOut, OwnerKeys, Params, Table, UpdateMode, CHECK, K32, META, NONCE};
use rand::seq::SliceRandom;
use rand::Rng;
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
    /// UOKVS keyed `(id, j)`; edits and removals are invalidation-mode
    /// updates (the highest version wins), so clients need no version records.
    /// Used by the stateless client model.
    Fixed,
}

#[derive(Clone, Debug)]
pub struct LogCfg {
    /// ValMap table (p = 1; val_len = chunk plaintext bytes)
    pub val: Params,
    pub posting_bytes: usize,
    pub policy: Policy,
    /// log appends between major compactions (0 = never, except when the
    /// DocMap runs out of never-used rows). A major compaction rebuilds ValMap
    /// with all postings (tombstones applied), and rewrites the log.
    pub major_every: usize,
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
}

impl LogCfg {
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
}

/// What one owner action did, for accounting.
#[derive(Clone, Debug, Default)]
pub struct Action {
    pub appends: usize,
    pub val_ops: Vec<OpOut>,
    pub doc_rows_written: Vec<usize>,
    pub doc_ops: Vec<OpOut>,
    pub batches: Vec<BatchOut>,
    pub compactions: usize,
    pub rebuilds: Vec<f64>,
    pub doc_compactions: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BatchOut {
    pub index: u64,
    pub real_ops: usize,
    pub dummy_ops: usize,
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
    pub tag_probes: usize,
}

pub struct MkseLog {
    pub cfg: LogCfg,
    ki: OwnerKeys,
    kl: OwnerKeys,
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
    since_major: usize,
    // counters
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
        let ki = OwnerKeys::random(rng);
        let kl = OwnerKeys::random(rng);
        let kd = OwnerKeys::random(rng);
        let val = Table::new(cfg.val.clone(), rng);
        let doc_table = match cfg.doc_layout {
            DocLayout::Cuckoo | DocLayout::Fixed => Some(Table::new(cfg.doc.clone(), rng)),
            DocLayout::Rows => None,
        };
        let mut s = Self {
            ki,
            kl,
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
            since_major: 0,
            total_appends: 0,
            k_overflow_batches: 0,
            deferred_total: 0,
            pend_total: 0,
            pend_peak: 0,
            max_pend_age: 0,
            pend_pad_overflow: 0,
            fillers: 0,
            superseded: 0,
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
            let mut d = Doc { cap, ver: 0, nch, rows: vec![], deleted: false, m: 0, keywords: doc_kws.remove(&id).unwrap() };
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
                        let k = Self::doc_key(id, 0, j);
                        let keys = derive(&cap, k.idx);
                        cuckoo_entries.push((k, keys, vec![0u8; 1]));
                    }
                }
            }
            s.docs.insert(id, d);
        }
        if let Some(t) = s.doc_table.as_mut() {
            t.encode(cuckoo_entries, rng);
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
        s.val.encode(entries, rng);
        // initial log = epoch-0 snapshot
        s.write_snapshot(rng);
        s
    }

    fn kw_index(&mut self, w: &str) -> u32 {
        if let Some(&i) = self.kw_ix.get(w) {
            return i;
        }
        let i = self.kws.len() as u32;
        let cap_val = self.ki.cap(w.as_bytes());
        let cap_log = self.kl.cap(w.as_bytes()).0;
        self.kws.push(Kw {
            name: w.to_string(),
            cap_val,
            cap_log,
            committed: vec![],
            pend: vec![],
            n: 0,
            m: 0,
            due_since: None,
        });
        self.kw_ix.insert(w.to_string(), i);
        i
    }

    /// Writes a fresh log for the current epoch: one flush record per keyword
    /// with postings (dummies pad to `kw_pad`), any pending postings (none
    /// after a major compaction), and the current version of every document.
    fn write_snapshot<R: Rng>(&mut self, rng: &mut R) {
        self.log.clear();
        self.tag_ix.clear();
        self.log_live_since_compaction = 0;
        let mut kw_records = 0usize;
        let mut pend_records = 0usize;
        for w in 0..self.kws.len() as u32 {
            self.kws[w as usize].m = 0;
            if self.kws[w as usize].n == 0 {
                continue;
            }
            self.append_flush(w);
            kw_records += 1;
            let (cap, vc, pend) = {
                let k = &self.kws[w as usize];
                (k.cap_log, k.committed.len(), k.pend.iter().map(|x| x.0).collect::<Vec<_>>())
            };
            for (o, p) in pend.into_iter().enumerate() {
                let t = tag_of(&cap, b"post", self.epoch, (vc + o) as u64, 0);
                self.push_rec(t, Body::Post(p));
                pend_records += 1;
            }
        }
        for _ in kw_records..self.cfg.kw_pad {
            self.append_dummy(rng);
        }
        let mut ids: Vec<u32> = self.docs.keys().copied().collect();
        ids.sort_unstable();
        let mut doc_records = 0usize;
        for id in ids {
            self.docs.get_mut(&id).unwrap().m = 0;
            doc_records += self.cfg.doc_records(self.docs[&id].nch as usize);
            self.append_doc_version(id);
        }
        // leakage check: with nothing pending, the compacted log has the public
        // size kw_pad + (records of every document ever inserted)
        if pend_records == 0 && kw_records <= self.cfg.kw_pad {
            assert_eq!(self.log.len(), self.cfg.kw_pad + doc_records);
        }
        self.compacted_records = self.log.len();
        self.log_live_since_compaction = 0;
    }

    // ------------------------------------------------------------ owner ops

    fn tick<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        self.total_appends += 1;
        self.since_batch += 1;
        self.since_major += 1;
        act.appends += 1;
        let major_every = match self.cfg.policy {
            Policy::Batched { b, k, .. } => {
                if k > 0 && self.since_batch >= b {
                    self.since_batch = 0;
                    let bo = self.run_batch(rng);
                    act.batches.push(bo);
                }
                self.cfg.major_every
            }
            Policy::Rebuild { r } => r,
        };
        if major_every > 0 && self.since_major >= major_every {
            self.major(act, rng);
        }
    }

    /// Major compaction: rebuild ValMap (all postings, tombstones applied)
    /// and rewrite the log in a new epoch.
    pub fn major<R: Rng>(&mut self, act: &mut Action, rng: &mut R) {
        let t = Instant::now();
        self.since_major = 0;
        self.rebuild_val(rng);
        self.compact(rng);
        act.rebuilds.push(t.elapsed().as_secs_f64() * 1e3);
        act.compactions += 1;
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
        self.pend_total += 1;
        self.pend_peak = self.pend_peak.max(self.pend_total);
        self.tick(act, rng);
    }

    /// Inserts posting (w, id); a new id also inserts the document.
    pub fn insert<R: Rng>(&mut self, w: &str, id: u32, doc_len: usize, rng: &mut R) -> Action {
        let mut act = Action::default();
        let wi = self.kw_index(w);
        if !self.docs.contains_key(&id) || self.docs[&id].deleted {
            self.insert_doc(id, doc_len, &mut act, rng);
        }
        let d = self.docs.get_mut(&id).unwrap();
        d.keywords.push(wi);
        let nch = d.nch;
        self.add_posting(wi, Posting { id, nch, tomb: false }, &mut act, rng);
        act
    }

    pub fn delete_posting<R: Rng>(&mut self, w: &str, id: u32, rng: &mut R) -> Action {
        let mut act = Action::default();
        let wi = self.kw_index(w);
        if let Some(d) = self.docs.get_mut(&id) {
            d.keywords.retain(|&x| x != wi);
        }
        self.add_posting(wi, Posting { id, nch: 0, tomb: true }, &mut act, rng);
        act
    }

    fn alloc_rows<R: Rng>(&mut self, n: usize, act: &mut Action, rng: &mut R) -> Vec<u32> {
        if self.virgin.len() < n {
            self.compact_docmap(rng);
            act.doc_compactions += 1;
            self.major(act, rng);
            assert!(self.virgin.len() >= n, "DocMap too small even after compaction; raise --doc-spare");
        }
        (0..n).map(|_| self.virgin.pop().unwrap()).collect()
    }

    fn write_doc_chunks<R: Rng>(&mut self, id: u32, act: &mut Action, rng: &mut R) {
        let (nch, ver, cap) = {
            let d = &self.docs[&id];
            (d.nch, d.ver, d.cap)
        };
        match self.cfg.doc_layout {
            DocLayout::Rows => {
                let rows = self.alloc_rows(nch as usize, act, rng);
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
                let t = self.doc_table.as_mut().unwrap();
                for j in 0..nch {
                    let k = Self::doc_key(id, 0, j);
                    let v = vec![(ver % 251) as u8; 1];
                    if t.contains(&k) {
                        act.doc_ops.push(t.update(&k, v, false, UpdateMode::Invalidation, rng));
                    } else {
                        let (lk, dk) = derive(&cap, k.idx);
                        act.doc_ops.push(t.insert(&k, lk, dk, v, rng));
                    }
                }
            }
        }
    }

    fn free_doc_chunks(&mut self, id: u32) {
        let (nch, ver, rows) = {
            let d = &self.docs[&id];
            (d.nch, d.ver, d.rows.clone())
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
            // fixed keys are overwritten by the next version (or a tombstone)
            DocLayout::Fixed => {}
        }
        self.docs.get_mut(&id).unwrap().rows.clear();
    }

    fn insert_doc<R: Rng>(&mut self, id: u32, len: usize, act: &mut Action, rng: &mut R) {
        let mut nch = self.doc_nch(len);
        if self.cfg.doc_layout == DocLayout::Fixed {
            // fixed keys: a re-inserted identifier keeps its chunk count
            if let Some(d) = self.docs.get(&id) {
                nch = d.nch;
            }
        }
        let prev = self.docs.get(&id).map(|d| (d.ver, d.m));
        let cap = self.kd.cap(&id.to_le_bytes());
        let (ver, m) = prev.map(|(v, m)| (v + 1, m)).unwrap_or((0, 0));
        self.docs.insert(id, Doc { cap, ver, nch, rows: vec![], deleted: false, m, keywords: vec![] });
        self.write_doc_chunks(id, act, rng);
        self.append_doc_version(id);
        for _ in 0..self.cfg.doc_records(nch as usize) {
            self.tick(act, rng);
        }
    }

    /// Replaces a document's content (new length `len`).
    pub fn update_doc<R: Rng>(&mut self, id: u32, len: usize, rng: &mut R) -> Action {
        let mut act = Action::default();
        if self.docs.get(&id).map_or(true, |d| d.deleted) {
            return act;
        }
        self.free_doc_chunks(id);
        let nch = if self.cfg.doc_layout == DocLayout::Fixed {
            // fixed keys: an edit keeps the chunk count (padded or truncated),
            // so it does not even reveal the new length
            self.docs[&id].nch
        } else {
            self.doc_nch(len)
        };
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
        act
    }

    /// Deletes a document: a version record marks it deleted, its postings get
    /// tombstones, its chunks are freed (no server-visible write).
    pub fn delete_doc<R: Rng>(&mut self, id: u32, rng: &mut R) -> Action {
        let mut act = Action::default();
        if self.docs.get(&id).map_or(true, |d| d.deleted) {
            return act;
        }
        if self.cfg.doc_layout == DocLayout::Fixed {
            let nch = self.docs[&id].nch;
            let t = self.doc_table.as_mut().unwrap();
            for j in 0..nch {
                act.doc_ops.push(t.update(&Self::doc_key(id, 0, j), vec![], true, UpdateMode::Invalidation, rng));
            }
        }
        self.free_doc_chunks(id);
        let kws = {
            let d = self.docs.get_mut(&id).unwrap();
            d.deleted = true;
            std::mem::take(&mut d.keywords)
        };
        let nch = self.docs[&id].nch as usize;
        self.append_doc_version(id);
        for _ in 0..self.cfg.doc_records(nch) {
            self.tick(&mut act, rng);
        }
        for w in kws {
            self.add_posting(w, Posting { id, nch: 0, tomb: true }, &mut act, rng);
        }
        act
    }

    // ------------------------------------------------------------ batches

    fn run_batch<R: Rng>(&mut self, rng: &mut R) -> BatchOut {
        let t = Instant::now();
        self.batch += 1;
        let (k, theta, age) = match self.cfg.policy {
            Policy::Batched { k, theta, age, .. } => (k, theta, age as u64),
            _ => unreachable!(),
        };
        let now = self.batch;
        let mut due: Vec<(u64, u32)> = Vec::new();
        for &w in &self.pending_kws {
            let kw = &mut self.kws[w as usize];
            let oldest = kw.pend.first().map(|x| x.1).unwrap_or(now);
            self.max_pend_age = self.max_pend_age.max(now - oldest);
            if kw.pend.len() >= theta || now - oldest >= age {
                if kw.due_since.is_none() {
                    kw.due_since = Some(now);
                }
                due.push((kw.due_since.unwrap(), w));
            }
        }
        due.sort_unstable();
        let mut out = BatchOut { index: now, due: due.len(), ..Default::default() };
        let mut ops_left = k;
        let mut served: Vec<u32> = Vec::new();
        for &(_, w) in &due {
            if ops_left == 0 {
                out.deferred += 1;
                continue;
            }
            let (ops, moved) = self.flush_kw(w, ops_left, rng, &mut out.accessed);
            ops_left -= ops;
            out.real_ops += ops;
            out.postings_flushed += moved;
            if ops > 0 {
                served.push(w);
            }
            if !self.kws[w as usize].pend.is_empty() {
                out.deferred += 1;
            }
        }
        for _ in 0..ops_left {
            let o = if self.cfg.reuse_freed {
                self.val.dummy(rng)
            } else {
                // filler insertion of a fresh random key, dropped at compaction
                self.fillers += 1;
                let key = Key::sub(b"\0filler", self.fillers);
                let mut lk = [0u8; 32];
                let mut dk = [0u8; 32];
                rng.fill(&mut lk);
                rng.fill(&mut dk);
                self.val.insert(&key, lk, dk, vec![], rng)
            };
            out.accessed.extend_from_slice(&o.accessed);
            out.dummy_ops += 1;
        }
        out.keywords_served = served.len();
        assert!(served.len() <= k);
        for &w in &served {
            self.append_flush(w);
        }
        for _ in served.len()..k {
            self.append_dummy(rng);
        }
        if self.cfg.stateless && self.pend_total > self.cfg.pend_pad {
            self.pend_pad_overflow += 1;
        }
        if out.deferred > 0 {
            self.k_overflow_batches += 1;
            self.deferred_total += out.deferred;
        }
        assert_eq!(out.accessed.len(), k * self.cfg.val.accesses_per_op(), "a batch must make exactly k(alpha+1) accesses");
        out.owner_ms = t.elapsed().as_secs_f64() * 1e3;
        out
    }

    /// Commits pending postings of w using at most `budget` ValMap operations:
    /// first fills the partial last chunk (a fresh key (w, i, len')), then
    /// starts new chunks. Returns (operations, postings committed).
    fn flush_kw<R: Rng>(&mut self, w: u32, budget: usize, rng: &mut R, acc: &mut Vec<usize>) -> (usize, usize) {
        let c = self.cfg.chunk_cap();
        let mut ops = 0;
        let mut moved = 0;
        while ops < budget && !self.kws[w as usize].pend.is_empty() {
            let (name, vc, npend) = {
                let k = &self.kws[w as usize];
                (k.name.clone(), k.committed.len(), k.pend.len())
            };
            let i = vc / c;
            let have = vc % c;
            let take = (c - have).min(npend);
            if have > 0 && self.cfg.reuse_freed {
                let old = self.chunk_key(&name, i, have);
                let removed = self.val.remove(&old);
                debug_assert!(removed, "missing chunk {name}/{i}/{have}");
            } else if have > 0 {
                self.superseded += 1;
            }
            let mut content: Vec<Posting> = self.kws[w as usize].committed[i * c..].to_vec();
            let drained: Vec<(Posting, u64)> = self.kws[w as usize].pend.drain(..take).collect();
            content.extend(drained.iter().map(|x| x.0));
            let key = self.chunk_key(&name, i, have + take);
            let (lk, dk) = self.chunk_keys(&self.kws[w as usize], &key);
            let o = self.val.insert(&key, lk, dk, enc_postings(&content, self.cfg.posting_bytes), rng);
            acc.extend_from_slice(&o.accessed);
            let kw = &mut self.kws[w as usize];
            kw.committed.extend(drained.iter().map(|x| x.0));
            ops += 1;
            moved += take;
        }
        self.pend_total -= moved;
        let kw = &mut self.kws[w as usize];
        if kw.pend.is_empty() {
            kw.due_since = None;
            self.pending_kws.remove(&w);
        }
        (ops, moved)
    }

    // ------------------------------------------------------------ compaction

    pub fn compact<R: Rng>(&mut self, rng: &mut R) {
        self.epoch += 1;
        self.write_snapshot(rng);
    }

    /// Re-places every live document chunk in a fresh random order of rows;
    /// all rows are rewritten (clients take a fresh DocMap hint). The caller
    /// runs a major compaction so version records carry the new row numbers.
    pub fn compact_docmap<R: Rng>(&mut self, rng: &mut R) {
        let n = self.doc_rows.len();
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
        self.virgin = perm;
        self.doc_freed = 0;
    }

    /// Static rebuild of ValMap (baseline): every keyword's postings, with
    /// tombstones applied, re-chunked canonically into a new table.
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
        self.val = Table::new(self.cfg.val.clone(), rng);
        self.val.encode(entries, rng);
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
        // 4. documents: latest version record from the log
        for &id in &out.ids {
            let d = &self.docs[&id];
            if self.cfg.doc_layout == DocLayout::Fixed {
                // the chunk count comes with the posting; the newest copy wins
                let t = self.doc_table.as_ref().unwrap();
                for j in 0..d.nch {
                    let k = Self::doc_key(id, 0, j);
                    let (lk, _) = derive(&d.cap, k.idx);
                    let (_, tomb) = t.lookup(&k, &lk).expect("document chunk missing");
                    assert!(!tomb, "search returned a removed document");
                    out.doc_queries.extend(t.candidates(&lk));
                }
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
}

pub const PIR_COMBOS: [(PirKind, PirKind); 4] = [
    (PirKind::Frodo, PirKind::Frodo),
    (PirKind::Spiral, PirKind::Frodo),
    (PirKind::Frodo, PirKind::Spiral),
    (PirKind::Spiral, PirKind::Spiral),
];

pub fn pir_name(k: PirKind) -> &'static str {
    match k {
        PirKind::Frodo => "frodo",
        PirKind::Spiral => "spiral",
    }
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
}

impl MapPir {
    pub fn new(rows: usize, row_bytes: usize) -> Self {
        Self {
            frodo_query: rows * 4,
            frodo_response: crate::pir_backend::frodo_width(row_bytes) * 4,
            spiral_query: SPIRAL_QUERY_BYTES,
            spiral_response: spiral_response_bytes(row_bytes),
        }
    }
    pub fn per_query(&self, k: PirKind) -> usize {
        match k {
            PirKind::Frodo => self.frodo_query + self.frodo_response,
            PirKind::Spiral => self.spiral_query + self.spiral_response,
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
        let h = |k: PirKind, hint: usize| if k == PirKind::Frodo { hint } else { SPIRAL_PP_BYTES };
        state + h(val, self.val_hint_bytes()) + h(doc, self.doc_hint_bytes())
    }
}

/// A client that searches every `interval` appends and refreshes its state
/// only then. Its download is split into components so that every pairing of
/// PIRs (ValMap, DocMap) can be costed from one run:
/// * sync: new log records (or the compacted log), or for stateless clients
///   VolMap plus the pending stash (when a batch ran since the last search);
/// * stash: the ValMap and DocMap stashes if they changed;
/// * hints (FrodoPIR only): packed deltas of changed rows, or a fresh hint,
///   whichever is smaller;
/// * queries and responses of the search itself, per PIR.
#[derive(Clone, Debug)]
pub struct Profile {
    pub interval: usize,
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
    pub stash: usize,
    pub val_queries: usize,
    pub doc_queries: usize,
    pub full_hint_refreshes: usize,
    /// per pairing in PIR_COMBOS order: total and max bytes per search
    pub total: [usize; 4],
    pub max: [usize; 4],
    pub hist: [Vec<usize>; 4],
}

impl Profile {
    pub fn new(interval: usize) -> Self {
        Self {
            interval,
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
            stash: 0,
            val_queries: 0,
            doc_queries: 0,
            full_hint_refreshes: 0,
            total: [0; 4],
            max: [0; 4],
            hist: [vec![], vec![], vec![], vec![]],
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
            let appended = act.appends + act.batches.len() * match m.cfg.policy { Policy::Batched { k, .. } => k, _ => 0 };
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

    /// The client refreshes and runs a search that sends `qv` ValMap and `qd`
    /// DocMap queries. Returns the bytes for the pairing (Frodo, Frodo).
    pub fn sync(&mut self, m: &MkseLog, qv: usize, qd: usize) -> usize {
        let hv = if self.val_full { self.full_hint_refreshes += 1; m.val_hint_bytes() } else { self.val_rows.len() * m.val_delta_bytes(true) };
        let hd = if self.doc_full { m.doc_hint_bytes() } else { self.doc_rows.len() * m.doc_delta_bytes(true) };
        let mut st = if self.dirty { m.val_stash_bytes() } else { 0 };
        if self.doc_dirty && m.cfg.doc_layout != DocLayout::Rows {
            st += m.cfg.doc.stash_bytes();
        }
        let sync = if m.cfg.stateless {
            if self.dirty { m.cfg.volmap_bytes() + m.cfg.pending_stash_bytes() } else { 0 }
        } else {
            self.log_bytes
        };
        let (vp, dp) = (m.val_pir(), m.doc_pir());
        for (c, &(kv, kd)) in PIR_COMBOS.iter().enumerate() {
            let mut b = sync + st + qv * vp.per_query(kv) + qd * dp.per_query(kd);
            if kv == PirKind::Frodo {
                b += hv;
            }
            if kd == PirKind::Frodo {
                b += hd;
            }
            self.total[c] += b;
            self.max[c] = self.max[c].max(b);
            self.hist[c].push(b);
        }
        self.hint_val += hv;
        self.hint_doc += hd;
        self.stash += st;
        self.sync += sync;
        self.val_queries += qv;
        self.doc_queries += qd;
        self.searches += 1;
        self.val_rows.clear();
        self.doc_rows.clear();
        self.val_full = false;
        self.doc_full = false;
        self.log_bytes = 0;
        self.dirty = false;
        self.doc_dirty = false;
        self.hist[0].last().copied().unwrap_or(0)
    }
}
