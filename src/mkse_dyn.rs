//! Dynamic MKSE from three UOKVS instances (Figure `fig:composition`), with
//! insertions, deletions (tombstones), and document updates.
//!
//! * VolMap: stash-only map (mu = 0), keyword -> number of flushed chunks.
//! * ValMap: hierarchical map, (keyword, j) -> chunk of postings. New postings
//!   are buffered in the ValMap stash ("pending" slots) and flushed into a fresh
//!   chunk key once `flush_at` postings are pending for that keyword
//!   (default: half a chunk). Every MKSE operation makes the same number of
//!   ValMap accesses: a flush is an insertion, otherwise a dummy operation.
//! * DocMap: hierarchical map, (id, c) -> c-th chunk of the document.
//!
//! Deletions add a tombstone posting through the same pending path;
//! deleting a document also turns its DocMap chunks into tombstones with an
//! invalidation-mode update.

use crate::uokvs::{derive, Key, OpOut, OwnerKeys, Params, Table, UpdateMode, K32};
use rand::Rng;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug)]
pub struct MkseCfg {
    pub val: Params,
    pub doc: Params,
    /// capacity of the stash-only VolMap
    pub vol_sigma: usize,
    /// pending-posting slots in the ValMap stash
    pub pend_sigma: usize,
    /// bytes per posting: id (4) + chunk count (2) + flags (1) + DocMap capability (32) + pad
    pub posting_bytes: usize,
    /// flush threshold in postings
    pub flush_at: usize,
    /// optional cap on pending postings: when exceeded, the keyword with the
    /// most pending postings is flushed as a partial chunk (one flush per op)
    pub pend_budget: Option<usize>,
    pub update_mode: UpdateMode,
    /// bytes per DocMap chunk (= doc.val_len)
    pub doc_chunk_bytes: usize,
}

impl MkseCfg {
    pub fn chunk_cap(&self) -> usize {
        self.val.val_len / self.posting_bytes
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Posting {
    pub id: u32,
    pub nch: u16,
    pub tomb: bool,
}

#[derive(Clone, Debug)]
pub struct DocInfo {
    pub nch: u16,
    pub keywords: Vec<String>,
    pub deleted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    Insert,
    DeletePosting,
    DeleteDoc,
    UpdateDoc,
}

/// Server-visible effect of one MKSE operation plus owner counters.
#[derive(Clone, Debug)]
pub struct MkseOp {
    pub kind: OpKind,
    /// ValMap operations (one per posting added: a flush insertion or a dummy)
    pub val: Vec<OpOut>,
    /// DocMap operations (insertions or updates of chunks)
    pub doc: Vec<OpOut>,
    pub flushes: usize,
    pub new_doc: bool,
    /// number of VolMap / ValMap-stash refreshes
    pub stash_refreshes: usize,
}

impl MkseOp {
    fn new(kind: OpKind) -> Self {
        Self { kind, val: vec![], doc: vec![], flushes: 0, new_doc: false, stash_refreshes: 0 }
    }
    pub fn val_accesses(&self) -> usize {
        self.val.iter().map(|o| o.accessed.len()).sum()
    }
    pub fn doc_accesses(&self) -> usize {
        self.doc.iter().map(|o| o.accessed.len()).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchOut {
    pub ids: Vec<u32>,
    pub chunks: usize,
    /// ValMap PIR positions (s per chunk)
    pub val_queries: Vec<usize>,
    /// DocMap PIR positions (s per document chunk)
    pub doc_queries: Vec<usize>,
    pub pending_seen: usize,
}

pub struct Mkse {
    pub cfg: MkseCfg,
    kv: OwnerKeys,
    ki: OwnerKeys,
    kd: OwnerKeys,
    pub vol: Table,
    pub val: Table,
    pub doc: Table,
    chunks: HashMap<String, u32>,
    pending: HashMap<String, Vec<Posting>>,
    pub pend_total: usize,
    pub pend_peak: usize,
    pub pend_overflow: usize,
    pub docs: HashMap<u32, DocInfo>,
}

fn enc_postings(ps: &[Posting], pb: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(ps.len() * pb);
    for p in ps {
        let start = v.len();
        v.extend_from_slice(&p.id.to_le_bytes());
        v.extend_from_slice(&p.nch.to_le_bytes());
        v.push(p.tomb as u8);
        // capability placeholder (the real capability is derivable by the owner;
        // its bytes only matter for size)
        v.resize(start + pb, 0);
    }
    v
}

fn dec_postings(b: &[u8], pb: usize) -> Vec<Posting> {
    b.chunks(pb)
        .filter(|c| c.len() >= 7)
        .map(|c| Posting {
            id: u32::from_le_bytes(c[0..4].try_into().unwrap()),
            nch: u16::from_le_bytes(c[4..6].try_into().unwrap()),
            tomb: c[6] != 0,
        })
        .collect()
}

impl Mkse {
    fn vol_keys(&self, w: &str) -> (K32, K32) {
        derive(&self.kv.cap(w.as_bytes()), None)
    }
    fn val_keys(&self, w: &str, j: u64) -> (K32, K32) {
        derive(&self.ki.cap(w.as_bytes()), Some(j))
    }
    fn doc_keys(&self, id: u32, c: u64) -> (K32, K32) {
        derive(&self.kd.cap(&id.to_le_bytes()), Some(c))
    }
    fn doc_nch(&self, len: usize) -> u16 {
        ((len + self.cfg.doc_chunk_bytes - 1) / self.cfg.doc_chunk_bytes).max(1) as u16
    }

    /// Setup: every keyword's postings are written as chunks (the last one may
    /// be partial).
    pub fn setup<R: Rng>(
        cfg: MkseCfg,
        mm: &[(String, Vec<u32>)],
        doc_len: &HashMap<u32, usize>,
        default_doc_len: usize,
        rng: &mut R,
    ) -> Self {
        let vol_prm = Params {
            mu: 0,
            p: 1,
            sigma: cfg.vol_sigma,
            val_len: 4,
            ..cfg.val.clone()
        };
        let mut m = Self {
            kv: OwnerKeys::random(rng),
            ki: OwnerKeys::random(rng),
            kd: OwnerKeys::random(rng),
            vol: Table::new(vol_prm, rng),
            val: Table::new(cfg.val.clone(), rng),
            doc: Table::new(cfg.doc.clone(), rng),
            chunks: HashMap::new(),
            pending: HashMap::new(),
            pend_total: 0,
            pend_peak: 0,
            pend_overflow: 0,
            docs: HashMap::new(),
            cfg,
        };
        let cap = m.cfg.chunk_cap();
        assert!(cap >= 1 && m.cfg.flush_at >= 1 && m.cfg.flush_at <= cap);

        // DocMap
        let mut doc_entries = Vec::new();
        for (w, ids) in mm {
            for &id in ids {
                let e = m.docs.entry(id).or_insert_with(|| DocInfo {
                    nch: 0,
                    keywords: vec![],
                    deleted: false,
                });
                e.keywords.push(w.clone());
            }
        }
        let mut doc_ids: Vec<u32> = m.docs.keys().copied().collect();
        doc_ids.sort_unstable();
        for &id in &doc_ids {
            let len = *doc_len.get(&id).unwrap_or(&default_doc_len);
            let nch = m.doc_nch(len);
            m.docs.get_mut(&id).unwrap().nch = nch;
            for c in 0..nch as u64 {
                let k = Key::sub(&id.to_le_bytes(), c);
                let keys = m.doc_keys(id, c);
                let clen = m.cfg.doc_chunk_bytes.min(len.saturating_sub(c as usize * m.cfg.doc_chunk_bytes)).max(1);
                doc_entries.push((k, keys, vec![0u8; clen]));
            }
        }
        m.doc.encode(doc_entries, rng);

        // ValMap + VolMap
        let mut val_entries = Vec::new();
        let mut vol_entries = Vec::new();
        for (w, ids) in mm {
            let ps: Vec<Posting> = ids
                .iter()
                .map(|&id| Posting { id, nch: m.docs[&id].nch, tomb: false })
                .collect();
            let mut j = 0u32;
            let mut rest = &ps[..];
            // Setup is a joint encoding, so the remainder becomes a (partial)
            // chunk too; only postings inserted later wait in the stash.
            while !rest.is_empty() {
                let take = rest.len().min(cap);
                let k = Key::sub(w.as_bytes(), j as u64);
                val_entries.push((k, m.val_keys(w, j as u64), enc_postings(&rest[..take], m.cfg.posting_bytes)));
                rest = &rest[take..];
                j += 1;
            }
            if j > 0 {
                m.chunks.insert(w.clone(), j);
                vol_entries.push((Key::flat(w.as_bytes()), m.vol_keys(w), j.to_le_bytes().to_vec()));
            }
        }
        m.pend_peak = m.pend_total;
        m.val.encode(val_entries, rng);
        m.vol.encode(vol_entries, rng);
        m
    }

    fn add_pending<R: Rng>(&mut self, w: &str, p: Posting, rng: &mut R, r: &mut MkseOp) {
        let list = self.pending.entry(w.to_string()).or_default();
        list.push(p);
        self.pend_total += 1;
        self.pend_peak = self.pend_peak.max(self.pend_total);
        if self.pend_total > self.cfg.pend_sigma {
            self.pend_overflow += 1;
        }
        r.stash_refreshes += 1;
        let mut target: Option<String> = None;
        if list.len() >= self.cfg.flush_at {
            target = Some(w.to_string());
        } else if let Some(b) = self.cfg.pend_budget {
            if self.pend_total > b {
                // flush the keyword with the most pending postings
                target = self
                    .pending
                    .iter()
                    .max_by_key(|(_, v)| v.len())
                    .map(|(k, _)| k.clone());
            }
        }
        if let Some(w) = target {
            let w = w.as_str();
            let take: Vec<Posting> = std::mem::take(self.pending.get_mut(w).unwrap());
            self.pend_total -= take.len();
            let j = *self.chunks.get(w).unwrap_or(&0);
            let k = Key::sub(w.as_bytes(), j as u64);
            let (lk, dk) = self.val_keys(w, j as u64);
            let o = self.val.insert(&k, lk, dk, enc_postings(&take, self.cfg.posting_bytes), rng);
            r.val.push(o);
            r.flushes += 1;
            self.chunks.insert(w.to_string(), j + 1);
            let vk = Key::flat(w.as_bytes());
            let cnt = (j + 1).to_le_bytes().to_vec();
            if j == 0 {
                let (lk, dk) = self.vol_keys(w);
                self.vol.insert(&vk, lk, dk, cnt, rng);
            } else {
                self.vol.update(&vk, cnt, false, UpdateMode::Invalidation, rng);
            }
        } else {
            r.val.push(self.val.dummy(rng));
        }
    }

    pub fn insert<R: Rng>(&mut self, w: &str, id: u32, doc_len: usize, rng: &mut R) -> MkseOp {
        let mut r = MkseOp::new(OpKind::Insert);
        let nch;
        match self.docs.get_mut(&id) {
            Some(d) if !d.deleted => {
                d.keywords.push(w.to_string());
                nch = d.nch;
            }
            _ => {
                nch = self.doc_nch(doc_len);
                for c in 0..nch as u64 {
                    let k = Key::sub(&id.to_le_bytes(), c);
                    let (lk, dk) = self.doc_keys(id, c);
                    if self.doc.contains(&k) {
                        // re-insertion of a deleted document id: overwrite its chunks
                        r.doc.push(self.doc.update(&k, vec![0u8; self.cfg.doc_chunk_bytes], false, self.cfg.update_mode, rng));
                    } else {
                        r.doc.push(self.doc.insert(&k, lk, dk, vec![0u8; self.cfg.doc_chunk_bytes], rng));
                    }
                }
                r.new_doc = true;
                self.docs.insert(id, DocInfo { nch, keywords: vec![w.to_string()], deleted: false });
            }
        }
        self.add_pending(w, Posting { id, nch, tomb: false }, rng, &mut r);
        r
    }

    pub fn delete_posting<R: Rng>(&mut self, w: &str, id: u32, rng: &mut R) -> MkseOp {
        let mut r = MkseOp::new(OpKind::DeletePosting);
        if let Some(d) = self.docs.get_mut(&id) {
            d.keywords.retain(|x| x != w);
        }
        self.add_pending(w, Posting { id, nch: 0, tomb: true }, rng, &mut r);
        r
    }

    pub fn delete_doc<R: Rng>(&mut self, id: u32, rng: &mut R) -> MkseOp {
        let mut r = MkseOp::new(OpKind::DeleteDoc);
        let Some(d) = self.docs.get(&id).cloned() else { return r };
        if d.deleted {
            return r;
        }
        for w in &d.keywords {
            self.add_pending(w, Posting { id, nch: 0, tomb: true }, rng, &mut r);
        }
        for c in 0..d.nch as u64 {
            let k = Key::sub(&id.to_le_bytes(), c);
            r.doc.push(self.doc.update(&k, vec![], true, self.cfg.update_mode, rng));
        }
        let e = self.docs.get_mut(&id).unwrap();
        e.deleted = true;
        e.keywords.clear();
        r
    }

    /// Replaces a document's content (same number of chunks).
    pub fn update_doc<R: Rng>(&mut self, id: u32, rng: &mut R) -> MkseOp {
        let mut r = MkseOp::new(OpKind::UpdateDoc);
        let Some(d) = self.docs.get(&id).cloned() else { return r };
        if d.deleted {
            return r;
        }
        for c in 0..d.nch as u64 {
            let k = Key::sub(&id.to_le_bytes(), c);
            r.doc.push(self.doc.update(&k, vec![1u8; self.cfg.doc_chunk_bytes], false, self.cfg.update_mode, rng));
        }
        // keep ValMap/VolMap access counts uniform across operation types
        r.val.push(self.val.dummy(rng));
        r.stash_refreshes += 1;
        r
    }

    /// Client search (plaintext simulation of Figure `fig:composition`).
    pub fn search(&self, w: &str) -> SearchOut {
        let mut out = SearchOut::default();
        let (vlk, _) = self.vol_keys(w);
        let nch = self
            .vol
            .lookup(&Key::flat(w.as_bytes()), &vlk)
            .map(|(v, _)| u32::from_le_bytes(v[..4].try_into().unwrap()))
            .unwrap_or(0);
        out.chunks = nch as usize;
        let mut live: Vec<u32> = Vec::new();
        let mut set: HashSet<u32> = HashSet::new();
        let mut apply = |p: &Posting, live: &mut Vec<u32>, set: &mut HashSet<u32>| {
            if p.tomb {
                set.remove(&p.id);
            } else if set.insert(p.id) {
                live.push(p.id);
            }
        };
        let mut nchs: HashMap<u32, u16> = HashMap::new();
        for j in 0..nch as u64 {
            let (lk, _) = self.val_keys(w, j);
            out.val_queries.extend(self.val.candidates(&lk));
            if let Some((v, _)) = self.val.lookup(&Key::sub(w.as_bytes(), j), &lk) {
                for p in dec_postings(&v, self.cfg.posting_bytes) {
                    if !p.tomb {
                        nchs.insert(p.id, p.nch);
                    }
                    apply(&p, &mut live, &mut set);
                }
            }
        }
        if let Some(ps) = self.pending.get(w) {
            out.pending_seen = ps.len();
            for p in ps {
                if !p.tomb {
                    nchs.insert(p.id, p.nch);
                }
                apply(p, &mut live, &mut set);
            }
        }
        let mut seen: HashSet<u32> = HashSet::new();
        live.retain(|id| set.contains(id) && seen.insert(*id));
        for &id in &live {
            for c in 0..*nchs.get(&id).unwrap_or(&1) as u64 {
                let (lk, _) = self.doc_keys(id, c);
                out.doc_queries.extend(self.doc.candidates(&lk));
            }
        }
        out.ids = live;
        out
    }

    /// Bytes the owner uploads for an operation (buckets + refreshed stashes).
    pub fn upload_bytes(&self, r: &MkseOp) -> usize {
        let vb = self.cfg.val.bucket_bytes();
        let db = self.cfg.doc.bucket_bytes();
        r.val_accesses() * vb
            + r.doc_accesses() * db
            + r.stash_refreshes
                * (self.cfg.val.stash_bytes() + self.pend_slot_bytes() * self.cfg.pend_sigma + self.vol.prm.stash_bytes())
            + if r.doc.is_empty() { 0 } else { self.cfg.doc.stash_bytes() }
    }

    pub fn pend_slot_bytes(&self) -> usize {
        crate::uokvs::NONCE + self.cfg.posting_bytes + crate::uokvs::META + crate::uokvs::CHECK
    }

    /// Bytes a client downloads in a search besides PIR answers: the VolMap
    /// stash, the ValMap stash (overflow chunks + pending slots), the DocMap stash.
    pub fn search_stash_bytes(&self) -> usize {
        self.vol.prm.stash_bytes()
            + self.cfg.val.stash_bytes()
            + self.pend_slot_bytes() * self.cfg.pend_sigma
            + self.cfg.doc.stash_bytes()
    }

    pub fn pending_len(&self, w: &str) -> usize {
        self.pending.get(w).map_or(0, |v| v.len())
    }
    pub fn n_keywords_with_pending(&self) -> usize {
        self.pending.values().filter(|v| !v.is_empty()).count()
    }
}
