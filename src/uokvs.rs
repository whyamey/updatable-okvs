//! Updatable oblivious key-value store (UOKVS): Cuckoo hashing with invalidation.
//!
//! This module implements the construction of the paper (Figures 5-7 and
//! Algorithm 1) together with the variants needed for ablations:
//!
//! * `Discipline::Paper`: the invalidation discipline of Lemma `lem:fresh`.
//!   States `(key, j)` are tracked by index, a candidate position is evaluated
//!   only when the walk visits it, every state is used at most once (on visit
//!   for random walk, on pop for BFS), BFS accesses every popped position, and
//!   every operation makes exactly `alpha` (RW) or `alpha + 1` (BFS) accesses.
//! * `Discipline::Legacy`: what the original `dip.rs` does (first-fit over all
//!   candidates, no invalidation for RW, invalidation of moved states only for
//!   BFS, BFS accesses only the eviction chain).
//! * `Discipline::NoInvalidation`: the Paper access pattern without invalidation.
//!
//! Buckets hold up to `p` slots. `p = 1` is exactly the model of the stash
//! analysis; `p > 1` is a bucketized table. Every slot is encrypted under the
//! decryption key of its item (ChaCha20 with a 16-byte zero check), so the
//! server-visible array can be materialized and fed to the PIR backend.

use chacha20::cipher::{KeyIvInit, StreamCipher};
use chacha20::ChaCha20;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};

pub type K32 = [u8; 32];

pub const NONCE: usize = 12;
/// value length (u16) + version (u32) + flags (u8)
pub const META: usize = 7;
/// trailing zero check used by the trial-decryption in Figure `cons:flex basic`
pub const CHECK: usize = 16;
const NOCUR: u8 = u8::MAX;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evict {
    Rw,
    Bfs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Discipline {
    Paper,
    Legacy,
    NoInvalidation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateMode {
    /// Write the new version through a fresh state (Figure 6, default).
    Invalidation,
    /// Overwrite in place (leaks the update pattern).
    InPlace,
}

#[derive(Clone, Debug)]
pub struct Params {
    /// number of buckets (0 = stash-only map)
    pub mu: usize,
    /// slots per bucket
    pub p: usize,
    /// candidate positions per key (<= 8)
    pub s: usize,
    /// walk length (RW) or maximum number of expanded pops (BFS)
    pub alpha: usize,
    /// stash capacity (slots)
    pub sigma: usize,
    pub evict: Evict,
    pub disc: Discipline,
    /// BFS: occupants of a popped bucket that are expanded (matters for p > 1)
    pub bfs_max_branch: usize,
    /// BFS: maximum depth of the search tree
    pub bfs_max_depth: usize,
    /// plaintext bytes per slot value
    pub val_len: usize,
}

impl Params {
    pub fn accesses_per_op(&self) -> usize {
        if self.mu == 0 {
            return 0;
        }
        match self.evict {
            Evict::Rw => self.alpha,
            Evict::Bfs => self.alpha + 1,
        }
    }
    pub fn slot_bytes(&self) -> usize {
        NONCE + self.val_len + META + CHECK
    }
    pub fn bucket_bytes(&self) -> usize {
        self.p * self.slot_bytes()
    }
    pub fn stash_bytes(&self) -> usize {
        self.sigma * self.slot_bytes()
    }
}

/// Owner PRF keys for one map (Construction `cons:auth from sym`).
#[derive(Clone)]
pub struct OwnerKeys {
    kl: K32,
    kd: K32,
}

fn kh(k: &K32, dom: &[u8], msg: &[u8]) -> K32 {
    let mut h = blake3::Hasher::new_keyed(k);
    h.update(dom);
    h.update(&[0u8]);
    h.update(msg);
    *h.finalize().as_bytes()
}

impl OwnerKeys {
    pub fn random<R: Rng>(rng: &mut R) -> Self {
        let mut kl = [0u8; 32];
        let mut kd = [0u8; 32];
        rng.fill(&mut kl);
        rng.fill(&mut kd);
        Self { kl, kd }
    }
    /// Delegation key of a label (`Cap`).
    pub fn cap(&self, label: &[u8]) -> (K32, K32) {
        (kh(&self.kl, b"cap", label), kh(&self.kd, b"cap", label))
    }
}

/// `Derive`: flat keys use the delegation key itself; hierarchical keys
/// `(g, i)` apply the second-level PRF.
pub fn derive(cap: &(K32, K32), idx: Option<u64>) -> (K32, K32) {
    match idx {
        None => *cap,
        Some(i) => (
            kh(&cap.0, b"sub", &i.to_le_bytes()),
            kh(&cap.1, b"sub", &i.to_le_bytes()),
        ),
    }
}

/// A map key: a label and, for hierarchical maps, an index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub label: Vec<u8>,
    pub idx: Option<u64>,
}

impl Key {
    pub fn flat(label: &[u8]) -> Self {
        Self { label: label.to_vec(), idx: None }
    }
    pub fn sub(label: &[u8], i: u64) -> Self {
        Self { label: label.to_vec(), idx: Some(i) }
    }
    pub fn bytes(&self) -> Vec<u8> {
        let mut v = self.label.clone();
        match self.idx {
            None => v.push(0),
            Some(i) => {
                v.push(1);
                v.extend_from_slice(&i.to_le_bytes());
            }
        }
        v
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    Bucket(u32, u16),
    Stash,
    Nowhere,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Append,
    At(u16),
}

#[derive(Clone)]
struct Item {
    key: Vec<u8>,
    lk: K32,
    dk: K32,
    val: Vec<u8>,
    ver: u32,
    tomb: bool,
    /// index of the candidate the item currently occupies
    cur: u8,
    /// bitmask of used states (key, j)
    used: u8,
    stale: bool,
    alive: bool,
    at: At,
    /// stash slot in `0..sigma` while the item is in the stash (`PutStash`
    /// picks a uniformly random dummy slot; the item keeps it until it leaves)
    slot: usize,
}

/// What the server sees for one operation, plus owner-side counters.
#[derive(Clone, Debug, Default)]
pub struct OpOut {
    /// bucket positions accessed (and re-encrypted), in order
    pub accessed: Vec<usize>,
    pub to_stash: bool,
    pub overflow: bool,
    pub moves: usize,
    /// candidate positions evaluated by the owner
    pub evaluated: usize,
    /// whether the op was a dummy (padding only)
    pub dummy: bool,
}

pub struct Table {
    pub prm: Params,
    kdip: K32,
    buckets: Vec<Vec<u32>>,
    stash: Vec<u32>,
    items: Vec<Item>,
    free: Vec<u32>,
    index: HashMap<Vec<u8>, u32>,
    /// occupied stash slots (owner state)
    stash_occ: HashSet<usize>,
    /// insertions that found no dummy stash slot (each one halts the structure)
    pub overflow: usize,
    /// set by the first overflow: the structure has output a public failure
    /// and halts (the encrypted stash never grows beyond sigma)
    pub halted: bool,
    pub stash_peak: usize,
    pub stale_count: usize,
    /// set by `encode_joint` (initial entries outside Lemma `lem:fresh`)
    pub joint_built: bool,
}

fn rand_other<R: Rng>(rng: &mut R, s: usize, not: u8) -> u8 {
    if (not as usize) >= s || s == 1 {
        return rng.gen_range(0..s) as u8;
    }
    let r = rng.gen_range(0..s - 1) as u8;
    if r >= not {
        r + 1
    } else {
        r
    }
}

impl Table {
    pub fn new<R: Rng>(prm: Params, rng: &mut R) -> Self {
        assert!(prm.s >= 1 && prm.s <= 8, "s must be in 1..=8");
        let mut kdip = [0u8; 32];
        rng.fill(&mut kdip);
        let buckets = (0..prm.mu).map(|_| Vec::with_capacity(prm.p)).collect();
        Self {
            prm,
            kdip,
            buckets,
            stash: Vec::new(),
            items: Vec::new(),
            free: Vec::new(),
            index: HashMap::new(),
            stash_occ: HashSet::new(),
            overflow: 0,
            halted: false,
            stash_peak: 0,
            stale_count: 0,
            joint_built: false,
        }
    }

    // ------------------------------------------------------------ basics

    /// `H(key, j)`: evaluates only the j-th candidate (keyed BLAKE3 with the
    /// public DIP key, applied to the pseudorandom lookup key).
    pub fn hpos(&self, lk: &K32, j: u8) -> usize {
        let mut h = blake3::Hasher::new_keyed(&self.kdip);
        h.update(lk);
        h.update(&[j]);
        let out = h.finalize();
        let v = u64::from_le_bytes(out.as_bytes()[..8].try_into().unwrap());
        (v % self.prm.mu as u64) as usize
    }

    /// Candidate positions a client queries for a lookup key.
    pub fn candidates(&self, lk: &K32) -> Vec<usize> {
        if self.prm.mu == 0 {
            return vec![];
        }
        (0..self.prm.s as u8).map(|j| self.hpos(lk, j)).collect()
    }

    fn alloc(&mut self, it: Item) -> u32 {
        if let Some(h) = self.free.pop() {
            self.items[h as usize] = it;
            h
        } else {
            self.items.push(it);
            (self.items.len() - 1) as u32
        }
    }

    fn dealloc(&mut self, h: u32) {
        let it = &mut self.items[h as usize];
        it.alive = false;
        it.val = Vec::new();
        it.at = At::Nowhere;
        self.free.push(h);
    }

    fn peek_free(&self, pos: usize) -> Option<Slot> {
        let b = &self.buckets[pos];
        if b.len() < self.prm.p {
            return Some(Slot::Append);
        }
        b.iter()
            .position(|&h| self.items[h as usize].stale)
            .map(|i| Slot::At(i as u16))
    }

    /// Frees a stale item occupying `slot` before it is overwritten.
    fn claim(&mut self, pos: usize, slot: Slot) {
        if let Slot::At(i) = slot {
            let old = self.buckets[pos][i as usize];
            if self.items[old as usize].stale && self.items[old as usize].alive {
                self.stale_count -= 1;
                self.dealloc(old);
            }
        }
    }

    fn place(&mut self, h: u32, pos: usize, slot: Slot, j: u8) {
        let s = match slot {
            Slot::Append => {
                self.buckets[pos].push(h);
                (self.buckets[pos].len() - 1) as u16
            }
            Slot::At(i) => {
                self.buckets[pos][i as usize] = h;
                i
            }
        };
        let it = &mut self.items[h as usize];
        it.at = At::Bucket(pos as u32, s);
        it.cur = j;
    }

    /// `PutStash`: the item takes a uniformly random dummy slot of the
    /// sigma-slot stash. If none is left (an overflow), the operation outputs
    /// a public failure and the structure halts: `halted` is set, the
    /// overflow is counted, and `encrypt_stash` refuses to run. The item is
    /// still recorded (without a slot) so that the owner state stays
    /// consistent for diagnostics; callers must stop using the structure.
    fn put_stash<R: Rng>(&mut self, h: u32, rng: &mut R, out: &mut OpOut) {
        out.to_stash = true;
        let slot = if self.stash.len() >= self.prm.sigma {
            out.overflow = true;
            self.overflow += 1;
            self.halted = true;
            usize::MAX
        } else {
            // rejection sampling: a uniformly random slot among the dummies
            loop {
                let r = rng.gen_range(0..self.prm.sigma);
                if !self.stash_occ.contains(&r) {
                    break r;
                }
            }
        };
        if slot != usize::MAX {
            self.stash_occ.insert(slot);
        }
        self.stash.push(h);
        let it = &mut self.items[h as usize];
        it.at = At::Stash;
        it.cur = NOCUR;
        it.slot = slot;
        self.stash_peak = self.stash_peak.max(self.stash.len());
    }

    fn remove_from_stash(&mut self, h: u32) {
        if let Some(i) = self.stash.iter().position(|&x| x == h) {
            self.stash.swap_remove(i);
            let sl = self.items[h as usize].slot;
            self.stash_occ.remove(&sl);
        }
    }

    fn pad<R: Rng>(&self, rng: &mut R, out: &mut OpOut) {
        out.accessed.push(rng.gen_range(0..self.prm.mu));
    }

    fn finish_padding<R: Rng>(&self, rng: &mut R, out: &mut OpOut) {
        while out.accessed.len() < self.prm.accesses_per_op() {
            self.pad(rng, out);
        }
    }

    // ------------------------------------------------------------ public API

    pub fn len(&self) -> usize {
        self.index.len()
    }
    pub fn stash_len(&self) -> usize {
        self.stash.len()
    }
    pub fn stash_capacity(&self) -> usize {
        self.prm.sigma
    }
    /// fraction of bucket slots holding live (non-stale) items
    pub fn load(&self) -> f64 {
        if self.prm.mu == 0 {
            return 0.0;
        }
        let in_buckets = self.index.len() - self.stash.len();
        in_buckets as f64 / (self.prm.mu * self.prm.p) as f64
    }
    pub fn contains(&self, key: &Key) -> bool {
        self.index.contains_key(&key.bytes())
    }

    fn new_item(&self, key: &Key, lk: K32, dk: K32, val: Vec<u8>) -> Item {
        assert!(val.len() <= self.prm.val_len, "value longer than val_len");
        Item {
            key: key.bytes(),
            lk,
            dk,
            val,
            ver: 0,
            tomb: false,
            cur: NOCUR,
            used: 0,
            stale: false,
            alive: true,
            at: At::Nowhere,
            slot: usize::MAX,
        }
    }

    /// `Encode`: places entries one at a time with the same walk, without
    /// padding or messages, recording used states (rule 4 of the discipline).
    pub fn encode<R: Rng>(&mut self, entries: Vec<(Key, (K32, K32), Vec<u8>)>, rng: &mut R) {
        for (key, (lk, dk), val) in entries {
            let mut out = OpOut::default();
            self.insert_inner(&key, lk, dk, val, false, rng, &mut out);
        }
    }

    /// Joint build used for the "jointly prepared" curves of the paper: a
    /// first-fit random walk with walk length `alpha_build` and no
    /// invalidation. Later operations use `self.prm`. Under the discipline,
    /// every candidate a joint build evaluates would have to be marked used;
    /// this function does not, so it is outside Lemma `lem:fresh`.
    pub fn encode_joint<R: Rng>(
        &mut self,
        entries: Vec<(Key, (K32, K32), Vec<u8>)>,
        alpha_build: usize,
        rng: &mut R,
    ) {
        let saved = self.prm.clone();
        self.prm.alpha = alpha_build;
        self.prm.disc = Discipline::Legacy;
        self.prm.evict = Evict::Rw;
        for (key, (lk, dk), val) in entries {
            let mut out = OpOut::default();
            self.insert_inner(&key, lk, dk, val, false, rng, &mut out);
        }
        for it in self.items.iter_mut() {
            it.used = 0;
        }
        self.prm = saved;
        self.joint_built = true;
    }

    pub fn insert<R: Rng>(
        &mut self,
        key: &Key,
        lk: K32,
        dk: K32,
        val: Vec<u8>,
        rng: &mut R,
    ) -> OpOut {
        let mut out = OpOut::default();
        self.insert_inner(key, lk, dk, val, true, rng, &mut out);
        out
    }

    fn insert_inner<R: Rng>(
        &mut self,
        key: &Key,
        lk: K32,
        dk: K32,
        val: Vec<u8>,
        record: bool,
        rng: &mut R,
        out: &mut OpOut,
    ) {
        let kb = key.bytes();
        assert!(!self.index.contains_key(&kb), "insert of an existing key");
        let it = self.new_item(key, lk, dk, val);
        let h = self.alloc(it);
        self.index.insert(kb, h);
        self.place_new(h, record, rng, out);
    }

    /// `Update`; a deletion is an update to a tombstone.
    pub fn update<R: Rng>(
        &mut self,
        key: &Key,
        val: Vec<u8>,
        tomb: bool,
        mode: UpdateMode,
        rng: &mut R,
    ) -> OpOut {
        let kb = key.bytes();
        let h = *self.index.get(&kb).expect("update of a missing key");
        let mut out = OpOut::default();
        assert!(val.len() <= self.prm.val_len);
        if self.prm.mu == 0 {
            let it = &mut self.items[h as usize];
            it.val = val;
            it.ver += 1;
            it.tomb = tomb;
            return out;
        }
        match mode {
            UpdateMode::InPlace => {
                let it = &mut self.items[h as usize];
                it.val = val;
                it.ver += 1;
                it.tomb = tomb;
                let at = it.at;
                if let At::Bucket(pos, _) = at {
                    out.accessed.push(pos as usize);
                }
                self.finish_padding(rng, &mut out);
                // move the real access to a uniformly random index
                if !out.accessed.is_empty() {
                    let k = rng.gen_range(0..out.accessed.len());
                    out.accessed.swap(0, k);
                }
            }
            UpdateMode::Invalidation => {
                let old = self.items[h as usize].clone();
                match old.at {
                    At::Bucket(..) => {
                        self.items[h as usize].stale = true;
                        self.stale_count += 1;
                    }
                    At::Stash => {
                        self.remove_from_stash(h);
                        self.dealloc(h);
                    }
                    At::Nowhere => self.dealloc(h),
                }
                let mut it = old;
                it.val = val;
                it.ver += 1;
                it.tomb = tomb;
                it.cur = NOCUR;
                it.stale = false;
                it.alive = true;
                it.at = At::Nowhere;
                let n = self.alloc(it);
                self.index.insert(kb, n);
                self.place_new(n, true, rng, &mut out);
            }
        }
        out
    }

    /// Frees a key without any server-visible access: a bucket copy becomes
    /// stale (and counts as free for later walks), a stash copy is dropped
    /// at the next stash refresh. Used when a versioned key supersedes it.
    pub fn remove(&mut self, key: &Key) -> bool {
        let kb = key.bytes();
        let Some(h) = self.index.remove(&kb) else { return false };
        match self.items[h as usize].at {
            At::Bucket(..) => {
                self.items[h as usize].stale = true;
                self.stale_count += 1;
            }
            At::Stash => {
                self.remove_from_stash(h);
                self.dealloc(h);
            }
            At::Nowhere => self.dealloc(h),
        }
        true
    }

    /// Value of a key from the owner state (for rewriting a chunk).
    pub fn owner_value(&self, key: &Key) -> Option<&[u8]> {
        self.index.get(&key.bytes()).map(|&h| self.items[h as usize].val.as_slice())
    }

    /// Whether a key currently sits in the stash (owner state).
    pub fn in_stash(&self, key: &Key) -> bool {
        self.index
            .get(&key.bytes())
            .map_or(false, |&h| self.items[h as usize].at == At::Stash)
    }

    /// A dummy operation: uniformly random re-encryptions only.
    pub fn dummy<R: Rng>(&mut self, rng: &mut R) -> OpOut {
        let mut out = OpOut { dummy: true, ..Default::default() };
        self.finish_padding(rng, &mut out);
        out
    }

    fn place_new<R: Rng>(&mut self, h: u32, record: bool, rng: &mut R, out: &mut OpOut) {
        if self.prm.mu == 0 {
            self.put_stash(h, rng, out);
            return;
        }
        match (self.prm.evict, self.prm.disc) {
            (Evict::Rw, Discipline::Legacy) => self.rw_legacy(h, record, rng, out),
            (Evict::Rw, _) => {
                let j = rng.gen_range(0..self.prm.s) as u8;
                self.rw_paper(h, j, record, rng, out)
            }
            (Evict::Bfs, d) => self.bfs(h, d, record, rng, out),
        }
        if record {
            self.finish_padding(rng, out);
        }
    }

    // ------------------------------------------------------------ random walk

    /// Figure 7: one access per step; invalidated states go to the stash
    /// without being evaluated.
    fn rw_paper<R: Rng>(&mut self, mut h: u32, mut j: u8, record: bool, rng: &mut R, out: &mut OpOut) {
        let a = self.prm.alpha;
        let inval = self.prm.disc == Discipline::Paper;
        for i in 0..a {
            if inval && self.items[h as usize].used & (1 << j) != 0 {
                self.put_stash(h, rng, out);
                return;
            }
            self.items[h as usize].used |= 1 << j;
            let lk = self.items[h as usize].lk;
            let pos = self.hpos(&lk, j);
            out.evaluated += 1;
            if record {
                out.accessed.push(pos);
            }
            if let Some(sl) = self.peek_free(pos) {
                self.claim(pos, sl);
                self.place(h, pos, sl, j);
                return;
            }
            let sl = rng.gen_range(0..self.buckets[pos].len());
            let victim = self.buckets[pos][sl];
            self.place(h, pos, Slot::At(sl as u16), j);
            out.moves += 1;
            self.items[victim as usize].at = At::Nowhere;
            if i == a - 1 {
                self.put_stash(victim, rng, out);
                return;
            }
            let vc = self.items[victim as usize].cur;
            j = rand_other(rng, self.prm.s, vc);
            h = victim;
        }
    }

    /// The original code: check all candidates first-fit, otherwise evict at a
    /// uniformly random candidate; no invalidation.
    fn rw_legacy<R: Rng>(&mut self, mut h: u32, record: bool, rng: &mut R, out: &mut OpOut) {
        let s = self.prm.s as u8;
        for _ in 0..self.prm.alpha {
            let lk = self.items[h as usize].lk;
            for j in 0..s {
                let pos = self.hpos(&lk, j);
                out.evaluated += 1;
                if let Some(sl) = self.peek_free(pos) {
                    self.claim(pos, sl);
                    self.place(h, pos, sl, j);
                    if record {
                        out.accessed.push(pos);
                    }
                    return;
                }
            }
            let j = rng.gen_range(0..s);
            let pos = self.hpos(&lk, j);
            if record {
                out.accessed.push(pos);
            }
            let sl = rng.gen_range(0..self.buckets[pos].len());
            let victim = self.buckets[pos][sl];
            self.place(h, pos, Slot::At(sl as u16), j);
            out.moves += 1;
            self.items[victim as usize].at = At::Nowhere;
            h = victim;
        }
        self.put_stash(h, rng, out);
    }

    // ------------------------------------------------------------ BFS

    /// Algorithm 1. `Paper`: pop invalidation, positions evaluated at pop,
    /// every popped position accessed. `Legacy`: first-fit, moved-state
    /// invalidation, only the chain accessed. `NoInvalidation`: Paper access
    /// pattern without invalidation.
    fn bfs<R: Rng>(&mut self, root: u32, disc: Discipline, record: bool, rng: &mut R, out: &mut OpOut) {
        struct Node {
            item: u32,
            j: u8,
            parent: u32,
            depth: u8,
            pos: usize,
        }
        let s = self.prm.s as u8;
        let legacy = disc == Discipline::Legacy;
        let check_used = disc != Discipline::NoInvalidation;

        if legacy {
            let lk = self.items[root as usize].lk;
            for j in 0..s {
                if self.items[root as usize].used & (1 << j) != 0 {
                    continue;
                }
                let pos = self.hpos(&lk, j);
                out.evaluated += 1;
                if let Some(sl) = self.peek_free(pos) {
                    self.claim(pos, sl);
                    self.place(root, pos, sl, j);
                    self.items[root as usize].used |= 1 << j;
                    if record {
                        out.accessed.push(pos);
                    }
                    return;
                }
            }
        }

        let mut nodes: Vec<Node> = Vec::new();
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut order: Vec<u8> = (0..s).collect();
        order.shuffle(rng);
        for j in order {
            nodes.push(Node { item: root, j, parent: NONE, depth: 0, pos: 0 });
            q.push_back(nodes.len() - 1);
        }
        let mut expanded: HashSet<u32> = HashSet::new();
        expanded.insert(root);
        let mut seen_buckets: HashSet<usize> = HashSet::new();
        let mut steps = 0usize;
        let mut found: Option<usize> = None;
        let mut popped: Vec<usize> = Vec::new();

        while let Some(ni) = q.pop_front() {
            if steps >= self.prm.alpha {
                break;
            }
            let (it, j) = (nodes[ni].item, nodes[ni].j);
            if check_used && self.items[it as usize].used & (1 << j) != 0 {
                continue; // used state: neither evaluated nor accessed
            }
            if !legacy && check_used {
                self.items[it as usize].used |= 1 << j; // pop invalidation
            }
            let lk = self.items[it as usize].lk;
            let pos = self.hpos(&lk, j);
            out.evaluated += 1;
            nodes[ni].pos = pos;
            if !legacy {
                popped.push(pos);
            }
            if self.peek_free(pos).is_some() {
                found = Some(ni);
                break;
            }
            if legacy && !seen_buckets.insert(pos) {
                steps += 1;
                continue;
            }
            if (nodes[ni].depth as usize) < self.prm.bfs_max_depth {
                let b = self.buckets[pos].clone();
                let mut idxs: Vec<usize> = (0..b.len()).collect();
                idxs.shuffle(rng);
                let mut br = 0;
                for k in idxs {
                    if br >= self.prm.bfs_max_branch {
                        break;
                    }
                    let occ = b[k];
                    if self.items[occ as usize].stale || !expanded.insert(occ) {
                        continue;
                    }
                    br += 1;
                    let oc = self.items[occ as usize].cur;
                    let mut js: Vec<u8> = (0..s).filter(|&x| x != oc).collect();
                    js.shuffle(rng);
                    for jj in js {
                        nodes.push(Node {
                            item: occ,
                            j: jj,
                            parent: ni as u32,
                            depth: nodes[ni].depth + 1,
                            pos: 0,
                        });
                        q.push_back(nodes.len() - 1);
                    }
                }
            }
            steps += 1;
        }

        if record && !legacy {
            out.accessed.extend_from_slice(&popped);
        }

        match found {
            None => self.put_stash(root, rng, out),
            Some(m) => {
                let mut tpos = nodes[m].pos;
                let mut tslot = self.peek_free(tpos).unwrap();
                self.claim(tpos, tslot);
                let mut cur = m;
                loop {
                    let it = nodes[cur].item;
                    let j = nodes[cur].j;
                    let parent = nodes[cur].parent;
                    let old_at = self.items[it as usize].at;
                    self.place(it, tpos, tslot, j);
                    out.moves += 1;
                    if legacy {
                        self.items[it as usize].used |= 1 << j;
                        if record {
                            out.accessed.push(tpos);
                        }
                    }
                    if parent == NONE {
                        break;
                    }
                    match old_at {
                        At::Bucket(bp, bs) => {
                            debug_assert_eq!(bp as usize, nodes[parent as usize].pos);
                            tpos = bp as usize;
                            tslot = Slot::At(bs);
                        }
                        _ => unreachable!("chain item not in a bucket"),
                    }
                    cur = parent as usize;
                }
            }
        }
    }

    // ------------------------------------------------------------ client view

    /// Client-side lookup on plaintext state: scans the s candidates and the
    /// stash and keeps the highest version (stale copies have lower versions).
    /// Returns (value, tombstone) of the newest copy.
    pub fn lookup(&self, key: &Key, lk: &K32) -> Option<(Vec<u8>, bool)> {
        let kb = key.bytes();
        let mut best: Option<&Item> = None;
        let mut consider = |h: u32| {
            let it = &self.items[h as usize];
            if it.alive && it.key == kb && best.map_or(true, |b| it.ver > b.ver) {
                best = Some(it);
            }
        };
        for pos in self.candidates(lk) {
            for &h in &self.buckets[pos] {
                consider(h);
            }
        }
        for &h in &self.stash {
            consider(h);
        }
        best.map(|it| (it.val.clone(), it.tomb))
    }

    /// Stash contents visible to a client holding `dk` (plaintext simulation).
    pub fn stash_items(&self) -> impl Iterator<Item = (&[u8], &[u8], bool)> {
        self.stash.iter().map(move |&h| {
            let it = &self.items[h as usize];
            (it.key.as_slice(), it.val.as_slice(), it.tomb)
        })
    }

    // ------------------------------------------------------------ encryption

    fn enc_slot<R: Rng>(&self, it: &Item, rng: &mut R, out: &mut Vec<u8>) {
        let mut nonce = [0u8; NONCE];
        rng.fill(&mut nonce);
        let mut pt = vec![0u8; self.prm.val_len + META + CHECK];
        pt[..it.val.len()].copy_from_slice(&it.val);
        let o = self.prm.val_len;
        pt[o..o + 2].copy_from_slice(&(it.val.len() as u16).to_le_bytes());
        pt[o + 2..o + 6].copy_from_slice(&it.ver.to_le_bytes());
        pt[o + 6] = it.tomb as u8;
        let mut c = ChaCha20::new(&it.dk.into(), &nonce.into());
        c.apply_keystream(&mut pt);
        out.extend_from_slice(&nonce);
        out.extend_from_slice(&pt);
    }

    /// Fresh ciphertext of a bucket (what `Reencrypt` sends).
    pub fn encrypt_bucket<R: Rng>(&self, pos: usize, rng: &mut R) -> Vec<u8> {
        let sb = self.prm.slot_bytes();
        let mut out = Vec::with_capacity(self.prm.bucket_bytes());
        let b = &self.buckets[pos];
        for k in 0..self.prm.p {
            if k < b.len() {
                self.enc_slot(&self.items[b[k] as usize], rng, &mut out);
            } else {
                let start = out.len();
                out.resize(start + sb, 0);
                rng.fill(&mut out[start..]);
            }
        }
        out
    }

    /// Owner view of the sigma stash slots: the item handle in each slot
    /// (`None` = dummy). With buckets (mu > 0) every item keeps the slot
    /// `PutStash` gave it; a stash-only map (mu = 0) is stored in a fresh
    /// uniformly random order at every refresh.
    fn stash_layout<R: Rng>(&self, rng: &mut R) -> Vec<Option<u32>> {
        assert!(
            !self.halted && self.stash.len() <= self.prm.sigma,
            "stash overflow: more than sigma = {} items; the structure has halted (public failure)",
            self.prm.sigma
        );
        let mut slots: Vec<Option<u32>> = vec![None; self.prm.sigma];
        if self.prm.mu == 0 {
            let pos = rand::seq::index::sample(rng, self.prm.sigma, self.stash.len());
            for (k, p) in pos.into_iter().enumerate() {
                slots[p] = Some(self.stash[k]);
            }
        } else {
            for &h in &self.stash {
                let sl = self.items[h as usize].slot;
                debug_assert!(slots[sl].is_none());
                slots[sl] = Some(h);
            }
        }
        slots
    }

    /// Slot index of each stash item, in slot order (tests and diagnostics).
    pub fn stash_slots<R: Rng>(&self, rng: &mut R) -> Vec<Option<Vec<u8>>> {
        self.stash_layout(rng).into_iter().map(|x| x.map(|h| self.items[h as usize].key.clone())).collect()
    }

    /// Fresh ciphertext of the whole stash: exactly sigma slots (items in
    /// their slots, dummies elsewhere). Panics if the structure has halted.
    pub fn encrypt_stash<R: Rng>(&self, rng: &mut R) -> Vec<u8> {
        let sb = self.prm.slot_bytes();
        let layout = self.stash_layout(rng);
        let mut out = Vec::with_capacity(layout.len() * sb);
        for x in layout {
            match x {
                Some(h) => self.enc_slot(&self.items[h as usize], rng, &mut out),
                None => {
                    let start = out.len();
                    out.resize(start + sb, 0);
                    rng.fill(&mut out[start..]);
                }
            }
        }
        out
    }

    /// Trial decryption of one slot (client side).
    pub fn decrypt_slot(&self, slot: &[u8], dk: &K32) -> Option<(Vec<u8>, u32, bool)> {
        let (nonce, ct) = slot.split_at(NONCE);
        let mut pt = ct.to_vec();
        let n: [u8; NONCE] = nonce.try_into().unwrap();
        let mut c = ChaCha20::new(&(*dk).into(), &n.into());
        c.apply_keystream(&mut pt);
        let o = self.prm.val_len;
        if pt[o + META..].iter().any(|&b| b != 0) {
            return None;
        }
        let len = u16::from_le_bytes([pt[o], pt[o + 1]]) as usize;
        let ver = u32::from_le_bytes(pt[o + 2..o + 6].try_into().unwrap());
        Some((pt[..len].to_vec(), ver, pt[o + 6] != 0))
    }

    /// Bytes of an operation sent by the owner: accessed buckets + stash.
    pub fn op_upload_bytes(&self, out: &OpOut) -> usize {
        out.accessed.len() * self.prm.bucket_bytes() + self.prm.stash_bytes()
    }

    /// Initial server array (all buckets encrypted).
    pub fn encrypt_all<R: Rng>(&self, rng: &mut R) -> Vec<Vec<u8>> {
        (0..self.prm.mu).map(|pos| self.encrypt_bucket(pos, rng)).collect()
    }

    /// Consistency check of the owner state (tests).
    pub fn check_invariants(&self) {
        for (pos, b) in self.buckets.iter().enumerate() {
            assert!(b.len() <= self.prm.p);
            for (k, &h) in b.iter().enumerate() {
                let it = &self.items[h as usize];
                assert!(it.alive, "dead item in bucket");
                assert_eq!(it.at, At::Bucket(pos as u32, k as u16));
                if !it.stale {
                    assert_eq!(self.hpos(&it.lk, it.cur), pos, "item not at its candidate");
                    if self.prm.disc == Discipline::Paper && !self.joint_built {
                        assert!(it.used & (1 << it.cur) != 0, "occupied state not marked used");
                    }
                }
            }
        }
        for (_, &h) in self.index.iter() {
            let it = &self.items[h as usize];
            assert!(it.alive && !it.stale);
            assert!(matches!(it.at, At::Bucket(..) | At::Stash));
        }
        if !self.halted {
            assert!(self.stash.len() <= self.prm.sigma, "stash beyond sigma without a halt");
            let slots: HashSet<usize> = self.stash.iter().map(|&h| self.items[h as usize].slot).collect();
            assert_eq!(slots.len(), self.stash.len(), "two stash items share a slot");
            assert!(slots.iter().all(|&x| x < self.prm.sigma));
            assert_eq!(slots, self.stash_occ, "stash slot bookkeeping");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn prm(mu: usize, sigma: usize) -> Params {
        Params { mu, p: 1, s: 4, alpha: 2, sigma, evict: Evict::Bfs, disc: Discipline::Paper, bfs_max_branch: 2, bfs_max_depth: 64, val_len: 8 }
    }
    fn ins(t: &mut Table, i: u64, rng: &mut StdRng) -> OpOut {
        let mut lk = [0u8; 32];
        rng.fill(&mut lk);
        t.insert(&Key::sub(b"k", i), lk, [1u8; 32], vec![i as u8], rng)
    }

    #[test]
    fn stash_items_keep_random_slots() {
        let mut rng = StdRng::seed_from_u64(1);
        // a tiny table so that insertions reach the stash
        let mut t = Table::new(prm(6, 24), &mut rng);
        let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
        let mut first_slots = HashSet::new();
        for i in 0..20 {
            let o = ins(&mut t, i, &mut rng);
            assert!(!o.overflow);
            t.check_invariants();
            let slots = t.stash_slots(&mut rng);
            assert_eq!(slots.len(), 24);
            for (p, k) in slots.iter().enumerate() {
                if let Some(k) = k {
                    let e = *seen.entry(k.clone()).or_insert(p);
                    assert_eq!(e, p, "a stash item changed its slot");
                }
            }
            assert_eq!(t.encrypt_stash(&mut rng).len(), 24 * t.prm.slot_bytes());
        }
        for (_, &p) in &seen {
            first_slots.insert(p);
        }
        assert!(t.stash_len() >= 8);
        // not simply the first slots
        assert!(first_slots.iter().any(|&p| p >= t.stash_len()));
    }

    #[test]
    fn stash_only_map_is_reshuffled() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut t = Table::new(prm(0, 16), &mut rng);
        for i in 0..3 {
            ins(&mut t, i, &mut rng);
        }
        let mut layouts = HashSet::new();
        for _ in 0..40 {
            let l: Vec<bool> = t.stash_slots(&mut rng).iter().map(|x| x.is_some()).collect();
            assert_eq!(l.iter().filter(|x| **x).count(), 3);
            layouts.insert(l);
        }
        assert!(layouts.len() > 10, "a stash-only map must be stored in a fresh random order");
    }

    #[test]
    fn overflow_halts() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut t = Table::new(prm(0, 3), &mut rng);
        for i in 0..3 {
            assert!(!ins(&mut t, i, &mut rng).overflow);
        }
        assert!(!t.halted);
        let o = ins(&mut t, 3, &mut rng);
        assert!(o.overflow && t.halted && t.overflow == 1);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| t.encrypt_stash(&mut rng)));
        assert!(r.is_err(), "a halted stash must not be sent (it would exceed sigma)");
    }
}
