//! Experiments for the revision. Subcommands:
//!
//!   selftest   correctness checks (all eviction variants, encryption, MKSE, PIR)
//!   stash      stash size vs load for RW/BFS x {paper, legacy, noinv} (E5)
//!   ops        insertions / deletions / updates on the MKSE (E1, E3)
//!   pir        FrodoPIR microbenchmarks: setup, hint, row updates, batches (E2c)
//!   design     the full dynamic design (log + batched versioned flushes +
//!              row-addressed DocMap) and its baselines (per-op, rebuild)
//!
//! Run `uokvs_bench <cmd> --help` for options. Outputs are CSV files.

use mkse::mkse_dyn::{Mkse, MkseCfg, MkseOp, OpKind};
use mkse::mkse_log::{median, pir_name, Action, DocLayout, LogCfg, MkseLog, Policy, Profile, PIR_COMBOS};
use mkse::pir_backend::{random_rows, FrodoPir, Pir, SpiralPir};
use mkse::uokvs::{derive, Discipline, Evict, Key, OwnerKeys, Params, Table, UpdateMode};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// ------------------------------------------------------------------ args

struct Args(HashMap<String, String>);
impl Args {
    fn parse(v: &[String]) -> Self {
        let mut m = HashMap::new();
        let mut i = 0;
        while i < v.len() {
            if let Some(k) = v[i].strip_prefix("--") {
                if i + 1 < v.len() && !v[i + 1].starts_with("--") {
                    m.insert(k.to_string(), v[i + 1].clone());
                    i += 2;
                } else {
                    m.insert(k.to_string(), "true".into());
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        Args(m)
    }
    fn s(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn u(&self, k: &str, d: usize) -> usize {
        self.0.get(k).map(|x| parse_num(x) as usize).unwrap_or(d)
    }
    fn f(&self, k: &str, d: f64) -> f64 {
        self.0.get(k).map(|x| x.parse().unwrap()).unwrap_or(d)
    }
    fn b(&self, k: &str) -> bool {
        self.0.contains_key(k)
    }
    fn list(&self, k: &str, d: &str) -> Vec<usize> {
        self.s(k, d).split(',').filter(|x| !x.is_empty()).map(|x| parse_num(x) as usize).collect()
    }
}

fn parse_num(x: &str) -> f64 {
    x.replace('_', "").parse::<f64>().unwrap_or_else(|_| panic!("bad number {x}"))
}

fn evict_of(s: &str) -> Evict {
    match s {
        "rw" => Evict::Rw,
        "bfs" => Evict::Bfs,
        _ => panic!("--evict rw|bfs"),
    }
}
fn disc_of(s: &str) -> Discipline {
    match s {
        "paper" => Discipline::Paper,
        "legacy" => Discipline::Legacy,
        "noinv" => Discipline::NoInvalidation,
        _ => panic!("--disc paper|legacy|noinv"),
    }
}
fn mode_of(s: &str) -> UpdateMode {
    match s {
        "inval" => UpdateMode::Invalidation,
        "inplace" => UpdateMode::InPlace,
        _ => panic!("--update-mode inval|inplace"),
    }
}

fn csv(path: &str, header: &str) -> BufWriter<File> {
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let mut w = BufWriter::new(File::create(path).expect("create csv"));
    writeln!(w, "{header}").unwrap();
    w
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).cloned().unwrap_or_default();
    let a = Args::parse(&argv[2.min(argv.len())..]);
    if a.b("help") {
        println!("{}", HELP);
        return;
    }
    match cmd.as_str() {
        "selftest" => selftest(&a),
        "stash" => stash(&a),
        "churn" => churn(&a),
        "ops" => ops(&a),
        "pir" => pir(&a),
        "design" => design(&a),
        "report" => report(&a),
        _ => println!("{}", HELP),
    }
}

const HELP: &str = r#"uokvs_bench <selftest|stash|churn|ops|pir|design|report> [--options]

churn  --n 1000000 --mu-factor 1.5 --evict rw|bfs --alpha 10 --disc paper|legacy|noinv
       --ops-factor 4 --mode update|replace --zipf 0 --rebuild-every 0 --checkpoints 40
       fill to n live keys, then churn at constant load: `update` rewrites a key
       through an invalidation-mode update (same key, a fresh state), `replace`
       frees a key and inserts a new one (the versioned-key pattern of mkse_log);
       targets are uniform (zipf 0) or Zipf(s) over n logical records; a rebuild
       re-inserts every live key into a fresh table every R operations

stash  --n 1000000 --mu-factor 2.0 --p 1 --s 4 --alpha 10 --evict rw|bfs
       --disc paper|legacy|noinv --joint-frac 0.0 --alpha-build 100
       --branch 2 --depth 64 --checkpoints 50 --seed 1 --out results/stash.csv
ops    --data DIR (keyword_file_map.json, files.txt) | --synthetic 20000,100000,40
       --build-frac 0.8 --order fileid|random --max-docs 0 --del-frac 0.02 --upd-frac 0.02
       --evict bfs --disc paper --alpha 20 --update-mode inval|inplace
       --bucket-bytes 8192 --posting-bytes 40 --flush-frac 0.5 --doc-chunk 1024
       --mu-factor 2.0 --val-mu auto --doc-mu-factor 2.0 --sigma 64 --doc-sigma 256
       --pend-sigma auto --vol-sigma auto --encrypt --pir --search-every 1000
       --pend-budget none|N --search-sample 20 --pir-search 3 --no-doc-pir --seed 1 --out-prefix results/ops
pir    --rows 2^18 --elem-bytes 8192 --batches 4,8,16,64,784 --widths 8,16,32,64 --update-rows 10,21
       --trials 3 --seed 1 --out results/pir.csv
design --data DIR | --synthetic K,D,PER  --policy batched|perop|rebuild
       --b 10000 --k auto --theta C --age inf | --r 100000   --major-every 1000000
       --doc-layout rows|cuckoo|fixed --doc-row 1024 --doc-spare 0.2 --bucket-bytes 8192 --no-reuse
       --evict bfs --alpha 20 --mu-factor 2.0 --sigma 16 --clients 100,1000,10000,100000
       public rebuild sizes: --val-cap auto (ceil(0.46 mu) of the reuse-mode table;
       --val-cap-frac 0.46, keep mode uses mu at --reuse-mu-factor 2.0)
       --doc-live-cap auto (expected document chunks + edits, i.e. the DocMap rows
       without --doc-spare) --doc-repack-below 0 --ver-cap auto (version records of
       all documents x (1 + --ver-spare 0.1) + 64) --vmap-pad auto (fixed layout:
       1.2 x (upd-frac + docdel-frac) x major-every + 64)
       --build-frac 0.8 --del-frac 0.02 --upd-frac 0.02 --docdel-frac 0.002
       --pir --no-doc-pir --pir-width 32 --pir-search 2 --search-every 5000 --search-sample 10
       --max-appends 0 (replay only a prefix) --stateless --val-pir frodo|spiral --doc-pir frodo|spiral
       --max-docs-fetched 100 (capped document retrieval in the client costs)
       --label NAME --out-prefix results/design/NAME --summary results/design_summary.csv
report --dir results [--out results/REPORT.md]
selftest
"#;

// ------------------------------------------------------------------ selftest

fn small_params(evict: Evict, disc: Discipline, p: usize) -> Params {
    Params {
        mu: 400,
        p,
        s: 4,
        alpha: if evict == Evict::Rw { 10 } else { 20 },
        sigma: 10_000,
        evict,
        disc,
        bfs_max_branch: 2,
        bfs_max_depth: 64,
        val_len: 16,
    }
}

fn selftest(_a: &Args) {
    let mut rng = StdRng::seed_from_u64(7);
    // 1. table correctness for every variant, with updates and deletes
    for evict in [Evict::Rw, Evict::Bfs] {
        for disc in [Discipline::Paper, Discipline::Legacy, Discipline::NoInvalidation] {
            for p in [1usize, 4] {
                for mode in [UpdateMode::Invalidation, UpdateMode::InPlace] {
                    let prm = small_params(evict, disc, p);
                    let ok = OwnerKeys::random(&mut rng);
                    let mut t = Table::new(prm.clone(), &mut rng);
                    let n0 = (prm.mu * p) / 4;
                    let ent: Vec<_> = (0..n0)
                        .map(|i| {
                            let k = Key::flat(format!("k{i}").as_bytes());
                            let keys = derive(&ok.cap(&k.label), None);
                            (k, keys, (i as u32).to_le_bytes().to_vec())
                        })
                        .collect();
                    t.encode(ent, &mut rng);
                    let mut reference: HashMap<String, (Vec<u8>, bool)> = HashMap::new();
                    for i in 0..n0 {
                        reference.insert(format!("k{i}"), ((i as u32).to_le_bytes().to_vec(), false));
                    }
                    let total = (prm.mu * p) / 2;
                    let mut next = n0;
                    for step in 0..(total - n0) * 2 {
                        let r: f64 = rng.gen();
                        let out = if r < 0.5 && next < total {
                            let name = format!("k{next}");
                            let k = Key::flat(name.as_bytes());
                            let (lk, dk) = derive(&ok.cap(&k.label), None);
                            let v = (next as u32).to_le_bytes().to_vec();
                            reference.insert(name, (v.clone(), false));
                            next += 1;
                            t.insert(&k, lk, dk, v, &mut rng)
                        } else if r < 0.85 {
                            let i = rng.gen_range(0..next);
                            let name = format!("k{i}");
                            let k = Key::flat(name.as_bytes());
                            let v = vec![(step % 251) as u8; 3];
                            reference.insert(name, (v.clone(), false));
                            t.update(&k, v, false, mode, &mut rng)
                        } else {
                            let i = rng.gen_range(0..next);
                            let name = format!("k{i}");
                            let k = Key::flat(name.as_bytes());
                            reference.insert(name, (vec![], true));
                            t.update(&k, vec![], true, mode, &mut rng)
                        };
                        assert_eq!(out.accessed.len(), prm.accesses_per_op(), "access count must be fixed");
                    }
                    t.check_invariants();
                    for (name, (v, tomb)) in &reference {
                        let k = Key::flat(name.as_bytes());
                        let (lk, _) = derive(&ok.cap(&k.label), None);
                        let got = t.lookup(&k, &lk).expect("lookup");
                        assert_eq!(&got.0, v, "{evict:?} {disc:?} p={p} {mode:?} wrong value for {name}");
                        assert_eq!(got.1, *tomb);
                    }
                    println!(
                        "ok  table {:?}/{:?} p={} {:?}: {} keys, stash {}, load {:.2}",
                        evict, disc, p, mode, t.len(), t.stash_len(), t.load()
                    );
                }
            }
        }
    }

    // 2. encryption round trip + PIR retrieval of a bucket
    let prm = small_params(Evict::Bfs, Discipline::Paper, 2);
    let ok = OwnerKeys::random(&mut rng);
    let mut t = Table::new(prm.clone(), &mut rng);
    let k = Key::flat(b"needle");
    let (lk, dk) = derive(&ok.cap(&k.label), None);
    t.insert(&k, lk, dk, b"value!".to_vec(), &mut rng);
    for i in 0..300 {
        let kk = Key::flat(format!("x{i}").as_bytes());
        let (a, b) = derive(&ok.cap(&kk.label), None);
        t.insert(&kk, a, b, vec![1, 2, 3], &mut rng);
    }
    let rows = t.encrypt_all(&mut rng);
    let (mut pir, _) = FrodoPir::new(&rows, prm.bucket_bytes());
    let cands = t.candidates(&lk);
    let (_, got) = pir.query_batch(&cands);
    let sb = prm.slot_bytes();
    let mut found = false;
    for b in &got {
        for slot in b.chunks(sb) {
            if let Some((v, _, _)) = t.decrypt_slot(slot, &dk) {
                assert_eq!(v, b"value!");
                found = true;
            }
        }
    }
    let st = t.encrypt_stash(&mut rng);
    for slot in st.chunks(sb) {
        if let Some((v, _, _)) = t.decrypt_slot(slot, &dk) {
            assert_eq!(v, b"value!");
            found = true;
        }
    }
    assert!(found, "needle not retrievable through PIR + stash");
    // a row update keeps PIR correct
    let out = t.update(&k, b"value2".to_vec(), false, UpdateMode::Invalidation, &mut rng);
    let rows: Vec<(usize, Vec<u8>)> = out.accessed.iter().map(|&p| (p, t.encrypt_bucket(p, &mut rng))).collect();
    pir.update_rows(&rows);
    let (_, got) = pir.query_batch(&cands);
    let mut best = (0u32, Vec::new());
    for b in got.iter().map(|b| b.as_slice()).chain(std::iter::once(t.encrypt_stash(&mut rng).as_slice())) {
        for slot in b.chunks(sb) {
            if let Some((v, ver, _)) = t.decrypt_slot(slot, &dk) {
                if ver >= best.0 {
                    best = (ver, v);
                }
            }
        }
    }
    assert_eq!(best.1, b"value2", "updated value not retrieved through PIR");
    println!("ok  encryption + FrodoPIR retrieval before and after an update");

    // 2b. the same retrieval through Spiral (our server), before and after an update
    let rows = t.encrypt_all(&mut rng);
    let (mut sp, _) = Pir::new("spiral", &rows, prm.bucket_bytes());
    let found_val = |got: &[Vec<u8>], t: &Table| -> Option<Vec<u8>> {
        let mut best: Option<(u32, Vec<u8>)> = None;
        for b in got {
            for slot in b.chunks(sb) {
                if slot.len() == sb {
                    if let Some((v, ver, _)) = t.decrypt_slot(slot, &dk) {
                        if best.as_ref().map_or(true, |x| ver >= x.0) {
                            best = Some((ver, v));
                        }
                    }
                }
            }
        }
        best.map(|x| x.1)
    };
    let (_, got) = sp.query_batch_w(&t.candidates(&lk), 8);
    let st = t.encrypt_stash(&mut rng);
    let v0 = found_val(&got, &t).or_else(|| found_val(&[st.clone()], &t));
    assert_eq!(v0.as_deref(), Some(&b"value2"[..]), "Spiral: value not retrieved");
    let out = t.update(&k, b"value3".to_vec(), false, UpdateMode::Invalidation, &mut rng);
    let rows: Vec<(usize, Vec<u8>)> = out.accessed.iter().map(|&p| (p, t.encrypt_bucket(p, &mut rng))).collect();
    sp.update_rows(&rows);
    let (_, got) = sp.query_batch_w(&t.candidates(&lk), 8);
    let st = t.encrypt_stash(&mut rng);
    let mut all = got.clone();
    all.push(st);
    assert_eq!(found_val(&all, &t).as_deref(), Some(&b"value3"[..]), "Spiral: updated value not retrieved");
    println!("ok  encryption + Spiral retrieval before and after an update");

    // 3. MKSE insert / delete / update vs a reference multimap
    for evict in [Evict::Rw, Evict::Bfs] {
        let cfg = mk_cfg(&Args(HashMap::new()), evict, 2000, 4000, 200);
        let (mm, reference_docs) = synthetic(60, 400, 12, &mut rng);
        let docs_lens: HashMap<u32, usize> = HashMap::new();
        let mut m = Mkse::setup(cfg, &mm, &docs_lens, 1500, &mut rng);
        let mut refmm: HashMap<String, Vec<u32>> = mm.iter().cloned().collect();
        let kws: Vec<String> = mm.iter().map(|(w, _)| w.clone()).collect();
        let mut next_id = reference_docs as u32 + 1;
        for _ in 0..3000 {
            let r: f64 = rng.gen();
            if r < 0.7 {
                let w = kws.choose(&mut rng).unwrap().clone();
                let id = if rng.gen_bool(0.5) { next_id } else { rng.gen_range(1..next_id) };
                if id == next_id {
                    next_id += 1;
                }
                if m.docs.get(&id).map_or(false, |d| d.deleted || d.keywords.contains(&w)) {
                    continue;
                }
                m.insert(&w, id, 2500, &mut rng);
                refmm.entry(w).or_default().push(id);
            } else if r < 0.85 {
                let w = kws.choose(&mut rng).unwrap().clone();
                if let Some(list) = refmm.get_mut(&w) {
                    if !list.is_empty() {
                        let id = list.remove(rng.gen_range(0..list.len()));
                        m.delete_posting(&w, id, &mut rng);
                    }
                }
            } else if r < 0.92 {
                let id = rng.gen_range(1..next_id);
                if m.docs.get(&id).map_or(true, |d| d.deleted) {
                    continue;
                }
                m.delete_doc(id, &mut rng);
                for list in refmm.values_mut() {
                    list.retain(|x| *x != id);
                }
            } else {
                let id = rng.gen_range(1..next_id);
                m.update_doc(id, &mut rng);
            }
        }
        for w in &kws {
            let mut got = m.search(w).ids;
            let mut want = refmm.get(w).cloned().unwrap_or_default();
            got.sort_unstable();
            want.sort_unstable();
            want.dedup();
            assert_eq!(got, want, "MKSE search mismatch for {w} ({evict:?})");
        }
        m.val.check_invariants();
        m.doc.check_invariants();
        println!(
            "ok  MKSE {:?}: {} keywords checked, pending peak {}, val stash {}, doc stash {}",
            evict, kws.len(), m.pend_peak, m.val.stash_len(), m.doc.stash_len()
        );
    }
    selftest_design();
    println!("selftest passed");
}

// ------------------------------------------------------------------ stash

fn stash(a: &Args) {
    let n = a.u("n", 1_000_000);
    let prm = Params {
        mu: (a.f("mu-factor", 2.0) * n as f64 / a.u("p", 1) as f64).ceil() as usize,
        p: a.u("p", 1),
        s: a.u("s", 4),
        alpha: a.u("alpha", 10),
        sigma: usize::MAX / 2,
        evict: evict_of(&a.s("evict", "rw")),
        disc: disc_of(&a.s("disc", "paper")),
        bfs_max_branch: a.u("branch", 2),
        bfs_max_depth: a.u("depth", 64),
        val_len: 0,
    };
    let mut rng = StdRng::seed_from_u64(a.u("seed", 1) as u64);
    let joint = (a.f("joint-frac", 0.0) * n as f64) as usize;
    let cps = a.u("checkpoints", 50).max(1);
    let out = a.s("out", "results/stash.csv");
    let mut w = csv(&out, "evict,disc,p,s,alpha,mu,joint_frac,inserted,load,stash,stash_frac,evaluated,moves,secs");
    let mut t = Table::new(prm.clone(), &mut rng);
    let mut keyrng = StdRng::seed_from_u64(a.u("seed", 1) as u64 ^ 0xabcdef);
    let mut mk = |i: usize, r: &mut StdRng| -> (Key, ([u8; 32], [u8; 32]), Vec<u8>) {
        let mut lk = [0u8; 32];
        r.fill(&mut lk);
        (Key::flat(&(i as u64).to_le_bytes()), (lk, [0u8; 32]), vec![])
    };
    let t0 = Instant::now();
    if joint > 0 {
        let ent: Vec<_> = (0..joint).map(|i| mk(i, &mut keyrng)).collect();
        t.encode_joint(ent, a.u("alpha-build", 100), &mut rng);
    }
    let (mut ev, mut mv) = (0usize, 0usize);
    let step = ((n - joint) / cps).max(1);
    let tag = format!(
        "{},{},{},{},{},{},{}",
        a.s("evict", "rw"), a.s("disc", "paper"), prm.p, prm.s, prm.alpha, prm.mu, a.f("joint-frac", 0.0)
    );
    let line = |w: &mut BufWriter<File>, t: &Table, i: usize, ev: usize, mv: usize, secs: f64| {
        writeln!(
            w,
            "{tag},{i},{:.5},{},{:.3e},{ev},{mv},{secs:.1}",
            t.load(),
            t.stash_len(),
            t.stash_len() as f64 / i.max(1) as f64
        )
        .unwrap();
    };
    line(&mut w, &t, joint, 0, 0, 0.0);
    for i in joint..n {
        let (k, (lk, dk), v) = mk(i, &mut keyrng);
        let o = t.insert(&k, lk, dk, v, &mut rng);
        ev += o.evaluated;
        mv += o.moves;
        if (i + 1 - joint) % step == 0 || i + 1 == n {
            line(&mut w, &t, i + 1, ev, mv, t0.elapsed().as_secs_f64());
            w.flush().unwrap();
        }
    }
    println!("{out}: n={n} load={:.3} stash={} ({:.2}s)", t.load(), t.stash_len(), t0.elapsed().as_secs_f64());
}

/// Stash under churn at constant load (not covered by the stash theorems,
/// which bound insertions into an empty table).
fn churn(a: &Args) {
    let n = a.u("n", 1_000_000);
    let prm = Params {
        mu: (a.f("mu-factor", 1.5) * n as f64).ceil() as usize,
        p: 1,
        s: a.u("s", 4),
        alpha: a.u("alpha", 10),
        sigma: usize::MAX / 2,
        evict: evict_of(&a.s("evict", "rw")),
        disc: disc_of(&a.s("disc", "paper")),
        bfs_max_branch: a.u("branch", 2),
        bfs_max_depth: a.u("depth", 64),
        val_len: 0,
    };
    let ops = (a.f("ops-factor", 4.0) * n as f64) as usize;
    let mode = a.s("mode", "update");
    assert!(mode == "update" || mode == "replace", "--mode update|replace");
    let zipf = a.f("zipf", 0.0);
    let rebuild_every = a.u("rebuild-every", 0);
    let cps = a.u("checkpoints", 40).max(1);
    let out = a.s("out", "results/churn.csv");
    let mut rng = StdRng::seed_from_u64(a.u("seed", 1) as u64);
    let mut keyrng = StdRng::seed_from_u64(a.u("seed", 1) as u64 ^ 0x5eed);
    let mut w = csv(&out, "evict,disc,alpha,mu,n,mode,zipf,rebuild_every,ops,load,stash,stash_frac,stash_max,stale,rebuilds,secs");
    // n logical records; slot i holds the current key of record i
    let mut next_id = 0u64;
    let mut fresh = |r: &mut StdRng| -> (Key, [u8; 32]) {
        let mut lk = [0u8; 32];
        r.fill(&mut lk);
        next_id += 1;
        (Key::flat(&next_id.to_le_bytes()), lk)
    };
    let mut slots: Vec<(Key, [u8; 32])> = (0..n).map(|_| fresh(&mut keyrng)).collect();
    let mut t = Table::new(prm.clone(), &mut rng);
    for (k, lk) in &slots {
        t.insert(k, *lk, [0u8; 32], vec![], &mut rng);
    }
    // Zipf(zipf) over record ranks (a random permutation decouples rank from slot order)
    let cdf: Vec<f64> = if zipf > 0.0 {
        let mut c = Vec::with_capacity(n);
        let mut acc = 0.0;
        for i in 0..n {
            acc += 1.0 / ((i + 1) as f64).powf(zipf);
            c.push(acc);
        }
        let tot = acc;
        c.iter_mut().for_each(|x| *x /= tot);
        c
    } else {
        vec![]
    };
    let mut perm: Vec<usize> = (0..n).collect();
    perm.shuffle(&mut rng);
    let mut pick = |r: &mut StdRng| -> usize {
        if cdf.is_empty() {
            r.gen_range(0..n)
        } else {
            let u: f64 = r.gen();
            perm[cdf.partition_point(|&x| x < u).min(n - 1)]
        }
    };
    let tag = format!(
        "{},{},{},{},{},{},{},{}",
        a.s("evict", "rw"), a.s("disc", "paper"), prm.alpha, prm.mu, n, mode, zipf, rebuild_every
    );
    let t0 = Instant::now();
    let mut smax = t.stash_len();
    let mut rebuilds = 0usize;
    let step = (ops / cps).max(1);
    let line = |w: &mut BufWriter<File>, t: &Table, i: usize, smax: usize, rebuilds: usize| {
        writeln!(
            w,
            "{tag},{i},{:.5},{},{:.3e},{smax},{},{rebuilds},{:.1}",
            t.load(),
            t.stash_len(),
            t.stash_len() as f64 / n as f64,
            t.stale_count,
            t0.elapsed().as_secs_f64()
        )
        .unwrap();
    };
    line(&mut w, &t, 0, smax, 0);
    for i in 0..ops {
        let x = pick(&mut rng);
        if mode == "update" {
            let k = slots[x].0.clone();
            t.update(&k, vec![], false, UpdateMode::Invalidation, &mut rng);
        } else {
            let old = slots[x].0.clone();
            t.remove(&old);
            let (k, lk) = fresh(&mut keyrng);
            t.insert(&k, lk, [0u8; 32], vec![], &mut rng);
            slots[x] = (k, lk);
        }
        smax = smax.max(t.stash_len());
        if rebuild_every > 0 && (i + 1) % rebuild_every == 0 {
            // a rebuild re-inserts the live keys into a fresh table: every key gets s fresh states
            let mut nt = Table::new(prm.clone(), &mut rng);
            for (k, lk) in &slots {
                nt.insert(k, *lk, [0u8; 32], vec![], &mut rng);
            }
            t = nt;
            rebuilds += 1;
        }
        if (i + 1) % step == 0 || i + 1 == ops {
            line(&mut w, &t, i + 1, smax, rebuilds);
            w.flush().unwrap();
        }
    }
    println!(
        "{out}: n={n} mode={mode} zipf={zipf} rebuild={rebuild_every} load={:.3} stash={} max={smax} ({:.1}s)",
        t.load(),
        t.stash_len(),
        t0.elapsed().as_secs_f64()
    );
}

// ------------------------------------------------------------------ data

/// keyword -> ids, sorted by id within a keyword; ids are the file names.
fn load_data(dir: &str) -> (Vec<(String, Vec<u32>)>, HashMap<u32, usize>) {
    let buf = std::fs::read_to_string(Path::new(dir).join("keyword_file_map.json")).expect("keyword_file_map.json");
    let v: sonic_rs::Value = sonic_rs::from_str(&buf).expect("json");
    use sonic_rs::{JsonContainerTrait, JsonValueTrait};
    let mut mm = Vec::new();
    let mut ids_seen = HashSet::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj.iter() {
            let mut ids: Vec<u32> = val
                .as_array()
                .map(|arr| arr.iter().filter_map(|x| x.as_str().and_then(|s| s.parse().ok())).collect())
                .unwrap_or_default();
            ids.sort_unstable();
            ids.dedup();
            ids_seen.extend(ids.iter().copied());
            mm.push((k.to_string(), ids));
        }
    }
    mm.sort_by(|a, b| a.0.cmp(&b.0));
    // document lengths from the files themselves, if present
    let mut lens = HashMap::new();
    for id in ids_seen {
        if let Ok(md) = std::fs::metadata(Path::new(dir).join(id.to_string())) {
            lens.insert(id, md.len() as usize);
        }
    }
    (mm, lens)
}

/// Zipf-like synthetic multimap: `kw` keywords over `docs` documents with
/// `per_doc` keywords per document on average.
fn synthetic(kw: usize, docs: usize, per_doc: usize, rng: &mut StdRng) -> (Vec<(String, Vec<u32>)>, usize) {
    let weights: Vec<f64> = (1..=kw).map(|r| 1.0 / r as f64).collect();
    let total: f64 = weights.iter().sum();
    let cdf: Vec<f64> = weights.iter().scan(0.0, |acc, w| { *acc += w / total; Some(*acc) }).collect();
    let mut mm: Vec<Vec<u32>> = vec![vec![]; kw];
    for d in 1..=docs as u32 {
        let k = rng.gen_range(1..=2 * per_doc);
        let mut chosen = HashSet::new();
        for _ in 0..k {
            let u: f64 = rng.gen();
            let i = cdf.partition_point(|&c| c < u).min(kw - 1);
            if chosen.insert(i) {
                mm[i].push(d);
            }
        }
    }
    (mm.into_iter().enumerate().filter(|(_, v)| !v.is_empty()).map(|(i, v)| (format!("w{i}"), v)).collect(), docs)
}

fn mk_cfg(a: &Args, evict: Evict, postings: usize, doc_chunks: usize, keywords: usize) -> MkseCfg {
    let bucket = a.u("bucket-bytes", 8192);
    let pb = a.u("posting-bytes", 40);
    let slot_over = mkse::uokvs::NONCE + mkse::uokvs::META + mkse::uokvs::CHECK;
    let val_len = bucket - slot_over;
    let cap = val_len / pb;
    let flush_at = ((a.f("flush-frac", 0.5) * cap as f64) as usize).max(1);
    let alpha = a.u("alpha", if evict == Evict::Rw { 10 } else { 20 });
    let disc = disc_of(&a.s("disc", "paper"));
    // expected chunks: one partial chunk per keyword plus one per flush; with a
    // pending budget, flushes are smaller, so allow an explicit override
    let mu_val = match a.s("val-mu", "auto").as_str() {
        "auto" => ((a.f("mu-factor", 2.0) * (postings / flush_at + keywords) as f64).ceil() as usize).max(16),
        x => parse_num(x) as usize,
    };
    let doc_chunk = a.u("doc-chunk", 1024);
    let mu_doc = ((a.f("doc-mu-factor", 2.0) * doc_chunks as f64).ceil() as usize).max(16);
    let budget = match a.s("pend-budget", "none").as_str() {
        "none" => None,
        x => Some(parse_num(x) as usize),
    };
    let pend_sigma = match a.s("pend-sigma", "auto").as_str() {
        "auto" => budget.map(|b| b + 1).unwrap_or(keywords * flush_at / 2 + 1024),
        x => parse_num(x) as usize,
    };
    let vol_sigma = match a.s("vol-sigma", "auto").as_str() {
        "auto" => keywords + keywords / 2 + 64,
        x => parse_num(x) as usize,
    };
    MkseCfg {
        val: Params {
            mu: mu_val,
            p: 1,
            s: a.u("s", 4),
            alpha,
            sigma: a.u("sigma", 64),
            evict,
            disc,
            bfs_max_branch: a.u("branch", 2),
            bfs_max_depth: a.u("depth", 64),
            val_len,
        },
        doc: Params {
            mu: mu_doc,
            p: 1,
            s: a.u("s", 4),
            alpha,
            sigma: a.u("doc-sigma", 256),
            evict,
            disc,
            bfs_max_branch: a.u("branch", 2),
            bfs_max_depth: a.u("depth", 64),
            val_len: doc_chunk,
        },
        vol_sigma,
        pend_sigma,
        posting_bytes: pb,
        flush_at,
        pend_budget: budget,
        update_mode: mode_of(&a.s("update-mode", "inval")),
        doc_chunk_bytes: doc_chunk,
    }
}

// ------------------------------------------------------------------ ops

fn ops(a: &Args) {
    let mut rng = StdRng::seed_from_u64(a.u("seed", 1) as u64);
    let (mm_full, lens) = if a.b("data") {
        load_data(&a.s("data", ""))
    } else {
        let sp = a.list("synthetic", "20000,100000,40");
        let (mm, _) = synthetic(sp[0], sp[1], sp[2], &mut rng);
        (mm, HashMap::new())
    };
    let default_len = a.u("default-doc-len", 4096);
    // documents in insertion order (file id order approximates time)
    let mut doc_kw: HashMap<u32, Vec<String>> = HashMap::new();
    for (w, ids) in &mm_full {
        for &id in ids {
            doc_kw.entry(id).or_default().push(w.clone());
        }
    }
    let mut docs: Vec<u32> = doc_kw.keys().copied().collect();
    docs.sort_unstable();
    if a.s("order", "fileid") == "random" {
        docs.shuffle(&mut rng);
    }
    let maxd = a.u("max-docs", 0);
    if maxd > 0 && docs.len() > maxd {
        docs.truncate(maxd);
    }
    let nb = (a.f("build-frac", 0.8) * docs.len() as f64) as usize;
    let build: HashSet<u32> = docs[..nb].iter().copied().collect();
    let mm_build: Vec<(String, Vec<u32>)> = mm_full
        .iter()
        .map(|(w, ids)| (w.clone(), ids.iter().copied().filter(|i| build.contains(i)).collect::<Vec<_>>()))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let postings_total: usize = docs.iter().map(|d| doc_kw[d].len()).sum();
    let doc_chunks_total: usize = docs
        .iter()
        .map(|d| (lens.get(d).copied().unwrap_or(default_len) + a.u("doc-chunk", 1024) - 1) / a.u("doc-chunk", 1024))
        .sum();
    let evict = evict_of(&a.s("evict", "bfs"));
    let cfg = mk_cfg(a, evict, postings_total, doc_chunks_total, mm_full.len());
    println!(
        "dataset: {} keywords, {} docs ({} in setup), {} postings, {} doc chunks; ValMap mu={} chunk cap={} flush_at={}; DocMap mu={}",
        mm_full.len(), docs.len(), nb, postings_total, doc_chunks_total, cfg.val.mu, cfg.chunk_cap(), cfg.flush_at, cfg.doc.mu
    );
    let t = Instant::now();
    let mut m = Mkse::setup(cfg.clone(), &mm_build, &lens, default_len, &mut rng);
    let setup_s = t.elapsed().as_secs_f64();
    println!("setup {:.1}s: ValMap {} chunks (load {:.3}), pending {}, DocMap load {:.3}", setup_s, m.val.len(), m.val.load(), m.pend_total, m.doc.load());

    let encrypt = a.b("encrypt") || a.b("pir");
    let no_doc_pir = a.b("no-doc-pir");
    let use_pir = a.b("pir");
    let (mut pv, mut pd) = (None, None);
    if use_pir {
        let t = Instant::now();
        let rows = m.val.encrypt_all(&mut rng);
        pv = Some(FrodoPir::new(&rows, cfg.val.bucket_bytes()).0);
        drop(rows);
        let doc_rows = if a.b("no-doc-pir") { vec![vec![0u8; 8]; 16] } else { m.doc.encrypt_all(&mut rng) };
        pd = Some(FrodoPir::new(&doc_rows, if a.b("no-doc-pir") { 8 } else { cfg.doc.bucket_bytes() }).0);
        println!(
            "PIR setup {:.1}s; ValMap hint {:.1} MB, DocMap hint {:.1} MB",
            t.elapsed().as_secs_f64(),
            pv.as_ref().unwrap().hint_bytes() as f64 / 1e6,
            pd.as_ref().unwrap().hint_bytes() as f64 / 1e6
        );
    }

    let pre = a.s("out-prefix", "results/ops");
    let mut wo = csv(
        &format!("{pre}_ops.csv"),
        "i,kind,val_ops,val_accesses,doc_ops,doc_accesses,flushes,new_doc,owner_us,encrypt_us,upload_bytes,pir_val_db_ms,pir_val_hint_ms,pir_doc_db_ms,pir_doc_hint_ms,hint_refresh_bytes_raw,hint_refresh_bytes_packed",
    );
    let mut ws = csv(
        &format!("{pre}_snap.csv"),
        "i,pend_total,pend_peak,pend_keywords,pend_overflow,val_stash,val_stash_peak,val_overflow,val_load,val_stale,doc_stash,doc_stash_peak,doc_overflow,doc_load,doc_stale,vol_len",
    );
    let mut wq = csv(
        &format!("{pre}_search.csv"),
        "i,keyword,volume,chunks,pending_seen,val_queries,doc_queries,stash_download_bytes,pir_val_server_ms,pir_val_client_ms,pir_val_up,pir_val_down,pir_doc_server_ms,pir_doc_client_ms,pir_doc_up,pir_doc_down",
    );

    let del_frac = a.f("del-frac", 0.02);
    let upd_frac = a.f("upd-frac", 0.02);
    let search_every = a.u("search-every", 1000);
    let search_sample = a.u("search-sample", 20);
    let mut refmm: HashMap<String, Vec<u32>> = mm_build.iter().cloned().collect();
    let mut live_docs: Vec<u32> = docs[..nb].to_vec();
    let mut i = 0usize;
    let kws: Vec<String> = mm_full.iter().map(|(w, _)| w.clone()).collect();

    let mut record = |i: usize, r: &MkseOp, m: &Mkse, owner_us: f64, rng: &mut StdRng,
                      pv: &mut Option<FrodoPir>, pd: &mut Option<FrodoPir>, wo: &mut BufWriter<File>| {
        let mut enc_us = 0.0;
        let (mut vdb, mut vh, mut ddb, mut dh, mut raw, mut packed) = (0.0, 0.0, 0.0, 0.0, 0usize, 0usize);
        if encrypt {
            let t = Instant::now();
            let vrows: Vec<(usize, Vec<u8>)> = r.val.iter().flat_map(|o| o.accessed.iter()).map(|&p| (p, m.val.encrypt_bucket(p, rng))).collect();
            let drows: Vec<(usize, Vec<u8>)> = r.doc.iter().flat_map(|o| o.accessed.iter()).map(|&p| (p, m.doc.encrypt_bucket(p, rng))).collect();
            let _vs = m.val.encrypt_stash(rng);
            let _ds = if r.doc.is_empty() { vec![] } else { m.doc.encrypt_stash(rng) };
            let _vol = m.vol.encrypt_stash(rng);
            enc_us = t.elapsed().as_secs_f64() * 1e6;
            if let (Some(pv), Some(pd)) = (pv.as_mut(), pd.as_mut()) {
                let c = pv.update_rows(&vrows);
                vdb = c.server_db_ms;
                vh = c.server_hint_ms;
                raw += c.client_bytes_raw;
                packed += c.client_bytes_packed;
                if !drows.is_empty() && !no_doc_pir {
                    let c = pd.update_rows(&drows);
                    ddb = c.server_db_ms;
                    dh = c.server_hint_ms;
                    raw += c.client_bytes_raw;
                    packed += c.client_bytes_packed;
                }
            }
        }
        let kind = match r.kind {
            OpKind::Insert => "insert",
            OpKind::DeletePosting => "delete_posting",
            OpKind::DeleteDoc => "delete_doc",
            OpKind::UpdateDoc => "update_doc",
        };
        writeln!(
            wo,
            "{i},{kind},{},{},{},{},{},{},{owner_us:.1},{enc_us:.1},{},{vdb:.3},{vh:.3},{ddb:.3},{dh:.3},{raw},{packed}",
            r.val.len(), r.val_accesses(), r.doc.len(), r.doc_accesses(), r.flushes, r.new_doc as u8, m.upload_bytes(r)
        )
        .unwrap();
    };

    let snap = |i: usize, m: &Mkse, ws: &mut BufWriter<File>| {
        writeln!(
            ws,
            "{i},{},{},{},{},{},{},{},{:.4},{},{},{},{},{:.4},{},{}",
            m.pend_total, m.pend_peak, m.n_keywords_with_pending(), m.pend_overflow,
            m.val.stash_len(), m.val.stash_peak, m.val.overflow, m.val.load(), m.val.stale_count,
            m.doc.stash_len(), m.doc.stash_peak, m.doc.overflow, m.doc.load(), m.doc.stale_count, m.vol.len()
        )
        .unwrap();
    };

    let do_search = |i: usize, m: &Mkse, refmm: &HashMap<String, Vec<u32>>, rng: &mut StdRng,
                     pv: &mut Option<FrodoPir>, pd: &mut Option<FrodoPir>, wq: &mut BufWriter<File>| {
        for k in 0..search_sample {
            let w = kws.choose(rng).unwrap();
            let s = m.search(w);
            let mut want = refmm.get(w).cloned().unwrap_or_default();
            want.sort_unstable();
            want.dedup();
            let mut got = s.ids.clone();
            got.sort_unstable();
            assert_eq!(got, want, "search mismatch for {w}");
            let mut cols = String::from(",,,,,,,");
            if let (true, Some(pv), Some(pd)) = (k < a.u("pir-search", 3), pv.as_mut(), pd.as_mut()) {
                let (cv, _) = pv.query_batch(&s.val_queries);
                let dq: Vec<usize> = s.doc_queries.iter().copied().take(a.u("max-doc-queries", 512)).collect();
                let (cd, _) = if dq.is_empty() || no_doc_pir { (Default::default(), vec![]) } else { pd.query_batch(&dq) };
                cols = format!(
                    "{:.1},{:.1},{},{},{:.1},{:.1},{},{}",
                    cv.server_ms, cv.client_prep_ms + cv.client_query_ms + cv.client_decode_ms, cv.upload_bytes, cv.download_bytes,
                    cd.server_ms, cd.client_prep_ms + cd.client_query_ms + cd.client_decode_ms, cd.upload_bytes, cd.download_bytes
                );
            }
            writeln!(
                wq,
                "{i},{w},{},{},{},{},{},{},{cols}",
                got.len(), s.chunks, s.pending_seen, s.val_queries.len(), s.doc_queries.len(), m.search_stash_bytes()
            )
            .unwrap();
        }
    };

    snap(0, &m, &mut ws);
    do_search(0, &m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
    for &d in &docs[nb..] {
        for w in doc_kw[&d].clone() {
            let t = Instant::now();
            let r = m.insert(&w, d, lens.get(&d).copied().unwrap_or(default_len), &mut rng);
            let us = t.elapsed().as_secs_f64() * 1e6;
            refmm.entry(w).or_default().push(d);
            i += 1;
            record(i, &r, &m, us, &mut rng, &mut pv, &mut pd, &mut wo);
            if i % search_every == 0 {
                snap(i, &m, &mut ws);
                do_search(i, &m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
            }
            // interleaved deletions and updates
            let r2: f64 = rng.gen();
            if r2 < del_frac {
                let w2 = kws.choose(&mut rng).unwrap().clone();
                if let Some(list) = refmm.get_mut(&w2) {
                    if !list.is_empty() {
                        let id = list.remove(rng.gen_range(0..list.len()));
                        let t = Instant::now();
                        let r = m.delete_posting(&w2, id, &mut rng);
                        let us = t.elapsed().as_secs_f64() * 1e6;
                        i += 1;
                        record(i, &r, &m, us, &mut rng, &mut pv, &mut pd, &mut wo);
                    }
                }
            } else if r2 < del_frac + upd_frac && !live_docs.is_empty() {
                let id = *live_docs.choose(&mut rng).unwrap();
                let t = Instant::now();
                let r = m.update_doc(id, &mut rng);
                let us = t.elapsed().as_secs_f64() * 1e6;
                i += 1;
                record(i, &r, &m, us, &mut rng, &mut pv, &mut pd, &mut wo);
            } else if r2 < del_frac + upd_frac + a.f("docdel-frac", 0.002) && !live_docs.is_empty() {
                let k = rng.gen_range(0..live_docs.len());
                let id = live_docs.swap_remove(k);
                let t = Instant::now();
                let r = m.delete_doc(id, &mut rng);
                let us = t.elapsed().as_secs_f64() * 1e6;
                for list in refmm.values_mut() {
                    list.retain(|x| *x != id);
                }
                i += 1;
                record(i, &r, &m, us, &mut rng, &mut pv, &mut pd, &mut wo);
            }
        }
        live_docs.push(d);
    }
    snap(i, &m, &mut ws);
    do_search(i, &m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
    wo.flush().unwrap();
    ws.flush().unwrap();
    wq.flush().unwrap();
    println!(
        "done: {i} operations; pending peak {} (pend-sigma {}), ValMap stash peak {} (sigma {}), DocMap stash peak {}, overflows val/doc/pend = {}/{}/{}",
        m.pend_peak, cfg.pend_sigma, m.val.stash_peak, cfg.val.sigma, m.doc.stash_peak, m.val.overflow, m.doc.overflow, m.pend_overflow
    );
    if m.val.load() > 0.6 || m.doc.load() > 0.6 {
        println!("WARNING: final load ValMap {:.2} / DocMap {:.2} is high; raise --val-mu / --doc-mu-factor", m.val.load(), m.doc.load());
    }
    println!("outputs: {pre}_ops.csv {pre}_snap.csv {pre}_search.csv");
}

// ------------------------------------------------------------------ pir

fn pir(a: &Args) {
    let mut rng = StdRng::seed_from_u64(a.u("seed", 1) as u64);
    let rows = a.u("rows", 1 << 18);
    let eb = a.u("elem-bytes", 8192);
    let trials = a.u("trials", 3);
    let out = a.s("out", "results/pir.csv");
    let mut w = csv(&out, "scheme,rows,elem_bytes,metric,param,value");
    let mut data = random_rows(rows, eb, &mut rng);
    let (mut fp, setup_ms) = FrodoPir::new(&data, eb);
    let base = format!("frodopir,{rows},{eb}");
    writeln!(w, "{base},setup_ms,,{setup_ms:.1}").unwrap();
    writeln!(w, "{base},hint_bytes,,{}", fp.hint_bytes()).unwrap();
    writeln!(w, "{base},query_bytes,,{}", fp.query_bytes()).unwrap();
    writeln!(w, "{base},response_bytes,,{}", fp.response_bytes()).unwrap();
    for k in a.list("update-rows", "10,21") {
        for _ in 0..trials {
            let upd: Vec<(usize, Vec<u8>)> = (0..k)
                .map(|_| {
                    let mut v = vec![0u8; eb];
                    rng.fill(&mut v[..]);
                    (rng.gen_range(0..rows), v)
                })
                .collect();
            let c = fp.update_rows(&upd);
            for (i, v) in &upd {
                data[*i] = v.clone();
            }
            writeln!(w, "{base},update_server_ms,{k},{:.3}", c.server_db_ms + c.server_hint_ms).unwrap();
            writeln!(w, "{base},update_client_bytes_raw,{k},{}", c.client_bytes_raw).unwrap();
            writeln!(w, "{base},update_client_bytes_packed,{k},{}", c.client_bytes_packed).unwrap();
        }
    }
    // correctness check of a query
    let probe = rng.gen_range(0..rows);
    let (_, got) = fp.query_batch(&[probe]);
    let mut want = data[probe].clone();
    want.truncate(got[0].len());
    assert_eq!(got[0][..want.len().min(64)], want[..want.len().min(64)], "FrodoPIR query returned a wrong row");
    for width in a.list("widths", "8") {
        let base = format!("frodopir_w{width},{rows},{eb}");
        for b in a.list("batches", "4,8,16,64,784") {
            for _ in 0..trials {
                let idx: Vec<usize> = (0..b).map(|_| rng.gen_range(0..rows)).collect();
                let t = Instant::now();
                let (c, got) = fp.query_batch_w(&idx, width);
                let wall = t.elapsed().as_secs_f64() * 1e3;
                let k = rng.gen_range(0..b);
                assert_eq!(got[k][..64.min(eb)], data[idx[k]][..64.min(eb)], "wrong row at width {width}");
                writeln!(w, "{base},batch_client_prep_ms,{b},{:.1}", c.client_prep_ms).unwrap();
                writeln!(w, "{base},batch_client_online_ms,{b},{:.1}", c.client_query_ms + c.client_decode_ms).unwrap();
                writeln!(w, "{base},batch_server_ms,{b},{:.1}", c.server_ms).unwrap();
                writeln!(w, "{base},batch_wall_ms,{b},{:.1}", wall).unwrap();
                writeln!(w, "{base},batch_up_bytes,{b},{}", c.upload_bytes).unwrap();
                writeln!(w, "{base},batch_down_bytes,{b},{}", c.download_bytes).unwrap();
            }
        }
    }
    w.flush().unwrap();
    println!("{out}: rows={rows} elem={eb}B hint={:.1}MB setup={:.1}s", fp.hint_bytes() as f64 / 1e6, setup_ms / 1e3);
}

// ------------------------------------------------------------------ design

struct Workload {
    mm_full: Vec<(String, Vec<u32>)>,
    lens: HashMap<u32, usize>,
    docs: Vec<u32>,
    doc_kw: HashMap<u32, Vec<String>>,
    nb: usize,
    default_len: usize,
}

fn workload(a: &Args, rng: &mut StdRng) -> Workload {
    let (mm_full, lens) = if a.b("data") {
        load_data(&a.s("data", ""))
    } else {
        let sp = a.list("synthetic", "20000,100000,40");
        let (mm, _) = synthetic(sp[0], sp[1], sp[2], rng);
        (mm, HashMap::new())
    };
    let mut doc_kw: HashMap<u32, Vec<String>> = HashMap::new();
    for (w, ids) in &mm_full {
        for &id in ids {
            doc_kw.entry(id).or_default().push(w.clone());
        }
    }
    let mut docs: Vec<u32> = doc_kw.keys().copied().collect();
    docs.sort_unstable();
    if a.s("order", "fileid") == "random" {
        docs.shuffle(rng);
    }
    let maxd = a.u("max-docs", 0);
    if maxd > 0 && docs.len() > maxd {
        docs.truncate(maxd);
    }
    let keep: HashSet<u32> = docs.iter().copied().collect();
    let mm_full: Vec<(String, Vec<u32>)> = mm_full
        .into_iter()
        .map(|(w, ids)| (w, ids.into_iter().filter(|i| keep.contains(i)).collect::<Vec<_>>()))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    doc_kw.retain(|d, _| keep.contains(d));
    let nb = (a.f("build-frac", 0.8) * docs.len() as f64) as usize;
    Workload { mm_full, lens, docs, doc_kw, nb, default_len: a.u("default-doc-len", 4096) }
}

fn log_cfg(a: &Args, wl: &Workload) -> LogCfg {
    let bucket = a.u("bucket-bytes", 8192);
    let pb = a.u("posting-bytes", 40);
    let slot_over = mkse::uokvs::NONCE + mkse::uokvs::META + mkse::uokvs::CHECK;
    let val_len = bucket - slot_over;
    let c = val_len / pb;
    let evict = evict_of(&a.s("evict", "bfs"));
    // design default: BFS with alpha = 20 (with sigma = 16; Theorem intro-bfs)
    let alpha = a.u("alpha", 20);
    // ValMap: canonical chunks of the full dataset, inflated by --mu-factor
    let del = a.f("del-frac", 0.02) + a.f("docdel-frac", 0.002) * 40.0;
    let chunks: usize = wl
        .mm_full
        .iter()
        .map(|(_, ids)| ((ids.len() as f64 * (1.0 + del)).ceil() as usize + c - 1) / c)
        .sum();
    let mu_of = |mf: f64| ((mf * chunks as f64).ceil() as usize).max(64);
    let mu_val = match a.s("val-mu", "auto").as_str() {
        "auto" => mu_of(a.f("mu-factor", 2.0)),
        x => parse_num(x) as usize,
    };
    // public ValMap rebuild size: ceil(0.46 mu) of the reuse-mode table (keep
    // mode, --no-reuse, takes mu at --reuse-mu-factor, default 2.0, so its
    // larger table absorbs the fillers and superseded chunks of an epoch)
    let val_cap = match a.s("val-cap", "auto").as_str() {
        "auto" => {
            let mu_r = if a.b("no-reuse") { mu_of(a.f("reuse-mu-factor", 2.0)) } else { mu_val };
            (a.f("val-cap-frac", 0.46) * mu_r as f64).ceil() as usize
        }
        x => parse_num(x) as usize,
    };
    let doc_row = a.u("doc-row", 1024);
    let doc_chunks: usize = wl
        .docs
        .iter()
        .map(|d| ((wl.lens.get(d).copied().unwrap_or(wl.default_len) + doc_row - 1) / doc_row).max(1))
        .sum();
    let edits = a.f("upd-frac", 0.02) * wl.doc_kw.values().map(|v| v.len()).sum::<usize>() as f64 * (1.0 - a.f("build-frac", 0.8));
    let avg = doc_chunks as f64 / wl.docs.len().max(1) as f64;
    let doc_rows = ((doc_chunks as f64 + edits * avg) * (1.0 + a.f("doc-spare", 0.2))).ceil() as usize + 64;
    // public cap on live DocMap chunks at a repack / rebuild: the expected
    // content (all documents of the data set + edits) without the spare rows,
    // so a repack leaves doc_rows - doc_live_cap ~ doc-spare * doc_live_cap + 64
    // never-written rows
    let doc_live_cap = match a.s("doc-live-cap", "auto").as_str() {
        "auto" => (doc_chunks as f64 + edits * avg).ceil() as usize,
        x => parse_num(x) as usize,
    };
    let stateless = a.b("stateless");
    let policy = match a.s("policy", "batched").as_str() {
        // stateless clients: every batch commits everything pending (theta = 1),
        // so the pending stash stays below 2B and VolMap is refreshed per batch
        "batched" if stateless => {
            let b = a.u("b", 100);
            let k = match a.s("k", "auto").as_str() {
                "auto" => b + b / 8,
                x => parse_num(x) as usize,
            };
            Policy::Batched { b, k, theta: a.u("theta", 1).max(1), age: usize::MAX / 4 }
        }
        "batched" => {
            // recommended schedule: large batches, flush a keyword once a full chunk is pending
            let b = a.u("b", 10_000);
            let theta = a.u("theta", c).max(1);
            let k = match a.s("k", "auto").as_str() {
                "auto" => (3 * b + 2 * theta - 1) / (2 * theta) + 2,
                x => parse_num(x) as usize,
            };
            let age = match a.s("age", "inf").as_str() {
                "inf" => usize::MAX / 4,
                x => parse_num(x) as usize,
            };
            Policy::Batched { b, k, theta, age }
        }
        "perop" => Policy::Batched { b: 1, k: 1, theta: a.u("theta", c / 2).max(1), age: usize::MAX / 4 },
        "rebuild" => Policy::Rebuild { r: a.u("r", 100_000) },
        x => panic!("--policy batched|perop|rebuild, got {x}"),
    };
    let doc_layout = match a.s("doc-layout", if stateless { "fixed" } else { "rows" }).as_str() {
        "rows" => DocLayout::Rows,
        "cuckoo" => DocLayout::Cuckoo,
        "fixed" => DocLayout::Fixed,
        x => panic!("--doc-layout rows|cuckoo|fixed, got {x}"),
    };
    assert!(!stateless || doc_layout == DocLayout::Fixed, "--stateless needs --doc-layout fixed (clients keep no version records)");
    let pend_pad = match &policy {
        Policy::Batched { b, .. } => 2 * b,
        _ => 0,
    };
    // public number of version records in a log snapshot: those of every
    // document of the data set at its original length, plus --ver-spare
    // (default 10%) headroom for edits, plus 64
    let record_body = a.u("record-body", pb.max(40));
    let rpr = ((record_body - 8) / 4).max(1);
    let ver_cap = match a.s("ver-cap", "auto").as_str() {
        "auto" => {
            let recs: usize = match doc_layout {
                DocLayout::Rows => wl
                    .docs
                    .iter()
                    .map(|d| {
                        let n = ((wl.lens.get(d).copied().unwrap_or(wl.default_len) + doc_row - 1) / doc_row).max(1);
                        (n + rpr - 1) / rpr
                    })
                    .sum(),
                DocLayout::Cuckoo => wl.docs.len(),
                DocLayout::Fixed => 0,
            };
            if recs == 0 { 0 } else { (recs as f64 * (1.0 + a.f("ver-spare", 0.1))).ceil() as usize + 64 }
        }
        x => parse_num(x) as usize,
    };
    // fixed DocMap layout: version-map slots per epoch (one per edit or
    // removal): 1.2 x the expected edits and removals between scheduled rebuilds
    let major_every = a.u("major-every", 1_000_000);
    let vmap_pad = match a.s("vmap-pad", "auto").as_str() {
        "auto" => {
            let e = if major_every > 0 { major_every } else { 1_000_000 };
            (1.2 * (a.f("upd-frac", 0.02) + a.f("docdel-frac", 0.002)) * e as f64).ceil() as usize + 64
        }
        x => parse_num(x) as usize,
    };
    let tparams = |mu: usize, val_len: usize, sigma: usize| Params {
        mu,
        p: 1,
        s: a.u("s", 4),
        alpha,
        sigma,
        evict,
        disc: disc_of(&a.s("disc", "paper")),
        bfs_max_branch: a.u("branch", 2),
        bfs_max_depth: a.u("depth", 64),
        val_len,
    };
    LogCfg {
        val: tparams(mu_val, val_len, a.u("sigma", 16)),
        posting_bytes: pb,
        policy,
        major_every,
        doc_layout,
        doc: tparams(((a.f("doc-mu-factor", 2.0) * (doc_chunks as f64 + edits * avg)).ceil() as usize).max(64), doc_row, a.u("doc-sigma", 64)),
        doc_row_bytes: doc_row,
        doc_rows,
        record_body,
        kw_pad: wl.mm_full.len(),
        reuse_freed: !a.b("no-reuse"),
        stateless,
        pend_pad,
        val_cap,
        doc_live_cap,
        doc_repack_below: a.u("doc-repack-below", 0),
        ver_cap,
        vmap_pad,
    }
}

fn policy_cols(p: &Policy) -> String {
    match p {
        Policy::Batched { b, k, theta, age } => {
            let age = if *age >= usize::MAX / 8 { "inf".to_string() } else { age.to_string() };
            format!("batched,{b},{k},{theta},{age},")
        }
        Policy::Rebuild { r } => format!("rebuild,,,,,{r}"),
    }
}

fn append_csv(path: &str, header: &str) -> BufWriter<File> {
    // an existing file with another column layout (an older version) is moved aside
    if let Ok(old) = std::fs::read_to_string(path) {
        if old.lines().next().map_or(false, |l| l != header) {
            let aside = format!("{path}.old-{}", chrono::Local::now().format("%Y%m%d-%H%M%S"));
            std::fs::rename(path, &aside).ok();
            println!("note: {path} had an older column layout; moved to {aside}");
        }
    }
    let exists = Path::new(path).exists();
    if let Some(dir) = Path::new(path).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    let f = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("open summary");
    let mut w = BufWriter::new(f);
    if !exists {
        writeln!(w, "{header}").unwrap();
    }
    w
}

fn design(a: &Args) {
    let mut rng = StdRng::seed_from_u64(a.u("seed", 1) as u64);
    let wl = workload(a, &mut rng);
    let cfg = log_cfg(a, &wl);
    let label = a.s("label", "run");
    let pre = a.s("out-prefix", "results/design/run");
    let build: HashSet<u32> = wl.docs[..wl.nb].iter().copied().collect();
    let mm_build: Vec<(String, Vec<u32>)> = wl
        .mm_full
        .iter()
        .map(|(w, ids)| (w.clone(), ids.iter().copied().filter(|i| build.contains(i)).collect::<Vec<_>>()))
        .filter(|(_, v)| !v.is_empty())
        .collect();
    let all_kw: Vec<String> = wl.mm_full.iter().map(|(w, _)| w.clone()).collect();
    let postings_total: usize = wl.mm_full.iter().map(|(_, v)| v.len()).sum();
    println!(
        "[{label}] {} keywords, {} docs ({} at setup), {} postings; C={} ValMap mu={} DocMap {:?} rows={} policy={:?} major_every={}",
        all_kw.len(), wl.docs.len(), wl.nb, postings_total, cfg.chunk_cap(), cfg.val.mu, cfg.doc_layout,
        if cfg.doc_layout == DocLayout::Rows { cfg.doc_rows } else { cfg.doc.mu }, cfg.policy, cfg.major_every
    );
    let t = Instant::now();
    let mut m = MkseLog::setup(cfg.clone(), &mm_build, &all_kw, &wl.lens, wl.default_len, &mut rng);
    let setup_ms = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "[{label}] setup {:.1}s: ValMap {} chunks (load {:.3}), log {:.1} MB, ValMap hint {:.1} MB, DocMap hint {:.2} MB",
        setup_ms / 1e3, m.val.len(), m.val.load(), m.log_bytes() as f64 / 1e6, m.val_hint_bytes() as f64 / 1e6, m.doc_hint_bytes() as f64 / 1e6
    );

    // optional PIR
    // --pir runs real PIR databases; --val-pir / --doc-pir pick FrodoPIR or Spiral per map
    let use_pir = a.b("pir");
    let doc_pir = use_pir && !a.b("no-doc-pir");
    let width = a.u("pir-width", 32);
    let val_kind = a.s("val-pir", "frodo");
    let doc_kind = a.s("doc-pir", "frodo");
    let (mut pv, mut pd): (Option<Pir>, Option<Pir>) = (None, None);
    let mut pir_setup_ms = (0.0, 0.0);
    if use_pir {
        let rows = m.val.encrypt_all(&mut rng);
        let (p, ms) = Pir::new(&val_kind, &rows, cfg.val_row_bytes());
        pv = Some(p);
        pir_setup_ms.0 = ms;
        drop(rows);
        if doc_pir {
            let rows: Vec<Vec<u8>> = match cfg.doc_layout {
                DocLayout::Rows => (0..m.doc_mu()).map(|r| m.doc_row_cipher(r, &mut rng)).collect(),
                _ => m.doc_table.as_ref().unwrap().encrypt_all(&mut rng),
            };
            let (p, ms) = Pir::new(&doc_kind, &rows, cfg.doc_row_cipher_bytes());
            pd = Some(p);
            pir_setup_ms.1 = ms;
        }
        println!("[{label}] PIR setup ValMap ({val_kind}) {:.1}s DocMap ({doc_kind}) {:.1}s", pir_setup_ms.0 / 1e3, pir_setup_ms.1 / 1e3);
    }

    let mut wb = csv(
        &format!("{pre}_batches.csv"),
        "batch,appends,real_ops,dummy_ops,keywords_served,postings_flushed,due,deferred,rows_accessed,rows_distinct,owner_ms,encrypt_ms,pir_db_ms,pir_hint_ms,val_stash,val_load,val_stale",
    );
    let mut ws = csv(
        &format!("{pre}_snap.csv"),
        "appends,log_records,log_bytes,pending,pending_keywords,max_pending_age,pend_peak,val_stash,val_stash_peak,val_overflow,val_load,val_stale,doc_load,doc_virgin_rows,doc_freed_rows,doc_stash,k_overflow_batches,deferred_total,epoch",
    );
    let mut wq = csv(
        &format!("{pre}_search.csv"),
        "appends,keyword,volume,committed,pending,chunks,val_queries,doc_queries,tag_probes,pir_val_server_ms,pir_val_client_ms,pir_val_up,pir_val_down,pir_doc_server_ms,pir_doc_client_ms,pir_doc_up,pir_doc_down",
    );
    let mut we = csv(&format!("{pre}_events.csv"), "appends,event,ms,bytes");

    let intervals = a.list("clients", "100,1000,10000,100000");
    // a search fetches at most this many documents in the capped cost (all of them in the full cost)
    let doc_cap = a.u("max-docs-fetched", 100);
    let mut profiles: Vec<Profile> = intervals.iter().map(|&i| Profile::new(i, doc_cap)).collect();

    let del_frac = a.f("del-frac", 0.02);
    let upd_frac = a.f("upd-frac", 0.02);
    let docdel_frac = a.f("docdel-frac", 0.002);
    let search_every = a.u("search-every", 5000);
    let search_sample = a.u("search-sample", 10);
    let pir_search = a.u("pir-search", 2);
    let snap_every = a.u("snap-every", 5000);
    let mut refmm: HashMap<String, Vec<u32>> = mm_build.iter().cloned().collect();
    let mut live_docs: Vec<u32> = wl.docs[..wl.nb].to_vec();
    let mut deleted: HashSet<u32> = HashSet::new();

    // accounting
    let mut owner_upload: usize = 0;
    let mut val_rows_total: usize = 0;
    let mut doc_rows_total: usize = 0;
    let mut batches = 0usize;
    let mut rebuild_ms: Vec<f64> = vec![];
    let mut compactions = 0usize;
    let mut doc_compactions = 0usize;
    let mut log_peak = m.log_bytes();
    let mut last_snap = 0usize;
    let mut last_search = 0usize;
    let mut owner_ms_total = 0.0;
    let mut pir_hint_ms_total = 0.0;
    let mut stash_peak_seen = 0usize;

    let mut handle = |act: Action, owner_ms: f64, m: &mut MkseLog, rng: &mut StdRng,
                      pv: &mut Option<Pir>, pd: &mut Option<Pir>,
                      wb: &mut BufWriter<File>, we: &mut BufWriter<File>| {
        owner_ms_total += owner_ms;
        owner_upload += act.appends * m.cfg.record_bytes();
        for b in &act.batches {
            batches += 1;
            let distinct: HashSet<usize> = b.accessed.iter().copied().collect();
            val_rows_total += b.accessed.len();
            let k = match m.cfg.policy { Policy::Batched { k, .. } => k, _ => 0 };
            owner_upload += distinct.len() * m.cfg.val_row_bytes() + k * m.cfg.record_bytes() + m.val_stash_bytes();
            if m.cfg.stateless {
                // VolMap and the pending stash are re-encrypted and resent after each batch
                owner_upload += m.cfg.volmap_bytes() + m.cfg.pending_stash_bytes();
            }
            let (mut enc_ms, mut db_ms, mut hint_ms) = (0.0, 0.0, 0.0);
            if let Some(pv) = pv.as_mut() {
                let t = Instant::now();
                let rows: Vec<(usize, Vec<u8>)> = distinct.iter().map(|&p| (p, m.val.encrypt_bucket(p, rng))).collect();
                enc_ms = t.elapsed().as_secs_f64() * 1e3;
                let c = pv.update_rows(&rows);
                db_ms = c.server_db_ms;
                hint_ms = c.server_hint_ms;
                pir_hint_ms_total += hint_ms;
            }
            writeln!(
                wb,
                "{},{},{},{},{},{},{},{},{},{},{:.3},{enc_ms:.3},{db_ms:.3},{hint_ms:.3},{},{:.4},{}",
                b.index, m.total_appends, b.real_ops, b.dummy_ops, b.keywords_served, b.postings_flushed, b.due, b.deferred,
                b.accessed.len(), distinct.len(), b.owner_ms, m.val.stash_len(), m.val.load(), m.val.stale_count
            )
            .unwrap();
        }
        // DocMap writes
        let drows: Vec<usize> = if m.cfg.doc_layout == DocLayout::Rows {
            act.doc_rows_written.clone()
        } else {
            act.doc_ops.iter().flat_map(|o| o.accessed.iter().copied()).collect()
        };
        doc_rows_total += drows.len();
        owner_upload += drows.len() * m.cfg.doc_row_cipher_bytes();
        if m.cfg.doc_layout != DocLayout::Rows && !act.doc_ops.is_empty() {
            owner_upload += m.cfg.doc.stash_bytes();
        }
        // fixed DocMap layout: the padded version map is re-encrypted and resent
        owner_upload += act.vmap_refreshes * m.cfg.vmap_bytes();
        if let (Some(pd), false) = (pd.as_mut(), drows.is_empty()) {
            let rows: Vec<(usize, Vec<u8>)> = drows
                .iter()
                .map(|&r| {
                    let v = match m.cfg.doc_layout {
                        DocLayout::Rows => m.doc_row_cipher(r, rng),
                        _ => m.doc_table.as_ref().unwrap().encrypt_bucket(r, rng),
                    };
                    (r, v)
                })
                .collect();
            pd.update_rows(&rows);
        }
        for &ms in &act.rebuilds {
            rebuild_ms.push(ms);
            let mut pir_ms = 0.0;
            if let Some(p) = pv.as_mut() {
                let rows = m.val.encrypt_all(rng);
                let (np, ms2) = Pir::new(p.name(), &rows, m.cfg.val_row_bytes());
                *p = np;
                pir_ms = ms2;
            }
            owner_upload += m.val.prm.mu * m.cfg.val_row_bytes();
            writeln!(we, "{},rebuild,{:.1},{}", m.total_appends, ms + pir_ms, m.val.prm.mu * m.cfg.val_row_bytes()).unwrap();
        }
        if act.doc_compactions > 0 {
            doc_compactions += act.doc_compactions;
            if let Some(p) = pd.as_mut() {
                let rows: Vec<Vec<u8>> = match m.cfg.doc_layout {
                    DocLayout::Rows => (0..m.doc_mu()).map(|r| m.doc_row_cipher(r, rng)).collect(),
                    _ => m.doc_table.as_ref().unwrap().encrypt_all(rng),
                };
                *p = Pir::new(p.name(), &rows, m.cfg.doc_row_cipher_bytes()).0;
            }
            owner_upload += m.doc_mu() * m.cfg.doc_row_cipher_bytes();
            writeln!(we, "{},doc_compaction,,{}", m.total_appends, m.doc_mu() * m.cfg.doc_row_cipher_bytes()).unwrap();
        }
        if act.compactions > 0 {
            compactions += act.compactions;
            owner_upload += m.compacted_bytes();
            writeln!(we, "{},log_compaction,,{}", m.total_appends, m.compacted_bytes()).unwrap();
        }
        log_peak = log_peak.max(m.log_bytes());
        stash_peak_seen = stash_peak_seen.max(m.val.stash_len());
        for p in profiles.iter_mut() {
            if p.observe(m, &act) {
                // the client's search: a uniformly random keyword
                let w = &all_kw[rng.gen_range(0..all_kw.len())];
                let sr = m.search(w);
                p.sync(m, &sr);
            }
        }
    };

    let snap = |m: &MkseLog, ws: &mut BufWriter<File>| {
        writeln!(
            ws,
            "{},{},{},{},{},{},{},{},{},{},{:.4},{},{:.4},{},{},{},{},{},{}",
            m.total_appends, m.log_bytes() / m.cfg.record_bytes(), m.log_bytes(), m.pend_total, m.pending_keywords(), m.max_pend_age,
            m.pend_peak, m.val.stash_len(), m.val.stash_peak, m.val.overflow, m.val.load(), m.val.stale_count, m.doc_load(),
            m.virgin_rows(), m.doc_freed, m.doc_table.as_ref().map_or(0, |t| t.stash_len()), m.k_overflow_batches, m.deferred_total, m.epoch
        )
        .unwrap();
    };

    let do_search = |m: &MkseLog, refmm: &HashMap<String, Vec<u32>>, rng: &mut StdRng,
                     pv: &mut Option<Pir>, pd: &mut Option<Pir>, wq: &mut BufWriter<File>| {
        for k in 0..search_sample {
            let w = &all_kw[rng.gen_range(0..all_kw.len())];
            let s = m.search(w);
            let mut want = refmm.get(w).cloned().unwrap_or_default();
            want.sort_unstable();
            want.dedup();
            let mut got = s.ids.clone();
            got.sort_unstable();
            assert_eq!(got, want, "search mismatch for {w}");
            let mut cols = String::from(",,,,,,,");
            if k < pir_search {
                if let Some(pv) = pv.as_mut() {
                    let (cv, _) = pv.query_batch_w(&s.val_queries, width);
                    let dq: Vec<usize> = s.doc_queries.iter().copied().take(a.u("max-doc-queries", 1024)).collect();
                    let cd = match pd.as_mut() {
                        Some(pd) if !dq.is_empty() => pd.query_batch_w(&dq, width).0,
                        _ => Default::default(),
                    };
                    cols = format!(
                        "{:.1},{:.1},{},{},{:.1},{:.1},{},{}",
                        cv.server_ms, cv.client_prep_ms + cv.client_query_ms + cv.client_decode_ms, cv.upload_bytes, cv.download_bytes,
                        cd.server_ms, cd.client_prep_ms + cd.client_query_ms + cd.client_decode_ms, cd.upload_bytes, cd.download_bytes
                    );
                }
            }
            writeln!(
                wq,
                "{},{w},{},{},{},{},{},{},{},{cols}",
                m.total_appends, s.ids.len(), s.committed, s.pending, s.chunks_queried, s.val_queries.len(), s.doc_queries.len(), s.tag_probes
            )
            .unwrap();
        }
    };

    snap(&m, &mut ws);
    if !m.halted {
        do_search(&m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
    }
    let t_replay = Instant::now();
    // --max-appends N: stop the replay after N log appends (a prefix of the workload)
    let max_appends = a.u("max-appends", 0);
    // a structure that halted already at setup (stash overflow of its Encode) replays nothing
    let replay_docs = if m.halted { &wl.docs[..0] } else { &wl.docs[wl.nb..] };
    'replay: for &d in replay_docs {
        let len = wl.lens.get(&d).copied().unwrap_or(wl.default_len);
        for w in wl.doc_kw[&d].clone() {
            if max_appends > 0 && m.total_appends >= max_appends {
                live_docs.push(d);
                break 'replay;
            }
            let t = Instant::now();
            let act = m.insert(&w, d, len, &mut rng);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            handle(act, ms, &mut m, &mut rng, &mut pv, &mut pd, &mut wb, &mut we);
            if m.halted {
                break 'replay;
            }
            refmm.entry(w).or_default().push(d);
            let r: f64 = rng.gen();
            if r < del_frac {
                let w2 = &all_kw[rng.gen_range(0..all_kw.len())];
                if let Some(list) = refmm.get_mut(w2) {
                    if !list.is_empty() {
                        let id = list.remove(rng.gen_range(0..list.len()));
                        let t = Instant::now();
                        let act = m.delete_posting(w2, id, &mut rng);
                        let ms = t.elapsed().as_secs_f64() * 1e3;
                        handle(act, ms, &mut m, &mut rng, &mut pv, &mut pd, &mut wb, &mut we);
                        if m.halted {
                            break 'replay;
                        }
                    }
                }
            } else if r < del_frac + upd_frac && !live_docs.is_empty() {
                let id = *live_docs.choose(&mut rng).unwrap();
                let base = wl.lens.get(&id).copied().unwrap_or(wl.default_len) as f64;
                let nl = (base * rng.gen_range(0.5..1.5)).max(1.0) as usize;
                let t = Instant::now();
                let act = m.update_doc(id, nl, &mut rng);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                handle(act, ms, &mut m, &mut rng, &mut pv, &mut pd, &mut wb, &mut we);
                if m.halted {
                    break 'replay;
                }
            } else if r < del_frac + upd_frac + docdel_frac && !live_docs.is_empty() {
                let k = rng.gen_range(0..live_docs.len());
                let id = live_docs.swap_remove(k);
                let t = Instant::now();
                let act = m.delete_doc(id, &mut rng);
                let ms = t.elapsed().as_secs_f64() * 1e3;
                handle(act, ms, &mut m, &mut rng, &mut pv, &mut pd, &mut wb, &mut we);
                if m.halted {
                    break 'replay;
                }
                deleted.insert(id);
                for list in refmm.values_mut() {
                    list.retain(|x| *x != id);
                }
            }
            if m.total_appends - last_snap >= snap_every {
                last_snap = m.total_appends;
                snap(&m, &mut ws);
            }
            if m.total_appends - last_search >= search_every {
                last_search = m.total_appends;
                do_search(&m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
            }
        }
        live_docs.push(d);
    }
    let replay_s = t_replay.elapsed().as_secs_f64();
    snap(&m, &mut ws);
    if m.halted {
        // a public failure (stash overflow): the structure stopped; counted in the summary
        println!("[{label}] HALTED after {} appends: {}", m.total_appends, m.halt_reason);
    } else {
        do_search(&m, &refmm, &mut rng, &mut pv, &mut pd, &mut wq);
    }
    for w in [&mut wb, &mut ws, &mut wq, &mut we] {
        w.flush().unwrap();
    }

    // summaries
    let appends = (m.total_appends).max(1);
    let summary_path = a.s("summary", "results/design_summary.csv");
    let mut sw = append_csv(
        &summary_path,
        "label,policy,b,k,theta,age,r,major_every,client_model,val_pir,doc_pir,pend_pad_overflow,val_rows,doc_rows,doc_cap,doc_layout,evict,alpha,mu_factor,keywords,docs_setup,docs_total,appends,batches,k_overflow_batches,deferred_total,pend_peak,max_pending_age,log_peak_bytes,compacted_log_bytes,val_stash_peak,val_overflow,val_load,val_stale,doc_load,doc_compactions,log_compactions,rebuilds,rebuild_ms_mean,setup_ms,replay_s,owner_ms_per_append,owner_upload_per_append,val_rows_per_append,doc_rows_per_append,pir_hint_ms_per_batch,val_hint_bytes,doc_hint_bytes,val_delta_packed,join_bytes,halted,val_cap,val_live_max,doc_live_cap,doc_live_max,ver_cap,ver_max,vmap_pad,vmap_peak,compacted_log_records",
    );
    writeln!(
        sw,
        "{}",
        [
            label.clone(),
            policy_cols(&cfg.policy),
            cfg.major_every.to_string(),
            (if cfg.stateless { "stateless" } else { "log" }).to_string(),
            (if use_pir { val_kind.clone() } else { "-".to_string() }),
            (if doc_pir { doc_kind.clone() } else { "-".to_string() }),
            m.pend_pad_overflow.to_string(),
            m.val.prm.mu.to_string(),
            m.doc_mu().to_string(),
            doc_cap.to_string(),
            format!("{:?}", cfg.doc_layout),
            format!("{:?}", cfg.val.evict),
            cfg.val.alpha.to_string(),
            a.f("mu-factor", 2.0).to_string(),
            all_kw.len().to_string(),
            wl.nb.to_string(),
            wl.docs.len().to_string(),
            m.total_appends.to_string(),
            batches.to_string(),
            m.k_overflow_batches.to_string(),
            m.deferred_total.to_string(),
            m.pend_peak.to_string(),
            m.max_pend_age.to_string(),
            log_peak.to_string(),
            m.compacted_bytes().to_string(),
            m.val.stash_peak.to_string(),
            m.val.overflow.to_string(),
            format!("{:.4}", m.val.load()),
            m.val.stale_count.to_string(),
            format!("{:.4}", m.doc_load()),
            doc_compactions.to_string(),
            compactions.to_string(),
            rebuild_ms.len().to_string(),
            format!("{:.1}", if rebuild_ms.is_empty() { 0.0 } else { rebuild_ms.iter().sum::<f64>() / rebuild_ms.len() as f64 }),
            format!("{:.1}", setup_ms),
            format!("{:.1}", replay_s),
            format!("{:.4}", owner_ms_total / appends as f64),
            format!("{:.1}", owner_upload as f64 / appends as f64),
            format!("{:.4}", val_rows_total as f64 / appends as f64),
            format!("{:.4}", doc_rows_total as f64 / appends as f64),
            format!("{:.3}", if batches == 0 { 0.0 } else { pir_hint_ms_total / batches as f64 }),
            m.val_hint_bytes().to_string(),
            m.doc_hint_bytes().to_string(),
            m.val_delta_bytes(true).to_string(),
            (m.val_hint_bytes() + m.doc_hint_bytes() + m.compacted_bytes()).to_string(),
            (m.halted as u8).to_string(),
            cfg.val_cap.to_string(),
            m.val_live_max.to_string(),
            cfg.doc_live_cap.to_string(),
            m.doc_live_max.to_string(),
            cfg.ver_cap.to_string(),
            m.ver_max.to_string(),
            (if cfg.doc_layout == DocLayout::Fixed { cfg.vmap_pad } else { 0 }).to_string(),
            m.vmap_peak.to_string(),
            (m.compacted_bytes() / cfg.record_bytes()).to_string(),
        ]
        .join(",")
    )
    .unwrap();
    let cpath = summary_path.replace(".csv", "_clients.csv");
    let mut cw = append_csv(
        &cpath,
        "label,client_model,val_pir,doc_pir,interval,searches,doc_cap,lookup_mean,lookup_median,lookup_max,lookup_per_append,lookup_sync_share,lookup_stash_share,lookup_hint_share,lookup_query_share,capped_docs_mean,lookup_plus_capped_mean,lookup_plus_capped_median,per_doc_bytes,all_docs_mean,mean_val_queries,mean_docs,mean_doc_queries_per_doc,join_bytes,full_val_hint_refreshes",
    );
    let model = if cfg.stateless { "stateless" } else { "log" };
    for p in profiles.iter() {
        let n = p.searches.max(1) as f64;
        for (c, &(kv, kd)) in PIR_COMBOS.iter().enumerate() {
            let look = p.lookup[c] as f64;
            let lt = look.max(1.0);
            let hv = if kv == mkse::mkse_log::PirKind::Frodo { p.hint_val } else { 0 } as f64;
            let q = look - p.sync as f64 - p.stash_val as f64 - hv;
            writeln!(
                cw,
                "{label},{model},{},{},{},{},{},{:.0},{},{},{:.1},{:.3},{:.3},{:.3},{:.3},{:.0},{:.0},{},{:.0},{:.0},{:.2},{:.1},{:.2},{},{}",
                pir_name(kv), pir_name(kd), p.interval, p.searches, p.doc_cap,
                look / n, median(&p.lookup_hist[c]), p.lookup_max[c], look / appends as f64,
                p.sync as f64 / lt, p.stash_val as f64 / lt, hv / lt, q / lt,
                p.docs_capped_bytes[c] as f64 / n, (look + p.docs_capped_bytes[c] as f64) / n, median(&p.capped_hist[c]),
                p.docs_all[c] as f64 / p.docs.max(1) as f64, p.docs_all[c] as f64 / n,
                p.val_queries as f64 / n, p.docs as f64 / n, p.doc_queries as f64 / p.docs.max(1) as f64,
                m.join_bytes(kv, kd), p.full_hint_refreshes
            )
            .unwrap();
        }
    }
    println!(
        "[{label}] done: {} appends in {:.1}s, {} batches, K-overflow batches {}, pending-stash overflows {}, pend peak {}, max pending age {} batches, log peak {:.1} MB, val stash peak {} (sigma {}, overflows {}{}), val load {:.3}, rebuilds {}, compactions {}, doc compactions {}",
        m.total_appends, replay_s, batches, m.k_overflow_batches, m.pend_pad_overflow, m.pend_peak, m.max_pend_age, log_peak as f64 / 1e6,
        m.val.stash_peak, cfg.val.sigma, m.val.overflow, if m.halted { ", HALTED" } else { "" }, m.val.load(), rebuild_ms.len(), compactions, doc_compactions
    );
    println!(
        "[{label}]   public caps: ValMap Encode {} items (live max {}), DocMap live cap {} (live max {}, {} rows), snapshot {} flush + {} version records (needed max {}){}",
        cfg.val_cap, m.val_live_max, cfg.doc_live_cap, m.doc_live_max, m.doc_mu(), cfg.kw_pad, cfg.ver_cap, m.ver_max,
        if cfg.doc_layout == DocLayout::Fixed { format!(", version map {} slots (peak {} entries)", cfg.vmap_pad, m.vmap_peak) } else { String::new() }
    );
    for p in profiles.iter() {
        let mb = |x: usize| x as f64 / 1e6;
        let n = p.searches.max(1);
        println!(
            "[{label}]   client every {:>7} appends: {:>5} searches; keyword lookup median MB frodo {:.2} / spiral {:.2}; lookup + up to {} docs median MB ff {:.1} sf {:.1} ss {:.1}; full retrieval mean MB ff {:.1} ss {:.1}",
            p.interval, p.searches, mb(median(&p.lookup_hist[0])), mb(median(&p.lookup_hist[1])), p.doc_cap,
            mb(median(&p.capped_hist[0])), mb(median(&p.capped_hist[1])), mb(median(&p.capped_hist[3])),
            mb(p.lookup[0] + p.docs_all[0]) / n as f64, mb(p.lookup[3] + p.docs_all[3]) / n as f64
        );
    }
    if m.val.load() > 0.6 {
        println!("WARNING: ValMap load {:.2} is high; raise --mu-factor", m.val.load());
    }
}

// ------------------------------------------------------------------ selftest of the full design

fn selftest_design() {
    let mut rng = StdRng::seed_from_u64(11);
    for (policy, layout) in [
        ("batched", "rows"),
        ("noreuse", "rows"),
        ("stateless", "fixed"),
        ("batched", "fixed"),
        ("batched", "cuckoo"),
        ("perop", "rows"),
        ("rebuild", "rows"),
    ] {
        let mut m = HashMap::new();
        for (k, v) in [
            ("policy", if policy == "noreuse" || policy == "stateless" { "batched" } else { policy }), ("doc-layout", layout), ("synthetic", "80,600,10"), ("b", "40"), ("k", "6"),
            ("theta", "20"), ("age", "6"), ("r", "300"), ("major-every", "700"), ("bucket-bytes", "1024"),
            ("doc-row", "512"), ("default-doc-len", "1500"), ("sigma", "64"), ("doc-sigma", "64"), ("mu-factor", "3.0"),
            ("doc-spare", "0.05"),
        ] {
            m.insert(k.to_string(), v.to_string());
        }
        if policy == "noreuse" {
            m.insert("no-reuse".to_string(), "true".to_string());
            m.insert("reuse-mu-factor".to_string(), "3.0".to_string());
        }
        if policy == "stateless" {
            m.insert("stateless".to_string(), "true".to_string());
            m.insert("b".to_string(), "12".to_string());
            m.insert("theta".to_string(), "1".to_string());
            m.insert("k".to_string(), "auto".to_string());
        }
        let a = Args(m);
        let wl = workload(&a, &mut rng);
        let cfg = log_cfg(&a, &wl);
        let build: HashSet<u32> = wl.docs[..wl.nb].iter().copied().collect();
        let mm_build: Vec<(String, Vec<u32>)> = wl
            .mm_full
            .iter()
            .map(|(w, ids)| (w.clone(), ids.iter().copied().filter(|i| build.contains(i)).collect::<Vec<_>>()))
            .filter(|(_, v)| !v.is_empty())
            .collect();
        let all_kw: Vec<String> = wl.mm_full.iter().map(|(w, _)| w.clone()).collect();
        let mut s = MkseLog::setup(cfg.clone(), &mm_build, &all_kw, &wl.lens, wl.default_len, &mut rng);
        let mut refmm: HashMap<String, Vec<u32>> = mm_build.iter().cloned().collect();
        let mut live: Vec<u32> = wl.docs[..wl.nb].to_vec();
        let (mut nb, mut ncomp, mut ndoc) = (0, 0, 0);
        let check = |s: &MkseLog, refmm: &HashMap<String, Vec<u32>>| {
            for w in &all_kw {
                let mut got = s.search(w).ids;
                let mut want = refmm.get(w).cloned().unwrap_or_default();
                got.sort_unstable();
                want.sort_unstable();
                want.dedup();
                assert_eq!(got, want, "{policy}/{layout}: search mismatch for {w}");
            }
        };
        for (step, &d) in wl.docs[wl.nb..].iter().enumerate() {
            for w in wl.doc_kw[&d].clone() {
                let act = s.insert(&w, d, 1500, &mut rng);
                nb += act.batches.len();
                ncomp += act.compactions;
                ndoc += act.doc_compactions;
                refmm.entry(w).or_default().push(d);
                let r: f64 = rng.gen();
                if r < 0.1 {
                    let w2 = &all_kw[rng.gen_range(0..all_kw.len())];
                    if let Some(list) = refmm.get_mut(w2) {
                        if !list.is_empty() {
                            let id = list.remove(rng.gen_range(0..list.len()));
                            s.delete_posting(w2, id, &mut rng);
                        }
                    }
                } else if r < 0.2 && !live.is_empty() {
                    let id = *live.choose(&mut rng).unwrap();
                    let act = s.update_doc(id, rng.gen_range(200..3000), &mut rng);
                    ndoc += act.doc_compactions;
                } else if r < 0.23 && !live.is_empty() {
                    let k = rng.gen_range(0..live.len());
                    let id = live.swap_remove(k);
                    s.delete_doc(id, &mut rng);
                    for list in refmm.values_mut() {
                        list.retain(|x| *x != id);
                    }
                }
            }
            live.push(d);
            if step % 20 == 0 {
                check(&s, &refmm);
            }
        }
        check(&s, &refmm);
        s.val.check_invariants();
        if let Some(t) = s.doc_table.as_ref() {
            t.check_invariants();
        }
        println!(
            "ok  design {policy}/{layout}: {} appends, {} batches, {} compactions ({} DocMap), K-overflow batches {}, pend peak {}, val stash {}",
            s.total_appends, nb, ncomp, ndoc, s.k_overflow_batches, s.pend_peak, s.val.stash_len()
        );
    }
}

// ------------------------------------------------------------------ report

fn read_csv(path: &Path) -> Option<(Vec<String>, Vec<Vec<String>>)> {
    let s = std::fs::read_to_string(path).ok()?;
    let mut lines = s.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines.next()?.split(',').map(|x| x.to_string()).collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split(',').map(|x| x.to_string()).collect::<Vec<_>>())
        .filter(|r| r.len() == header.len() && r[0] != header[0])
        .collect();
    Some((header, rows))
}

/// Keep the last row for each key (a rerun appends new rows for the same run).
fn dedup_last(h: &[String], rows: Vec<Vec<String>>, keys: &[&str]) -> Vec<Vec<String>> {
    let ix: Vec<usize> = keys.iter().filter_map(|k| h.iter().position(|x| x == k)).collect();
    let key = |r: &Vec<String>| ix.iter().map(|&i| r[i].as_str()).collect::<Vec<_>>().join("\u{1}");
    let mut last = std::collections::HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        last.insert(key(r), i);
    }
    rows.iter().enumerate().filter(|(i, r)| last[&key(r)] == *i).map(|(_, r)| r.clone()).collect()
}

fn col(h: &[String], name: &str) -> usize {
    h.iter().position(|x| x == name).unwrap_or_else(|| panic!("column {name} missing"))
}

fn fnum(x: &str) -> f64 {
    x.parse().unwrap_or(f64::NAN)
}

fn human(b: f64) -> String {
    if b.is_nan() {
        "-".into()
    } else if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.1} KB", b / 1e3)
    } else {
        format!("{:.0} B", b)
    }
}

/// Collects every CSV under --dir into one markdown report.
fn report(a: &Args) {
    let dir = a.s("dir", "results");
    let out = a.s("out", &format!("{dir}/REPORT.md"));
    let mut md = String::new();
    md.push_str("# Evaluation report\n\nGenerated by `uokvs_bench report` from the CSVs in this directory.\n\n");

    // design comparison
    let sp = Path::new(&dir).join("design_summary.csv");
    if let Some((h, rows)) = read_csv(&sp) {
        let rows = dedup_last(&h, rows, &["label"]);
        md.push_str("## Write path: schedules and DocMap layouts\n\n");
        md.push_str("| run | policy | b | k | θ | r | DocMap | α | appends | K-overflow batches | pending peak | log peak | ValMap rows / append | DocMap rows / append | owner upload / append | owner ms / append | hint update ms / batch | rebuilds (mean ms) | stash peak | ValMap load | halted (stash overflow) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for r in &rows {
            let g = |n: &str| r[col(&h, n)].clone();
            md.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {} | {:.3} | {} | {} ({}) | {} | {} | {} |\n",
                g("label"), g("policy"), g("b"), g("k"), g("theta"), g("r"), g("doc_layout"), g("alpha"), g("appends"),
                g("k_overflow_batches"), g("pend_peak"), human(fnum(&g("log_peak_bytes"))), fnum(&g("val_rows_per_append")),
                fnum(&g("doc_rows_per_append")), human(fnum(&g("owner_upload_per_append"))), fnum(&g("owner_ms_per_append")),
                g("pir_hint_ms_per_batch"), g("rebuilds"), g("rebuild_ms_mean"), g("val_stash_peak"), g("val_load"),
                h.iter().position(|x| x == "halted").map_or("-".to_string(), |i| if r[i] == "1" { "**yes**".to_string() } else { "no".to_string() })
            ));
        }
        md.push('\n');
    }
    let cp = Path::new(&dir).join("design_summary_clients.csv");
    if let Some((h, rows)) = read_csv(&cp).filter(|(h, _)| h.contains(&"lookup_mean".to_string())) {
        let rows = dedup_last(&h, rows, &["label", "client_model", "val_pir", "doc_pir", "interval", "doc_cap"]);
        let mut intervals: Vec<usize> = rows.iter().map(|r| fnum(&r[col(&h, "interval")]) as usize).collect();
        intervals.sort_unstable();
        intervals.dedup();
        let key = |r: &Vec<String>| (r[0].clone(), r[col(&h, "client_model")].clone(), r[col(&h, "val_pir")].clone(), r[col(&h, "doc_pir")].clone());
        let mut keys: Vec<(String, String, String, String)> = Vec::new();
        for r in &rows {
            let k = key(r);
            if !keys.contains(&k) {
                keys.push(k);
            }
        }
        let cap = rows.first().map(|r| r[col(&h, "doc_cap")].clone()).unwrap_or_default();
        let cell = |k: &(String, String, String, String), i: usize, colname: &str| -> f64 {
            rows.iter()
                .find(|r| &key(r) == k && fnum(&r[col(&h, "interval")]) as usize == i && fnum(&r[col(&h, "searches")]) > 0.0)
                .map(|r| fnum(&r[col(&h, colname)]))
                .unwrap_or(f64::NAN)
        };
        md.push_str(&format!(
            "## Client costs (paired PIRs)\n\nA client searches every `interval` appends for a uniformly random keyword and only then refreshes. `log` clients copy the append-only log; `stateless` clients download VolMap and the pending stash. FrodoPIR adds hint deltas or a fresh hint, whichever is smaller; Spiral has no hint. The `run` label names the replay; the PIR columns give the pairing being costed (real-PIR runs measure server time for the pairing in their label).\n\n- **Keyword lookup** = sync + ValMap stash + ValMap hint refresh + ValMap queries and answers. This is the part compared with prior MKSE.\n- **Documents** = DocMap stash + DocMap hint refresh + DocMap queries and answers, for at most {cap} documents per search (the capped cost) or for all matching documents (the full cost).\n\n"
        ));
        let tables: [(&str, &str, bool); 6] = [
            ("Keyword lookup per search, median", "lookup_median", false),
            ("Keyword lookup per search, mean", "lookup_mean", false),
            ("Keyword lookup per append (mean)", "lookup_per_append", false),
            ("Lookup + documents (capped) per search, median", "lookup_plus_capped_median", true),
            ("Lookup + documents (capped) per search, mean", "lookup_plus_capped_mean", true),
            ("Document retrieval per document fetched (mean)", "per_doc_bytes", true),
        ];
        for (title, colname, docs) in tables {
            md.push_str(&format!("### {title}\n\n| run | clients | ValMap PIR |{}", if docs { " DocMap PIR |" } else { "" }));
            for i in &intervals {
                md.push_str(&format!(" every {i} |"));
            }
            md.push_str(if docs { "\n|---|---|---|---|" } else { "\n|---|---|---|" });
            for _ in &intervals {
                md.push_str("---|");
            }
            md.push('\n');
            let mut seen: Vec<(String, String, String)> = Vec::new();
            for k in &keys {
                // the lookup does not depend on the DocMap PIR: one row per ValMap PIR
                if !docs {
                    let kk = (k.0.clone(), k.1.clone(), k.2.clone());
                    if seen.contains(&kk) {
                        continue;
                    }
                    seen.push(kk);
                }
                md.push_str(&format!("| {} | {} | {} |", k.0, k.1, k.2));
                if docs {
                    md.push_str(&format!(" {} |", k.3));
                }
                for &i in &intervals {
                    md.push_str(&format!(" {} |", human(cell(k, i, colname))));
                }
                md.push('\n');
            }
            md.push('\n');
        }
        md.push_str("### Full retrieval (every matching document), mean per search\n\nFor reference only: Enron keywords match up to 10^5 documents.\n\n| run | clients | ValMap PIR | DocMap PIR |");
        for i in &intervals {
            md.push_str(&format!(" every {i} |"));
        }
        md.push_str("\n|---|---|---|---|");
        for _ in &intervals {
            md.push_str("---|");
        }
        md.push('\n');
        for k in &keys {
            md.push_str(&format!("| {} | {} | {} | {} |", k.0, k.1, k.2, k.3));
            for &i in &intervals {
                md.push_str(&format!(" {} |", human(cell(k, i, "lookup_mean") + cell(k, i, "all_docs_mean"))));
            }
            md.push('\n');
        }
        md.push('\n');
        for (title, colname) in [
            ("Cheapest keyword lookup per search interval (median)", "lookup_median"),
            ("Cheapest lookup + capped documents per search interval (median)", "lookup_plus_capped_median"),
        ] {
            md.push_str(&format!("### {title}\n\n| interval | run | clients | ValMap PIR | DocMap PIR | per search |\n|---|---|---|---|---|---|\n"));
            for &i in &intervals {
                let best = keys
                    .iter()
                    .map(|k| (cell(k, i, colname), k))
                    .filter(|x| !x.0.is_nan())
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                if let Some((v, k)) = best {
                    md.push_str(&format!("| {i} | {} | {} | {} | {} | {} |\n", k.0, k.1, k.2, k.3, human(v)));
                }
            }
            md.push('\n');
        }
        md.push_str("### Queries per search and join cost (one-time client download)\n\n| run | clients | ValMap PIR | DocMap PIR | ValMap queries | documents matched | DocMap queries per document | join |\n|---|---|---|---|---|---|---|---|\n");
        for k in &keys {
            if let Some(r) = rows.iter().find(|r| &key(r) == k && fnum(&r[col(&h, "searches")]) > 0.0) {
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.1} | {:.1} | {:.2} | {} |\n",
                    k.0, k.1, k.2, k.3, fnum(&r[col(&h, "mean_val_queries")]), fnum(&r[col(&h, "mean_docs")]),
                    fnum(&r[col(&h, "mean_doc_queries_per_doc")]), human(fnum(&r[col(&h, "join_bytes")]))
                ));
            }
        }
        md.push('\n');
    } else if cp.exists() {
        md.push_str("## Client costs\n\n`design_summary_clients.csv` has the old column layout (before the lookup / document split); rerun the `write` and `paired` stages to regenerate it.\n\n");
    }

    // search costs (queries per search) from the design runs
    let ddir = Path::new(&dir).join("design");
    if let Ok(rd) = std::fs::read_dir(&ddir) {
        let mut files: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.to_string_lossy().ends_with("_search.csv")).collect();
        files.sort();
        if !files.is_empty() {
            md.push_str("## Searches (sampled during the replay)\n\n| run | searches | mean volume | mean ValMap queries | mean DocMap queries | mean tag probes | PIR ValMap server ms (mean) | PIR DocMap server ms (mean) |\n|---|---|---|---|---|---|---|---|\n");
            for f in files {
                if let Some((h, rows)) = read_csv(&f) {
                    let mean = |n: &str| {
                        let v: Vec<f64> = rows.iter().map(|r| fnum(&r[col(&h, n)])).filter(|x| !x.is_nan()).collect();
                        if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
                    };
                    let name = f.file_name().unwrap().to_string_lossy().replace("_search.csv", "");
                    md.push_str(&format!(
                        "| {name} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
                        rows.len(), mean("volume"), mean("val_queries"), mean("doc_queries"), mean("tag_probes"),
                        mean("pir_val_server_ms"), mean("pir_doc_server_ms")
                    ));
                }
            }
            md.push('\n');
        }
    }

    // stash
    let sdir = Path::new(&dir).join("stash");
    if let Ok(rd) = std::fs::read_dir(&sdir) {
        let mut files: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "csv")).collect();
        files.sort();
        if !files.is_empty() {
            md.push_str("## Stash vs load (final checkpoint of each run)\n\n| run | evict | disc | p | α | joint | n | load | stash | stash / n |\n|---|---|---|---|---|---|---|---|---|---|\n");
            for f in files {
                if let Some((h, rows)) = read_csv(&f) {
                    if let Some(r) = rows.last() {
                        let g = |n: &str| r[col(&h, n)].clone();
                        md.push_str(&format!(
                            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                            f.file_stem().unwrap().to_string_lossy(), g("evict"), g("disc"), g("p"), g("alpha"), g("joint_frac"),
                            g("inserted"), g("load"), g("stash"), g("stash_frac")
                        ));
                    }
                }
            }
            md.push('\n');
        }
    }

    // churn
    let cdir = Path::new(&dir).join("churn");
    if let Ok(rd) = std::fs::read_dir(&cdir) {
        let mut files: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "csv")).collect();
        files.sort();
        if !files.is_empty() {
            md.push_str("## Stash under churn (constant load, 4n operations after the fill)\n\n`replace` frees a key and inserts a new one (versioned keys, as in `mkse_log`); `update` rewrites the same key by an invalidation-mode update. Targets uniform (z0) or Zipf(1) (z1); `rebuild` = re-insert every live key into a fresh table every R operations (0 = never). Stash as a fraction of the n live keys.\n\n| run | evict | disc | α | load target | mode | zipf | rebuild | stash after fill | stash at n ops | stash at 4n ops | max stash | final load |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
            for f in files {
                let Some((h, rows)) = read_csv(&f) else { continue };
                let (Some(first), Some(last)) = (rows.first(), rows.last()) else { continue };
                let g = |r: &Vec<String>, n: &str| r[col(&h, n)].clone();
                let nn = fnum(&g(last, "n")).max(1.0);
                let frac = |x: &str| format!("{:.2e}", fnum(x) / nn);
                let at_n = rows.iter().find(|r| fnum(&g(r, "ops")) >= nn).map(|r| frac(&g(r, "stash"))).unwrap_or_default();
                md.push_str(&format!(
                    "| {} | {} | {} | {} | {:.2} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                    f.file_stem().unwrap().to_string_lossy(), g(last, "evict"), g(last, "disc"), g(last, "alpha"),
                    nn / fnum(&g(last, "mu")), g(last, "mode"), g(last, "zipf"), g(last, "rebuild_every"),
                    frac(&g(first, "stash")), at_n, frac(&g(last, "stash")), frac(&g(last, "stash_max")), g(last, "load")
                ));
            }
            md.push('\n');
        }
    }

    // PIR microbenchmarks
    let pdir = Path::new(&dir).join("pir");
    if let Ok(rd) = std::fs::read_dir(&pdir) {
        let mut files: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "csv")).collect();
        files.sort();
        for f in files {
            let Some((h, rows)) = read_csv(&f) else { continue };
            if !h.contains(&"metric".to_string()) || !h.contains(&"param".to_string()) {
                continue;
            }
            md.push_str(&format!("## PIR: {}\n\n| scheme | rows | row bytes | metric | param | mean | n |\n|---|---|---|---|---|---|---|\n", f.file_stem().unwrap().to_string_lossy()));
            let mut keys: Vec<(String, String, String, String, String)> = Vec::new();
            let mut acc: HashMap<(String, String, String, String, String), (f64, usize)> = HashMap::new();
            for r in &rows {
                let k = (r[0].clone(), r[1].clone(), r[2].clone(), r[col(&h, "metric")].clone(), r[col(&h, "param")].clone());
                if !acc.contains_key(&k) {
                    keys.push(k.clone());
                }
                let e = acc.entry(k).or_insert((0.0, 0));
                e.0 += fnum(&r[h.len() - 1]);
                e.1 += 1;
            }
            for k in keys {
                let (s, n) = acc[&k];
                md.push_str(&format!("| {} | {} | {} | {} | {} | {:.1} | {} |\n", k.0, k.1, k.2, k.3, k.4, s / n as f64, n));
            }
            md.push('\n');
        }
    }
    for extra in ["pir_compare.txt", "ypir.txt", "spiral.txt"] {
        if let Ok(t) = std::fs::read_to_string(Path::new(&dir).join(extra)) {
            md.push_str(&format!("## {extra}\n\n```\n{}\n```\n\n", t.trim()));
        }
    }
    std::fs::write(&out, md).expect("write report");
    println!("wrote {out}");
}
