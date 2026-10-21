//! ypir_bench check
//! ypir_bench bench --records N --record-bytes B [--per-row K|auto] [--seeds S]
//!                  [--batches 1,8,16,32] [--updates 1,100,1000] [--trials 3] [--ref]
use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use spiral_rs::aligned_memory::AlignedMemory64;
use ypir_fast::*;

fn rec(idx: usize, len: usize, salt: u64) -> Vec<u8> {
    let mut r = ChaCha20Rng::seed_from_u64((idx as u64) ^ (salt << 40));
    (0..len).map(|_| r.gen()).collect()
}

fn rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1).and_then(|x| x.parse::<f64>().ok()))
        })
        .map(|kb| kb / 1e6)
        .unwrap_or(0.0)
}

fn arg<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter().position(|a| a == k).map(|i| args[i + 1].as_str())
}
fn list(s: &str) -> Vec<usize> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
}

/// Build `nq` queries for random records: clients hold `seeds` queries each (seed = slot).
fn make_queries(
    srv: &YpirServer,
    nq: usize,
    rng: &mut ChaCha20Rng,
) -> (Vec<YpirClient>, Vec<(usize, usize, usize, AlignedMemory64)>, f64) {
    let s = srv.n_seeds();
    let mut clients = Vec::new();
    let mut qs = Vec::new();
    let t = Instant::now();
    for q in 0..nq {
        if q % s == 0 {
            clients.push(YpirClient::new(srv.shape, srv.params));
        }
        let c = clients.len() - 1;
        let j = q % s;
        let idx = rng.gen_range(0..srv.shape.records);
        qs.push((c, j, idx, clients[c].query(srv.seed_idx(j), idx)));
    }
    (clients, qs, t.elapsed().as_secs_f64() * 1e3 / nq as f64)
}

fn run_batch(
    srv: &YpirServer,
    clients: &[YpirClient],
    qs: &[(usize, usize, usize, AlignedMemory64)],
) -> (Vec<Vec<Vec<u8>>>, AnswerTiming) {
    let batch: Vec<(usize, &[u64], &[_])> =
        qs.iter().map(|(c, j, _, q)| (*j, q.as_slice(), clients[*c].pp.as_slice())).collect();
    srv.answer(&batch)
}

fn check() {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    for &(records, bytes, per_row, seeds) in
        &[(3000usize, 8192usize, 2usize, 2usize), (5000, 1059, 4, 1), (2048, 2048, 1, 3)]
    {
        let shape = Shape::new(records, bytes, per_row);
        let mut salt = vec![0u64; records];
        let mut srv = YpirServer::new(shape, seeds, |i| rec(i, bytes, 0));
        println!("check {} seeds={} setup {:?}", shape.label(), seeds, srv.setup);
        for (i, s) in [(0usize, 0u64), (records - 1, 0), (records / 2, 0)] {
            assert_eq!(srv.record(i), rec(i, bytes, s), "record readback");
        }
        let (clients, qs, _) = make_queries(&srv, 5, &mut rng);
        let (resp, t) = run_batch(&srv, &clients, &qs);
        for (k, (c, _, idx, _)) in qs.iter().enumerate() {
            assert_eq!(clients[*c].decode(*idx, &resp[k]), rec(*idx, bytes, salt[*idx]), "batch query {}", k);
        }
        // the reference path gives the same answers (seed 0)
        let rst = srv.offline_reference();
        let (r0, _) = srv.answer_reference(&rst, qs[0].3.as_slice(), &clients[qs[0].0].pp);
        assert_eq!(clients[qs[0].0].decode(qs[0].2, &r0), rec(qs[0].2, bytes, 0));
        // prepared packing (keys sent ahead of the query) gives the same answers
        let reqs: Vec<(usize, &[_])> = qs.iter().map(|(c, j, _, _)| (*j, clients[*c].pp.as_slice())).collect();
        let prep = srv.prepare_many(&reqs);
        let pq: Vec<(&Prepared, &[u64])> = prep.iter().zip(&qs).map(|(p, q)| (p, q.3.as_slice())).collect();
        let (resp2, t2) = srv.answer_prepared(&pq);
        for (k, (c, _, idx, _)) in qs.iter().enumerate() {
            assert_eq!(clients[*c].decode(*idx, &resp2[k]), rec(*idx, bytes, 0), "prepared query {}", k);
        }
        println!("  batch of 5 ok ({:.1} ms; prepared online {:.1} ms)", t.total_ms, t2.total_ms);
        // updates, including two records of one row and a repeated record
        let mut ups: Vec<usize> = (0..12).map(|_| rng.gen_range(0..records)).collect();
        ups.push(ups[0]);
        if per_row > 1 {
            ups.push((ups[1] / per_row) * per_row + (ups[1] + 1) % per_row);
        }
        let mut batch = Vec::new();
        for (n, &i) in ups.iter().enumerate() {
            salt[i] = 100 + n as u64;
            batch.push((i, rec(i, bytes, salt[i])));
        }
        assert!(srv.precomp_matches_reference(0, 0), "parallel precomputation != reference");
        let ut = srv.update_records(&batch);
        for j in 0..seeds {
            assert!(srv.hint_matches_full(j), "incremental hint != full hint (seed {})", j);
        }
        assert!(srv.precomp_matches_reference(seeds - 1, shape.instances - 1), "precomputation after update");
        for &i in &ups {
            assert_eq!(srv.record(i), rec(i, bytes, salt[i]));
        }
        // query the updated records
        let s = srv.n_seeds();
        let mut cl = Vec::new();
        let mut qs2 = Vec::new();
        for (q, &i) in ups.iter().take(6).enumerate() {
            if q % s == 0 {
                cl.push(YpirClient::new(shape, srv.params));
            }
            let c = cl.len() - 1;
            qs2.push((c, q % s, i, cl[c].query(srv.seed_idx(q % s), i)));
        }
        let (resp, _) = run_batch(&srv, &cl, &qs2);
        for (k, (c, _, idx, _)) in qs2.iter().enumerate() {
            assert_eq!(cl[*c].decode(*idx, &resp[k]), rec(*idx, bytes, salt[*idx]), "updated query {}", k);
        }
        println!("  {} updates ok: {:?}", ups.len(), ut);
    }
    println!("ypir_fast check ok");
}

fn bench(args: &[String]) {
    let records: usize = arg(args, "--records").unwrap().parse().unwrap();
    let bytes: usize = arg(args, "--record-bytes").unwrap().parse().unwrap();
    let shape = match arg(args, "--per-row").unwrap_or("auto") {
        "auto" => Shape::auto(records, bytes),
        k => Shape::new(records, bytes, k.parse().unwrap()),
    };
    let seeds: usize = arg(args, "--seeds").unwrap_or("1").parse().unwrap();
    let batches = list(arg(args, "--batches").unwrap_or("1,8,16,32"));
    let updates = list(arg(args, "--updates").unwrap_or("1,100,1000"));
    let trials: usize = arg(args, "--trials").unwrap_or("3").parse().unwrap();
    let reference = args.iter().any(|a| a == "--ref");
    let threads = rayon::current_num_threads();
    let head = format!(
        "shape={},records={},record_bytes={},per_row={},rows={},instances={},seeds={},threads={}",
        shape.label(),
        records,
        bytes,
        shape.per_row,
        shape.rows,
        shape.instances,
        seeds,
        threads
    );
    let t = Instant::now();
    let mut srv = YpirServer::new(shape, seeds, |i| rec(i, bytes, 0));
    println!(
        "impl=ypir_fast,{},setup_ms={:.0},db_ms={:.0},hint_ms={:.0},precomp_ms={:.0},db_gb={:.2},state_gb_per_seed={:.2},rss_gb={:.2},query_bytes={},response_bytes={},pp_bytes={}",
        head,
        t.elapsed().as_secs_f64() * 1e3,
        srv.setup.db_ms,
        srv.setup.hint_ms,
        srv.setup.precomp_ms,
        shape.db_bytes() as f64 / 1e9,
        srv.bytes_per_seed() as f64 / 1e9,
        rss_gb(),
        shape.query_bytes(),
        shape.response_bytes(),
        PP_BYTES
    );
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    for &b in &batches {
        let (clients, qs, gen_ms) = make_queries(&srv, b, &mut rng);
        let mut best = AnswerTiming { total_ms: f64::MAX, ..Default::default() };
        for trial in 0..trials + 1 {
            let (resp, t) = run_batch(&srv, &clients, &qs);
            if trial == 0 {
                let (c, _, idx, _) = &qs[0];
                assert_eq!(clients[*c].decode(*idx, &resp[0]), rec(*idx, bytes, 0));
            } else if t.total_ms < best.total_ms {
                best = t;
            }
        }
        let reqs: Vec<(usize, &[_])> = qs.iter().map(|(c, j, _, _)| (*j, clients[*c].pp.as_slice())).collect();
        let tp = Instant::now();
        let prep = srv.prepare_many(&reqs);
        let prep_ms = tp.elapsed().as_secs_f64() * 1e3;
        let pq: Vec<(&Prepared, &[u64])> = prep.iter().zip(&qs).map(|(p, q)| (p, q.3.as_slice())).collect();
        let mut online = f64::MAX;
        for trial in 0..trials + 1 {
            let (resp, t) = srv.answer_prepared(&pq);
            if trial == 0 {
                let (c, _, idx, _) = &qs[0];
                assert_eq!(clients[*c].decode(*idx, &resp[0]), rec(*idx, bytes, 0));
            } else {
                online = online.min(t.total_ms);
            }
        }
        let up = shape.query_bytes() as f64 + PP_BYTES as f64 / seeds.min(b) as f64;
        println!(
            "impl=ypir_fast,{},batch={},per_query_ms={:.2},total_ms={:.1},first_ms={:.1},pack_ms={:.1},client_query_ms={:.1},upload_per_query={:.0},download_per_query={},prepare_ms={:.1},online_prepared_ms={:.1}",
            head, b, best.total_ms / b as f64, best.total_ms, best.first_ms, best.pack_ms, gen_ms, up, shape.response_bytes(), prep_ms, online
        );
    }
    for &u in &updates {
        let ups: Vec<(usize, Vec<u8>)> = (0..u)
            .map(|n| {
                let i = rng.gen_range(0..records);
                (i, rec(i, bytes, 1000 + n as u64))
            })
            .collect();
        let ut = srv.update_records(&ups);
        println!(
            "impl=ypir_fast,{},updates={},rows={},blocks={},update_ms={:.1},db_ms={:.1},hint_ms={:.1},precomp_ms={:.1}",
            head, u, ut.rows, ut.blocks, ut.total_ms, ut.db_ms, ut.hint_ms, ut.precomp_ms
        );
    }
    if reference {
        let (clients, qs, _) = make_queries(&srv, 1, &mut rng);
        let mut best = AnswerTiming { total_ms: f64::MAX, ..Default::default() };
        let rst = srv.offline_reference();
        for _ in 0..trials {
            let (_, t) = srv.answer_reference(&rst, qs[0].3.as_slice(), &clients[0].pp);
            if t.total_ms < best.total_ms {
                best = t;
            }
        }
        println!(
            "impl=ypir_ref,{},batch=1,per_query_ms={:.2},first_ms={:.1},pack_ms={:.1}",
            head, best.total_ms, best.first_ms, best.pack_ms
        );
        let (h, p) = (rst.hint_ms, rst.precomp_ms);
        println!("impl=ypir_ref,{},offline_ms={:.0},hint_ms={:.0},precomp_ms={:.0}", head, h + p, h, p);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("check") => check(),
        Some("bench") => bench(&args),
        Some("shapes") => {
            let records: usize = arg(&args, "--records").unwrap().parse().unwrap();
            let bytes: usize = arg(&args, "--record-bytes").unwrap().parse().unwrap();
            for k in 1..=64 {
                let s = Shape::new(records, bytes, k);
                println!(
                    "per_row={} rows={} instances={} query={} response={} sum={}",
                    k,
                    s.rows,
                    s.instances,
                    s.query_bytes(),
                    s.response_bytes(),
                    s.query_bytes() + s.response_bytes()
                );
            }
            let a = Shape::auto(records, bytes);
            println!("auto: {}", a.label());
        }
        _ => eprintln!("usage: ypir_bench check | bench --records N --record-bytes B [...] | shapes --records N --record-bytes B"),
    }
}
