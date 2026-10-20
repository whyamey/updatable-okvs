//! FrodoPIR fork vs upstream on the operations UOKVS needs.
//!
//! cargo run --release -- --rows 65536 --elem-bytes 8192 --update-rows 10,21 --batches 4,16,64 --out pir_compare.csv
//!
//! * setup:   database construction + hint (A * D)
//! * update:  refreshing the hint after k changed rows; upstream has no
//!            incremental update and must recompute the hint
//! * answer:  server time for a batch of k queries (upstream answers one at a time)
//! * prep:    client per-query preprocessing (s*A and s*hint)

use rand::{Rng, SeedableRng};
use std::io::Write;
use std::time::Instant;

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| d.to_string())
}
fn list(s: &str) -> Vec<usize> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()
}

const DIM: usize = 1288;
const PT: usize = 9;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rows: usize = arg(&args, "--rows", "16384").parse().unwrap();
    let eb: usize = arg(&args, "--elem-bytes", "8192").parse().unwrap();
    let updates = list(&arg(&args, "--update-rows", "10,21"));
    let batches = list(&arg(&args, "--batches", "4,16,64"));
    let skip_up_update = args.iter().any(|a| a == "--skip-upstream-rebuild");
    let out = arg(&args, "--out", "pir_compare.csv");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).unwrap());
    writeln!(w, "impl,rows,elem_bytes,metric,param,ms").unwrap();
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let elems: Vec<String> = (0..rows)
        .map(|_| {
            let mut v = vec![0u8; eb];
            rng.fill(&mut v[..]);
            base64::encode(v)
        })
        .collect();
    let bits = eb * 8;
    let rec = |w: &mut std::io::BufWriter<std::fs::File>, imp: &str, m: &str, p: usize, ms: f64| {
        writeln!(w, "{imp},{rows},{eb},{m},{p},{ms:.2}").unwrap();
        println!("{imp:9} {m:22} {p:6} {ms:12.2} ms");
    };

    // ------------------------------------------------------------ upstream
    {
        use frodo_up::api::{CommonParams, QueryParams, Response, Shard};
        let t = Instant::now();
        let shard = Shard::from_base64_strings(&elems, DIM, rows, bits, PT).unwrap();
        let setup = t.elapsed().as_secs_f64() * 1e3;
        rec(&mut w, "upstream", "setup", 0, setup);
        for &k in &updates {
            // no incremental hint update upstream: rebuilding the shard is the only option
            let ms = if skip_up_update {
                setup
            } else {
                let mut e2 = elems.clone();
                for _ in 0..k {
                    let i = rng.gen_range(0..rows);
                    let mut v = vec![0u8; eb];
                    rng.fill(&mut v[..]);
                    e2[i] = base64::encode(v);
                }
                let t = Instant::now();
                let _s2 = Shard::from_base64_strings(&e2, DIM, rows, bits, PT).unwrap();
                t.elapsed().as_secs_f64() * 1e3
            };
            rec(&mut w, "upstream", "hint_update", k, ms);
        }
        let cp = CommonParams::from(shard.get_base_params());
        for &k in &batches {
            let mut prep = 0.0;
            let mut ans = 0.0;
            for _ in 0..k {
                let t = Instant::now();
                let mut qp = QueryParams::new(&cp, shard.get_base_params()).unwrap();
                prep += t.elapsed().as_secs_f64() * 1e3;
                let q = loop {
                    if let Ok(q) = qp.generate_query(rng.gen_range(0..rows)) {
                        break q;
                    }
                    qp = QueryParams::new(&cp, shard.get_base_params()).unwrap();
                };
                let t = Instant::now();
                let r = shard.respond(&q).unwrap();
                ans += t.elapsed().as_secs_f64() * 1e3;
                let _resp: Response = bincode::deserialize(&r).unwrap();
            }
            rec(&mut w, "upstream", "client_prep", k, prep);
            rec(&mut w, "upstream", "answer", k, ans);
        }
    }

    // ------------------------------------------------------------ fork
    {
        use frodo_fork::api::{BatchedQueryParams8, QueryParams};
        use frodo_fork::db::{BaseParams, CommonParams, Database};
        let t = Instant::now();
        let mut db = Database::new(&elems, rows, bits, PT).unwrap();
        let mut bp = BaseParams::new(&db, DIM);
        rec(&mut w, "fork", "setup", 0, t.elapsed().as_secs_f64() * 1e3);
        for &k in &updates {
            let mut pend = Vec::new();
            let t = Instant::now();
            for _ in 0..k {
                let i = rng.gen_range(0..rows);
                let mut v = vec![0u8; eb];
                rng.fill(&mut v[..]);
                let old = db.update_row_from_bytes(i, &v);
                pend.push((i, old, db.get_row(i)));
            }
            bp.update_hints_batched(&pend);
            rec(&mut w, "fork", "hint_update", k, t.elapsed().as_secs_f64() * 1e3);
        }
        let cp = CommonParams::from(&bp);
        for &k in &batches {
            let mut prep = 0.0;
            let mut ans = 0.0;
            let mut left = k;
            while left > 0 {
                let n = left.min(8);
                left -= n;
                let t = Instant::now();
                let mut b = BatchedQueryParams8::new(&cp, &bp).unwrap();
                let mut qs = Vec::new();
                for j in 0..8 {
                    let mut qp: QueryParams = b.extract_query_params(j).unwrap();
                    let q = loop {
                        if let Ok(q) = qp.generate_query(rng.gen_range(0..rows)) {
                            break q;
                        }
                        qp = QueryParams::new(&cp, &bp).unwrap();
                    };
                    qs.push(q);
                }
                prep += t.elapsed().as_secs_f64() * 1e3;
                let inter: Vec<[u32; 8]> = (0..rows)
                    .map(|r| {
                        let mut v = [0u32; 8];
                        for j in 0..8 {
                            v[j] = qs[j].as_slice()[r];
                        }
                        v
                    })
                    .collect();
                let t = Instant::now();
                let _res = db.eval_batched_8(&inter);
                ans += t.elapsed().as_secs_f64() * 1e3;
            }
            rec(&mut w, "fork", "client_prep", k, prep);
            rec(&mut w, "fork", "answer", k, ans);
        }
    }
    w.flush().unwrap();
    println!("wrote {out}");
}
