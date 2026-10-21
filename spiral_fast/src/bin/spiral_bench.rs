//! spiral_bench check            correctness: byte-identical responses vs the Blyss server
//! spiral_bench bench N,NU1,NU2 [--impl blyss|fast] [--batches 1,8,32] [--threads T]
use rand::{rngs::StdRng, Rng, RngCore, SeedableRng};
use spiral_fast::*;
use spiral_rs::{arith::*, client::*, params::*};
use spiral_server::db::loading::update_item;
use spiral_server::db::sparse_db::SparseDb;
use spiral_server::server::process_query;
use std::time::Instant;

fn random_item(params: &Params, rng: &mut impl RngCore) -> Vec<u8> {
    let mut v = vec![0u8; params.instances * params.n * params.n * params.bytes_per_chunk()];
    rng.fill_bytes(&mut v);
    v
}

/// The record stored at `idx` in the fast benchmark (regenerated to check answers).
fn item_for(params: &Params, idx: usize) -> Vec<u8> {
    random_item(params, &mut StdRng::seed_from_u64(0x5eed_0000 ^ idx as u64))
}

fn check() {
    let mut rng = rand::thread_rng();
    for (n, nu1, nu2) in [(2, 9, 3), (2, 9, 5), (1, 9, 6), (2, 8, 4)] {
        let params = table_params(n, nu1, nu2);
        let items = params.num_items();
        let mut client = Client::init(&params);
        let pp = client.generate_keys();
        let mut sparse = SparseDb::new();
        let mut dense = DenseDb::zero(&params);
        let mut kept = Vec::new();
        for idx in 0..items {
            let item = random_item(&params, &mut rng);
            let mut body = (idx as u32).to_be_bytes().to_vec();
            body.extend_from_slice(&item);
            update_item(&params, &body, &mut sparse).unwrap();
            dense.update_item(&params, idx, &item);
            kept.push(item);
        }
        let targets: Vec<usize> = (0..5).map(|_| rng.gen_range(0..items)).collect();
        let queries: Vec<Query> = targets.iter().map(|&t| client.generate_query(t)).collect();
        let (fast, _) = process_queries(&params, &pp, &queries, &dense);
        for (q, &t) in queries.iter().zip(&targets) {
            let reference = process_query(&params, &pp, q, &sparse);
            let i = targets.iter().position(|&x| x == t).unwrap();
            assert_eq!(fast[i], reference, "response differs from the Blyss server (shape {n},{nu1},{nu2})");
            let dec = client.decode_response(&fast[i]);
            assert_eq!(&dec[..64], &kept[t][..64], "decoded record is wrong");
        }
        // an update is visible to the next query
        let t = targets[0];
        let item = random_item(&params, &mut rng);
        dense.update_item(&params, t, &item);
        let (r, _) = process_queries(&params, &pp, &[client.generate_query(t)], &dense);
        assert_eq!(&client.decode_response(&r[0])[..64], &item[..64], "update not visible");
        println!("ok  n={n} nu1={nu1} nu2={nu2}: {items} records, responses identical to the Blyss server, update visible");
    }
}

fn bench(args: &[String]) {
    let v: Vec<usize> = args[0].split(',').map(|x| x.parse().unwrap()).collect();
    let get = |k: &str, d: &str| -> String {
        args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.to_string())
    };
    let imp = get("--impl", "fast");
    let batches: Vec<usize> = get("--batches", "1,8,32").split(',').map(|x| x.parse().unwrap()).collect();
    let threads: usize = get("--threads", "0").parse().unwrap();
    let trials: usize = get("--trials", "2").parse().unwrap();
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    let params = table_params(v[0], v[1], v[2]);
    let items = params.num_items();
    let mut rng = rand::thread_rng();
    let mut client = Client::init(&params);
    let pp = client.generate_keys();
    let tt = rayon::current_num_threads();
    let shape = format!("n{}_nu{}_{}", v[0], v[1], v[2]);
    let t = Instant::now();
    if imp == "blyss" {
        let mut db = SparseDb::new();
        for idx in 0..items {
            let mut body = (idx as u32).to_be_bytes().to_vec();
            body.extend_from_slice(&random_item(&params, &mut rng));
            update_item(&params, &body, &mut db).unwrap();
        }
        let load_s = t.elapsed().as_secs_f64();
        for &b in &batches {
            for _ in 0..trials {
                let qs: Vec<Query> = (0..b).map(|_| client.generate_query(rng.gen_range(0..items))).collect();
                // sequential (one query at a time, each using the server's own parallelism)
                let t = Instant::now();
                for q in &qs {
                    let _ = process_query(&params, &pp, q, &db);
                }
                let ms = t.elapsed().as_secs_f64() * 1e3;
                println!("impl=blyss,shape={shape},records={items},record_bytes={},threads={tt},batch={b},total_ms={ms:.1},per_query_ms={:.1},load_s={load_s:.1}", params.db_item_size, ms / b as f64);
                // concurrent (all queries in parallel, as a loaded server would run them)
                if b > 1 {
                    use rayon::prelude::*;
                    let t = Instant::now();
                    let _: Vec<Vec<u8>> = qs.par_iter().map(|q| process_query(&params, &pp, q, &db)).collect();
                    let ms = t.elapsed().as_secs_f64() * 1e3;
                    println!("impl=blyss_concurrent,shape={shape},records={items},record_bytes={},threads={tt},batch={b},total_ms={ms:.1},per_query_ms={:.1},load_s={load_s:.1}", params.db_item_size, ms / b as f64);
                }
            }
        }
        let mut body = (3u32).to_be_bytes().to_vec();
        body.extend_from_slice(&random_item(&params, &mut rng));
        let t = Instant::now();
        update_item(&params, &body, &mut db).unwrap();
        println!("impl=blyss,shape={shape},update_ms={:.3}", t.elapsed().as_secs_f64() * 1e3);
    } else {
        let mut db = DenseDb::zero(&params);
        for idx in 0..items {
            db.update_item(&params, idx, &item_for(&params, idx));
        }
        let load_s = t.elapsed().as_secs_f64();
        let mut bad = 0usize;
        let mut checked = 0usize;
        for &b in &batches {
            for _ in 0..trials {
                let targets: Vec<usize> = (0..b).map(|_| rng.gen_range(0..items)).collect();
                let qs: Vec<Query> = targets.iter().map(|&t| client.generate_query(t)).collect();
                let (resp, tm) = process_queries(&params, &pp, &qs, &db);
                // every answer must decode to the stored record (noise grows with nu2)
                for (r, &t) in resp.iter().zip(&targets).take(8) {
                    checked += 1;
                    if client.decode_response(r)[..64] != item_for(&params, t)[..64] {
                        bad += 1;
                    }
                }
                println!(
                    "impl=fast,shape={shape},records={items},record_bytes={},threads={tt},batch={b},total_ms={:.1},per_query_ms={:.1},expand_ms={:.1},firstdim_ms={:.1},kernel_ms={:.1},fold_ms={:.1},pack_encode_ms={:.1},db_gb={:.2},load_s={load_s:.1}",
                    params.db_item_size, tm.total_ms, tm.total_ms / b as f64, tm.expand_ms, tm.firstdim_ms, tm.firstdim_kernel_ms, tm.fold_ms, tm.pack_encode_ms, db.bytes() as f64 / 1e9
                );
            }
        }
        println!("impl=fast,shape={shape},decoded_ok={},decoded_checked={checked}", checked - bad);
        assert_eq!(bad, 0, "{bad} of {checked} answers decoded wrong at shape {shape}: the parameters are too noisy");
        let item = random_item(&params, &mut rng);
        let t = Instant::now();
        db.update_item(&params, 3, &item);
        println!("impl=fast,shape={shape},update_ms={:.3}", t.elapsed().as_secs_f64() * 1e3);
    }
    let _ = log2_ceil(params.pt_modulus);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("check") => check(),
        Some("bench") => bench(&args[2..]),
        _ => println!("spiral_bench check | bench N,NU1,NU2 [--impl blyss|fast] [--batches 1,8,32] [--threads T] [--trials 2]"),
    }
}
