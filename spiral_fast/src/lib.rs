//! A faster Spiral PIR server for the UOKVS tables.
//!
//! Same scheme, parameters, client, and wire format as the Blyss Spiral server
//! (`blyssprivacy/sdk`, `lib/server`); responses are byte-identical. Changes:
//!
//! 1. **Dense database, coefficient-major.** The Blyss server keeps every
//!    record polynomial in a hash map (it is built for sparse key-value
//!    stores) and walks records in the outer loop. Our tables are full, so
//!    the database is one array per (instance, trial) in layout
//!    `[z][j][i]` (NTT coefficient, first-dimension index, further index).
//!    For one coefficient `z`, a query contributes `dim0 x 2` words (8 KB),
//!    which stay in L1 while the database streams once.
//! 2. **Batched first dimension.** `k` queries share one pass over the
//!    database: each database word is loaded once and multiplied into the
//!    accumulators of all `k` queries. Accumulators are reduced lazily
//!    (`x = lo32(x) + hi32(x) * (2^32 mod q)` every 128 terms), so the inner
//!    loop is plain 32x32->64 multiply-adds that vectorize.
//! 3. **Parallel everywhere.** The first dimension is split into blocks of
//!    16 coefficients (128 tasks per (instance, trial) array) instead of the
//!    Blyss server's `instances x n^2` tasks (4 for 8 KB records). Query
//!    expansion and folding run in parallel across queries.
//! 5. **AVX-512 kernel.** For each 8 records and up to 4 queries, the 16
//!    accumulators stay in zmm registers for the whole first dimension
//!    (runtime-detected; a portable kernel is used otherwise).
//! 4. **Row updates.** `update_item` re-encodes one record (its `n^2` chunk
//!    polynomials) in place: what one UOKVS bucket write costs the server.

use rayon::prelude::*;
use spiral_rs::{arith::barrett_coeff_u64, client::*, params::*, poly::*};
use spiral_server::compute::{fold::*, pack::*, query_expansion::*};
use spiral_server::db::loading::{convert_pt_to_poly, pack_ntt_poly};
use spiral_server::server::encode;
use std::time::Instant;

/// Coefficients per first-dimension task.
pub const Z_BLOCK: usize = 16;
/// Terms accumulated between lazy reductions (products are < 2^56).
const LAZY_EVERY: usize = 128;

pub struct DenseDb {
    pub dim0: usize,
    pub num_per: usize,
    pub poly_len: usize,
    pub chunks: usize,
    /// one array per (instance, trial), layout [z][j][i], packed crt0 | crt1 << 32
    pub data: Vec<Vec<u64>>,
}

impl DenseDb {
    pub fn zero(params: &Params) -> Self {
        let dim0 = 1 << params.db_dim_1;
        let num_per = 1 << params.db_dim_2;
        let chunks = params.instances * params.n * params.n;
        let len = params.poly_len * dim0 * num_per;
        Self { dim0, num_per, poly_len: params.poly_len, chunks, data: (0..chunks).map(|_| vec![0u64; len]).collect() }
    }

    #[inline]
    fn at(&self, z: usize, j: usize, i: usize) -> usize {
        (z * self.dim0 + j) * self.num_per + i
    }

    /// Sets record `idx` from its plaintext bytes (at most `instances * n^2 *
    /// bytes_per_chunk`), exactly as the Blyss server's `update_item_raw`.
    pub fn update_item(&mut self, params: &Params, idx: usize, data: &[u8]) {
        let per = params.bytes_per_chunk();
        let mut bucket = vec![0u8; self.chunks * per];
        bucket[..data.len()].copy_from_slice(data);
        let polys: Vec<Vec<u64>> = bucket
            .par_chunks_exact(per)
            .map(|pt| pack_ntt_poly(&convert_pt_to_poly(params, pt)))
            .collect();
        let (j, i) = (idx / self.num_per, idx % self.num_per);
        for (c, p) in polys.iter().enumerate() {
            for z in 0..self.poly_len {
                let k = self.at(z, j, i);
                self.data[c][k] = p[z];
            }
        }
    }

    pub fn bytes(&self) -> usize {
        self.data.iter().map(|d| d.len() * 8).sum()
    }
}

/// Query vector of the first dimension in layout [z][j][4].
fn reorient(params: &Params, v_reg: &[u64], dim0: usize) -> Vec<u64> {
    let pl = params.poly_len;
    // [z][j][4]: row-0 word, its high half, row-1 word, its high half (the
    // multiplies use the low 32 bits of each 64-bit lane)
    let mut out = vec![0u64; pl * dim0 * 4];
    for j in 0..dim0 {
        for r in 0..2 {
            let src = &v_reg[(j * 2 + r) * pl..(j * 2 + r + 1) * pl];
            for z in 0..pl {
                let o = (z * dim0 + j) * 4 + 2 * r;
                out[o] = src[z];
                out[o + 1] = src[z] >> 32;
            }
        }
    }
    out
}

#[inline(always)]
fn lazy(x: u64, r: u64) -> u64 {
    (x & 0xffff_ffff) + (x >> 32) * r
}

/// First-dimension pass of `k` queries over one (instance, trial) array for
/// coefficients `z0..z1`. Returns accumulators [z - z0][q][4][i] reduced mod q.
fn firstdim_block(params: &Params, db: &DenseDb, chunk: usize, qs: &[Vec<u64>], z0: usize, z1: usize) -> Vec<u64> {
    let (dim0, np, k) = (db.dim0, db.num_per, qs.len());
    let (m0, m1) = (params.moduli[0], params.moduli[1]);
    let r0 = (1u64 << 32) % m0;
    let r1 = (1u64 << 32) % m1;
    let data = &db.data[chunk];
    let mut out = vec![0u64; (z1 - z0) * k * 4 * np];
    let mut acc = vec![0u64; k * 4 * np];
    for z in z0..z1 {
        acc.iter_mut().for_each(|x| *x = 0);
        let brow = &data[z * dim0 * np..(z + 1) * dim0 * np];
        for j in 0..dim0 {
            let b = &brow[j * np..(j + 1) * np];
            for (q, qv) in qs.iter().enumerate() {
                let a = &qv[(z * dim0 + j) * 4..(z * dim0 + j) * 4 + 4];
                let (a0l, a0h, a1l, a1h) = (a[0] & 0xffff_ffff, a[1], a[2] & 0xffff_ffff, a[3]);
                let base = q * 4 * np;
                let (c0, rest) = acc[base..base + 4 * np].split_at_mut(np);
                let (c1, rest) = rest.split_at_mut(np);
                let (c2, c3) = rest.split_at_mut(np);
                for i in 0..np {
                    let bl = b[i] & 0xffff_ffff;
                    let bh = b[i] >> 32;
                    c0[i] += a0l * bl;
                    c1[i] += a0h * bh;
                    c2[i] += a1l * bl;
                    c3[i] += a1h * bh;
                }
            }
            if (j + 1) % LAZY_EVERY == 0 {
                for q in 0..k {
                    let base = q * 4 * np;
                    for s in 0..4 {
                        let r = if s % 2 == 0 { r0 } else { r1 };
                        for x in &mut acc[base + s * np..base + (s + 1) * np] {
                            *x = lazy(*x, r);
                        }
                    }
                }
            }
        }
        let o = &mut out[(z - z0) * k * 4 * np..(z - z0 + 1) * k * 4 * np];
        for q in 0..k {
            for s in 0..4 {
                let m = if s % 2 == 0 { m0 } else { m1 };
                for i in 0..np {
                    let idx = (q * 4 + s) * np + i;
                    o[idx] = acc[idx] % m;
                }
            }
        }
    }
    out
}


// ---------------------------------------------------------------- AVX-512 kernel

/// Queries per register block: 4 accumulators per query x QG queries stay in
/// zmm registers while j runs over the whole first dimension.
const QG: usize = 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn block_avx512<const Q: usize>(
    brow: &[u64], qs: &[&[u64]], z: usize, dim0: usize, np: usize, ic: usize, r0: u64, r1: u64, out: &mut [[u64; 8]],
) {
    use std::arch::x86_64::*;
    let mask = _mm512_set1_epi64(0xffff_ffff);
    let vr0 = _mm512_set1_epi64(r0 as i64);
    let vr1 = _mm512_set1_epi64(r1 as i64);
    let mut acc = [_mm512_setzero_si512(); 16];
    let qbase = z * dim0 * 4;
    for j in 0..dim0 {
        let b = _mm512_loadu_si512(brow.as_ptr().add(j * np + ic) as *const _);
        let bh = _mm512_srli_epi64::<32>(b);
        for q in 0..Q {
            let a = qs[q].as_ptr().add(qbase + j * 4);
            let a0 = _mm512_set1_epi64(*a as i64);
            let a0h = _mm512_set1_epi64(*a.add(1) as i64);
            let a1 = _mm512_set1_epi64(*a.add(2) as i64);
            let a1h = _mm512_set1_epi64(*a.add(3) as i64);
            acc[q * 4] = _mm512_add_epi64(acc[q * 4], _mm512_mul_epu32(a0, b));
            acc[q * 4 + 1] = _mm512_add_epi64(acc[q * 4 + 1], _mm512_mul_epu32(a0h, bh));
            acc[q * 4 + 2] = _mm512_add_epi64(acc[q * 4 + 2], _mm512_mul_epu32(a1, b));
            acc[q * 4 + 3] = _mm512_add_epi64(acc[q * 4 + 3], _mm512_mul_epu32(a1h, bh));
        }
        if (j + 1) % LAZY_EVERY == 0 {
            for q in 0..Q {
                for s in 0..4 {
                    let x = acc[q * 4 + s];
                    let r = if s % 2 == 0 { vr0 } else { vr1 };
                    acc[q * 4 + s] = _mm512_add_epi64(_mm512_and_si512(x, mask), _mm512_mul_epu32(_mm512_srli_epi64::<32>(x), r));
                }
            }
        }
    }
    for q in 0..Q {
        for s in 0..4 {
            _mm512_storeu_si512(out[q * 4 + s].as_mut_ptr() as *mut _, acc[q * 4 + s]);
        }
    }
}

/// AVX-512 version of `firstdim_block` (needs num_per a multiple of 8).
#[cfg(target_arch = "x86_64")]
fn firstdim_block_avx512(params: &Params, db: &DenseDb, chunk: usize, qs: &[Vec<u64>], z0: usize, z1: usize) -> Vec<u64> {
    let (dim0, np, k) = (db.dim0, db.num_per, qs.len());
    let (m0, m1) = (params.moduli[0], params.moduli[1]);
    let (r0, r1) = ((1u64 << 32) % m0, (1u64 << 32) % m1);
    let data = &db.data[chunk];
    let qrefs: Vec<&[u64]> = qs.iter().map(|v| v.as_slice()).collect();
    let mut out = vec![0u64; (z1 - z0) * k * 4 * np];
    let mut tmp = [[0u64; 8]; 16];
    for z in z0..z1 {
        let brow = &data[z * dim0 * np..(z + 1) * dim0 * np];
        let o = &mut out[(z - z0) * k * 4 * np..(z - z0 + 1) * k * 4 * np];
        for ic in (0..np).step_by(8) {
            let mut qg = 0;
            while qg < k {
                let qn = (k - qg).min(QG);
                let g = &qrefs[qg..qg + qn];
                unsafe {
                    match qn {
                        4 => block_avx512::<4>(brow, g, z, dim0, np, ic, r0, r1, &mut tmp),
                        3 => block_avx512::<3>(brow, g, z, dim0, np, ic, r0, r1, &mut tmp),
                        2 => block_avx512::<2>(brow, g, z, dim0, np, ic, r0, r1, &mut tmp),
                        _ => block_avx512::<1>(brow, g, z, dim0, np, ic, r0, r1, &mut tmp),
                    }
                }
                for q in 0..qn {
                    for s in 0..4 {
                        let crt = s % 2;
                        for l in 0..8 {
                            o[((qg + q) * 4 + s) * np + ic + l] = barrett_coeff_u64(params, tmp[q * 4 + s][l], crt);
                        }
                    }
                }
                qg += qn;
            }
        }
    }
    out
}

fn use_avx512(np: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        np % 8 == 0 && std::env::var("SPIRAL_FAST_SCALAR").is_err() && is_x86_feature_detected!("avx512f")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = np;
        false
    }
}

fn firstdim_dispatch(params: &Params, db: &DenseDb, chunk: usize, qs: &[Vec<u64>], z0: usize, z1: usize) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    if use_avx512(db.num_per) {
        return firstdim_block_avx512(params, db, chunk, qs, z0, z1);
    }
    firstdim_block(params, db, chunk, qs, z0, z1)
}

#[derive(Clone, Debug, Default)]
pub struct Timing {
    pub expand_ms: f64,
    pub firstdim_ms: f64,
    /// of which the multiply-accumulate kernel (the rest is assembling outputs)
    pub firstdim_kernel_ms: f64,
    pub fold_ms: f64,
    pub pack_encode_ms: f64,
    pub total_ms: f64,
}

/// Answers a batch of queries with one pass over the database.
pub fn process_queries<'a>(params: &'a Params, pp: &PublicParameters<'a>, queries: &[Query<'a>], db: &DenseDb) -> (Vec<Vec<u8>>, Timing) {
    let t0 = Instant::now();
    let mut tm = Timing::default();
    let (dim0, np, pl) = (db.dim0, db.num_per, params.poly_len);
    let trials = params.n * params.n;
    let k = queries.len();

    // 1. expansion, in parallel across queries
    let t = Instant::now();
    let expanded: Vec<(Vec<u64>, Vec<PolyMatrixNTT>, Vec<PolyMatrixNTT>)> = queries
        .par_iter()
        .map(|q| {
            let (v_reg, v_folding) = expand_query(params, pp, q, None);
            let neg = get_v_folding_neg(params, &v_folding);
            (reorient(params, v_reg.as_slice(), dim0), v_folding, neg)
        })
        .collect();
    tm.expand_ms = t.elapsed().as_secs_f64() * 1e3;

    // 2. first dimension: (chunk x z-block) tasks, all queries per task
    let t = Instant::now();
    // move the query vectors out (no copy); folding keys stay in `expanded`
    let mut expanded = expanded;
    let qs: Vec<Vec<u64>> = expanded.iter_mut().map(|e| std::mem::take(&mut e.0)).collect();
    let mut inter: Vec<Vec<Vec<PolyMatrixNTT>>> = (0..k)
        .map(|_| (0..db.chunks).map(|_| (0..np).map(|_| PolyMatrixNTT::zero(params, 2, 1)).collect()).collect())
        .collect();
    // one (instance, trial) array at a time, so the block outputs stay small;
    // within it, blocks of coefficients run in parallel
    for c in 0..db.chunks {
        let tk = Instant::now();
        let z0s: Vec<usize> = (0..pl).step_by(Z_BLOCK).collect();
        let blocks: Vec<Vec<u64>> = z0s
            .par_iter()
            .map(|&z0| firstdim_dispatch(params, db, c, &qs, z0, (z0 + Z_BLOCK).min(pl)))
            .collect();
        tm.firstdim_kernel_ms += tk.elapsed().as_secs_f64() * 1e3;
        for (bi, blk) in blocks.into_iter().enumerate() {
            let z0 = z0s[bi];
            let zn = (z0 + Z_BLOCK).min(pl) - z0;
            for dz in 0..zn {
                let z = z0 + dz;
                for q in 0..k {
                    for i in 0..np {
                        let d = inter[q][c][i].data.as_mut_slice();
                        for s in 0..4 {
                            d[s * pl + z] = blk[((dz * k + q) * 4 + s) * np + i];
                        }
                    }
                }
            }
        }
    }
    tm.firstdim_ms = t.elapsed().as_secs_f64() * 1e3;

    drop(qs);
    // 3. from_ntt + folding, parallel over (query, chunk)
    let t = Instant::now();
    let jobs: Vec<(usize, usize)> = (0..k).flat_map(|q| (0..db.chunks).map(move |c| (q, c))).collect();
    let folded: Vec<PolyMatrixRaw> = jobs
        .par_iter()
        .map(|&(q, c)| {
            let mut raw: Vec<PolyMatrixRaw> = (0..np).map(|_| PolyMatrixRaw::zero(params, 2, 1)).collect();
            for i in 0..np {
                from_ntt(&mut raw[i], &inter[q][c][i]);
            }
            fold_ciphertexts(params, &mut raw, &expanded[q].1, &expanded[q].2);
            raw[0].clone()
        })
        .collect();
    tm.fold_ms = t.elapsed().as_secs_f64() * 1e3;

    // 4. packing per instance and encoding, parallel over queries
    let t = Instant::now();
    let out: Vec<Vec<u8>> = (0..k)
        .into_par_iter()
        .map(|q| {
            let cts = &folded[q * db.chunks..(q + 1) * db.chunks];
            let packed: Vec<PolyMatrixRaw> = cts
                .chunks_exact(trials)
                .map(|c| pack(params, c, pp.v_packing.as_ref()).raw())
                .collect();
            encode(params, &packed)
        })
        .collect();
    tm.pack_encode_ms = t.elapsed().as_secs_f64() * 1e3;
    tm.total_ms = t0.elapsed().as_secs_f64() * 1e3;
    (out, tm)
}

/// Spiral parameters used for the UOKVS tables (the Blyss server's test
/// parameters with one instance): records of n^2 * 2048 bytes, 2^(nu1 + nu2) records.
pub fn table_params(n: usize, nu1: usize, nu2: usize) -> Params {
    let cfg = format!(
        r#"{{"n": {n}, "nu_1": {nu1}, "nu_2": {nu2}, "p": 256, "q2_bits": 22, "t_gsw": 7,
            "t_conv": 3, "t_exp_left": 5, "t_exp_right": 5, "instances": 1,
            "db_item_size": {} }}"#,
        n * n * 2048
    );
    spiral_rs::util::params_from_json(&cfg)
}
