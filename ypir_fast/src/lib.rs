//! Faster YPIR+SP server for the UOKVS tables.
//!
//! YPIR+SP (Menon-Wu, USENIX Security 2024) is SimplePIR with the hint moved to the server: the
//! client sends an RLWE-encoded one-hot vector over the database rows, the server multiplies it
//! with the database and then ring-packs the resulting LWE ciphertexts. The packing of the
//! hint-dependent part is precomputed offline. The reference code
//! (github.com/menonsamir/ypir, a73e550) is single-threaded, answers one query per pass, has no
//! row updates, and stores exactly one record per row. Here:
//!
//!  * **Several records per row** (`Shape::per_row`). There is no client hint, so the row width
//!    is free: wider rows shrink the query (7 bytes per row) and grow the response
//!    (12 KB per 3.5 KB of row), and `Shape::auto` picks the balance.
//!  * **Row updates** (`update_records`). The hint is linear in the database, so an update adds
//!    `a_rho * delta` for each touched block of 2048 rows only; the packing precomputation
//!    (nonlinear) is then recomputed for every output in parallel. The reference has no update
//!    path, so an update is a full offline rebuild.
//!  * **Batched, parallel online pass** (`answer`). All queries of a batch share one pass over the
//!    database (the reference AVX-512 kernel, run on column tiles in parallel), and packing runs
//!    in parallel over (query, output) pairs.
//!  * **Several public seeds** (`seeds`). Queries under one client key must use distinct public
//!    matrices, otherwise their difference leaks. With `S` seeds the server keeps `S` hints and a
//!    client's packing keys (~473 KB) are shared by up to `S` queries of one search instead of
//!    being sent with every query.
#![allow(stable_features)]
#![feature(stdarch_x86_avx512)]
use std::arch::x86_64::*;
use std::time::Instant;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::arith::rescale;
use spiral_rs::client::Client;
use spiral_rs::params::Params;
use spiral_rs::poly::*;
use ypir::client::{pack_query, raw_generate_expansion_params, YClient};
use ypir::kernel::fast_batched_dot_product_avx512;
use ypir::modulus_switch::ModulusSwitch;
use spiral_rs::arith::barrett_reduction_u128;
use spiral_rs::gadget::gadget_invert_rdim;
use spiral_rs::ntt::ntt_forward;
use ypir::packing::{
    apply_automorph_ntt, condense_matrix, fast_add_into, fast_add_into_no_reduce,
    fast_reduce, generate_automorph_tables_brute_force,
    pack_using_precomp_vals, precompute_pack, prep_pack_lwes,
};
use ypir::util::{add_into_at_no_reduce, scalar_multiply_avx};
use ypir::params::{params_for_scenario_simplepir, GetQPrime};
use ypir::scheme::STATIC_SEED_2;
use ypir::server::{generate_fake_pack_pub_params, generate_y_constants, YServer};

pub const POLY_LEN: usize = 2048;
pub const PT_BITS: usize = 14;
/// response bytes per output ciphertext (q'_1 = 2^20, q_2 = 28 bits)
pub const RESP_BYTES_PER_OUTPUT: usize = (20 + 28) * POLY_LEN / 8;
/// query bytes per database row (56-bit modulus)
pub const QUERY_BYTES_PER_ROW: usize = 7;
/// the client's packing keys: 11 key-switching rows x t_exp_left = 3 polynomials x 2048 x 7 B
pub const PP_BYTES: usize = 11 * 3 * POLY_LEN * 7;

/// Packing precomputation for one output: the result's first row and, for each of the 2047
/// merges, the gadget-decomposed ciphertext in condensed NTT form, stored compactly
/// (`t_exp_left x poly_len` words; the reference keeps them in full-size matrices, twice the memory).
pub type Pc = (PolyMatrixNTT<'static>, Vec<u64>);
/// the reference representation (for the comparison runs)
pub type RefPc = (PolyMatrixNTT<'static>, Vec<PolyMatrixNTT<'static>>, Vec<Vec<usize>>);

pub struct RefState {
    pub hint_ms: f64,
    pub precomp_ms: f64,
    precomp: Vec<RefPc>,
}

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub records: usize,
    pub record_bytes: usize,
    pub per_row: usize,
    /// database rows (a power of two, at least 2048)
    pub rows: usize,
    /// output ciphertexts per answer (2048 elements of 14 bits each)
    pub instances: usize,
}

impl Shape {
    pub fn new(records: usize, record_bytes: usize, per_row: usize) -> Shape {
        let rows = ((records + per_row - 1) / per_row).next_power_of_two().max(POLY_LEN);
        let row_bits = per_row * record_bytes * 8;
        let instances = (row_bits + POLY_LEN * PT_BITS - 1) / (POLY_LEN * PT_BITS);
        Shape { records, record_bytes, per_row, rows, instances }
    }
    /// per_row minimizing query + response bytes (packing keys are a fixed cost per client)
    pub fn auto(records: usize, record_bytes: usize) -> Shape {
        (1..=256)
            .map(|k| Shape::new(records, record_bytes, k))
            .min_by_key(|s| s.query_bytes() + s.response_bytes())
            .unwrap()
    }
    pub fn cols(&self) -> usize {
        self.instances * POLY_LEN
    }
    pub fn row_bytes(&self) -> usize {
        self.per_row * self.record_bytes
    }
    pub fn query_bytes(&self) -> usize {
        self.rows * QUERY_BYTES_PER_ROW
    }
    pub fn response_bytes(&self) -> usize {
        self.instances * RESP_BYTES_PER_OUTPUT
    }
    pub fn db_bytes(&self) -> usize {
        self.rows * self.cols() * 2
    }
    pub fn label(&self) -> String {
        format!("k{}_r{}_i{}", self.per_row, self.rows, self.instances)
    }
    pub fn params(&self) -> &'static Params {
        let p = params_for_scenario_simplepir(self.rows, self.row_bytes() * 8);
        assert_eq!(p.instances, self.instances);
        assert_eq!(1 << (p.db_dim_1 + p.poly_len_log2), self.rows);
        Box::leak(Box::new(p))
    }
}

/// pack bytes LSB-first into 14-bit elements
pub fn bytes_to_elems(bytes: &[u8], out: &mut [u16]) {
    out.fill(0);
    let mask = (1u64 << PT_BITS) - 1;
    let (mut acc, mut nbits, mut o) = (0u64, 0usize, 0usize);
    for &b in bytes {
        acc |= (b as u64) << nbits;
        nbits += 8;
        while nbits >= PT_BITS {
            out[o] = (acc & mask) as u16;
            o += 1;
            acc >>= PT_BITS;
            nbits -= PT_BITS;
        }
    }
    if nbits > 0 {
        out[o] = (acc & mask) as u16;
    }
}

pub fn elems_to_bytes(elems: &[u64], n_bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes);
    let (mut acc, mut nbits) = (0u64, 0usize);
    let mut it = elems.iter();
    while out.len() < n_bytes {
        while nbits < 8 {
            acc |= (it.next().copied().unwrap_or(0) & ((1 << PT_BITS) - 1)) << nbits;
            nbits += PT_BITS;
        }
        out.push(acc as u8);
        acc >>= 8;
        nbits -= 8;
    }
    out
}

struct SeedState {
    seed: u8,
    /// public query polynomials (negacyclic-permuted, NTT), one per block of 2048 rows
    aq: Vec<PolyMatrixNTT<'static>>,
    /// poly_len x cols, mod q
    hint: Vec<u64>,
    precomp: Vec<Pc>,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct AnswerTiming {
    pub first_ms: f64,
    pub pack_ms: f64,
    pub total_ms: f64,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct UpdateTiming {
    pub rows: usize,
    pub blocks: usize,
    pub db_ms: f64,
    pub hint_ms: f64,
    pub precomp_ms: f64,
    pub total_ms: f64,
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SetupTiming {
    pub db_ms: f64,
    pub hint_ms: f64,
    pub precomp_ms: f64,
}

pub struct YpirServer {
    pub shape: Shape,
    pub params: &'static Params,
    server: YServer<'static, u16>,
    y_constants: (Vec<PolyMatrixNTT<'static>>, Vec<PolyMatrixNTT<'static>>),
    fake_pp: Vec<PolyMatrixNTT<'static>>,
    seeds: Vec<SeedState>,
    /// NTT-domain automorphism tables (the reference recomputes them for every output)
    tables: Vec<Vec<usize>>,
    pub setup: SetupTiming,
    /// bumped by every update (prepared packing is only valid within one epoch)
    pub epoch: u64,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

struct SendPtr(*mut u16);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl YpirServer {
    pub fn new<F>(shape: Shape, n_seeds: usize, record: F) -> YpirServer
    where
        F: Fn(usize) -> Vec<u8> + Sync,
    {
        assert!(n_seeds >= 1 && n_seeds <= 200);
        let params = shape.params();
        let t = Instant::now();
        let mut server =
            YServer::<u16>::new(params, std::iter::repeat(0u16), true, false, true);
        let rows = shape.rows;
        let cols = shape.cols();
        {
            let db = server.db_mut();
            assert_eq!(db.len(), rows * cols);
            let ptr = SendPtr(db.as_mut_ptr());
            // blocks of 64 rows, written column by column (contiguous runs of 64)
            let blk = 64.min(rows);
            (0..rows / blk).into_par_iter().for_each(|bi| {
                let p = &ptr;
                let r0 = bi * blk;
                let mut el = vec![0u16; blk * cols];
                for k in 0..blk {
                    let bytes = row_bytes(&shape, r0 + k, &record);
                    bytes_to_elems(&bytes, &mut el[k * cols..(k + 1) * cols]);
                }
                for c in 0..cols {
                    for k in 0..blk {
                        unsafe { *p.0.add(c * rows + r0 + k) = el[k * cols + c] };
                    }
                }
            });
        }
        let db_ms = ms(t);
        let y_constants = generate_y_constants(params);
        let fake_pp = generate_fake_pack_pub_params(params);
        let mut s = YpirServer {
            shape,
            params,
            server,
            y_constants,
            fake_pp,
            seeds: Vec::new(),
            tables: generate_automorph_tables_brute_force(params),
            setup: SetupTiming { db_ms, ..Default::default() },
            epoch: 0,
        };
        for j in 0..n_seeds {
            let seed = j as u8; // seed 0 is the reference's SEED_0
            let aq = s.server.generate_pseudorandom_query(seed);
            let t = Instant::now();
            let hint = s.full_hint(&aq, seed);
            s.setup.hint_ms += ms(t);
            let t = Instant::now();
            let precomp = s.all_precomp(&hint);
            s.setup.precomp_ms += ms(t);
            s.seeds.push(SeedState { seed, aq, hint, precomp });
        }
        s
    }

    pub fn n_seeds(&self) -> usize {
        self.seeds.len()
    }
    pub fn seed_idx(&self, j: usize) -> u8 {
        self.seeds[j].seed
    }

    fn full_hint(&self, aq: &[PolyMatrixNTT<'static>], seed: u8) -> Vec<u64> {
        let cols = self.shape.cols();
        let tile = 16;
        let parts: Vec<(usize, Vec<u64>)> = (0..cols / tile)
            .into_par_iter()
            .map(|t| (t * tile, self.server.multiply_with_db_ring(aq, t * tile..(t + 1) * tile, seed)))
            .collect();
        let mut hint = vec![0u64; POLY_LEN * cols];
        hint.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
            for (c0, p) in &parts {
                row[*c0..*c0 + tile].copy_from_slice(&p[i * tile..(i + 1) * tile]);
            }
        });
        hint
    }

    fn precomp_output(&self, hint: &[u64], i: usize) -> Pc {
        let n = POLY_LEN;
        let cols = self.shape.cols();
        let mut v = Vec::with_capacity((n + 1) * n);
        for j in 0..n {
            v.extend_from_slice(&hint[j * cols + i * n..j * cols + (i + 1) * n]);
        }
        v.extend(std::iter::repeat(0u64).take(n));
        let prep = prep_pack_lwes(self.params, &v, n);
        drop(v);
        precompute_pack_par(self.params, prep, &self.fake_pp, &self.y_constants)
    }

    /// The parallel precomputation equals the reference `precompute_pack` (seed j, output i).
    pub fn precomp_matches_reference(&self, j: usize, i: usize) -> bool {
        let n = POLY_LEN;
        let cols = self.shape.cols();
        let hint = &self.seeds[j].hint;
        let mut v = Vec::with_capacity((n + 1) * n);
        for r in 0..n {
            v.extend_from_slice(&hint[r * cols + i * n..r * cols + (i + 1) * n]);
        }
        v.extend(std::iter::repeat(0u64).take(n));
        let prep = prep_pack_lwes(self.params, &v, n);
        let (r0, vals, _) =
            precompute_pack(self.params, self.params.poly_len_log2, &prep, &self.fake_pp, &self.y_constants);
        let (m0, mvals) = &self.seeds[j].precomp[i];
        let w = self.params.t_exp_left * POLY_LEN;
        r0.as_slice() == m0.as_slice()
            && vals.len() * w == mvals.len()
            && vals.iter().enumerate().all(|(k, a)| {
                (0..self.params.t_exp_left).all(|r| {
                    a.get_poly(r, 0)[..POLY_LEN] == mvals[k * w + r * POLY_LEN..k * w + (r + 1) * POLY_LEN]
                })
            })
    }

    /// Bytes held per seed: hint plus packing precomputation.
    pub fn bytes_per_seed(&self) -> usize {
        let st = &self.seeds[0];
        let pc: usize = st
            .precomp
            .iter()
            .map(|(r, v)| (r.as_slice().len() + v.len()) * 8)
            .sum();
        st.hint.len() * 8 + pc
    }

    fn all_precomp(&self, hint: &[u64]) -> Vec<Pc> {
        (0..self.shape.instances).into_par_iter().map(|i| self.precomp_output(hint, i)).collect()
    }

    /// The elements of one database row.
    pub fn row_elems(&self, r: usize) -> Vec<u16> {
        let db = self.server.db();
        let rows = self.shape.rows;
        (0..self.shape.cols()).map(|c| db[c * rows + r]).collect()
    }

    pub fn record(&self, idx: usize) -> Vec<u8> {
        let s = &self.shape;
        let el: Vec<u64> = self.row_elems(idx / s.per_row).iter().map(|&x| x as u64).collect();
        let row = elems_to_bytes(&el, s.row_bytes());
        let off = (idx % s.per_row) * s.record_bytes;
        row[off..off + s.record_bytes].to_vec()
    }

    /// Replace records; the hint is updated for the touched blocks of 2048 rows and the packing
    /// precomputation is redone (all seeds).
    pub fn update_records(&mut self, updates: &[(usize, Vec<u8>)]) -> UpdateTiming {
        let t0 = Instant::now();
        let s = self.shape;
        let rows = s.rows;
        let cols = s.cols();
        // old rows, new rows
        let mut touched: std::collections::BTreeMap<usize, (Vec<u16>, Vec<u8>)> = Default::default();
        for (idx, bytes) in updates {
            assert!(*idx < s.records && bytes.len() == s.record_bytes);
            let r = idx / s.per_row;
            if !touched.contains_key(&r) {
                let old = self.row_elems(r);
                let el: Vec<u64> = old.iter().map(|&x| x as u64).collect();
                let rb = elems_to_bytes(&el, s.row_bytes());
                touched.insert(r, (old, rb));
            }
            let e = touched.get_mut(&r).unwrap();
            let off = (idx % s.per_row) * s.record_bytes;
            e.1[off..off + s.record_bytes].copy_from_slice(bytes);
        }
        // write the new rows, keep (row, old, new) elements
        let mut changes: Vec<(usize, Vec<u16>, Vec<u16>)> = Vec::with_capacity(touched.len());
        {
            let db = self.server.db_mut();
            for (r, (old, rb)) in touched {
                let mut new = vec![0u16; cols];
                bytes_to_elems(&rb, &mut new);
                for c in 0..cols {
                    db[c * rows + r] = new[c];
                }
                changes.push((r, old, new));
            }
        }
        let db_ms = ms(t0);
        // group by block of 2048 rows
        let mut blocks: Vec<(usize, Vec<usize>)> = Vec::new();
        for (k, (r, _, _)) in changes.iter().enumerate() {
            let b = r / POLY_LEN;
            match blocks.last_mut() {
                Some((bb, v)) if *bb == b => v.push(k),
                _ => blocks.push((b, vec![k])),
            }
        }
        let params = self.params;
        let q = params.modulus;
        let mut hint_ms = 0.0;
        let mut precomp_ms = 0.0;
        for j in 0..self.seeds.len() {
            let t = Instant::now();
            let aq = &self.seeds[j].aq;
            // per column: sum over touched blocks of aq[block] * delta poly
            let mut delta = vec![0u64; cols * POLY_LEN];
            delta.par_chunks_mut(POLY_LEN).enumerate().for_each(|(c, out)| {
                let mut sum = PolyMatrixNTT::zero(params, 1, 1);
                let mut raw = PolyMatrixRaw::zero(params, 1, 1);
                let mut ntt = PolyMatrixNTT::zero(params, 1, 1);
                let mut prod = PolyMatrixNTT::zero(params, 1, 1);
                for (b, ks) in &blocks {
                    raw.data.as_mut_slice().fill(0);
                    let mut any = false;
                    for &k in ks {
                        let (r, old, new) = &changes[k];
                        if old[c] != new[c] {
                            raw.data[r % POLY_LEN] = (q + new[c] as u64 - old[c] as u64) % q;
                            any = true;
                        }
                    }
                    if !any {
                        continue;
                    }
                    to_ntt(&mut ntt, &raw);
                    multiply(&mut prod, &aq[*b], &ntt);
                    add_into(&mut sum, &prod);
                }
                out.copy_from_slice(sum.raw().as_slice());
            });
            let hint = &mut self.seeds[j].hint;
            hint.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
                for c in 0..cols {
                    row[c] = (row[c] + delta[c * POLY_LEN + i]) % q;
                }
            });
            hint_ms += ms(t);
            let t = Instant::now();
            // (a serving system would double-buffer; here the old state is dropped first)
            self.seeds[j].precomp = Vec::new();
            let pc = self.all_precomp(&self.seeds[j].hint);
            self.seeds[j].precomp = pc;
            precomp_ms += ms(t);
        }
        self.epoch += 1;
        UpdateTiming {
            rows: changes.len(),
            blocks: blocks.len(),
            db_ms,
            hint_ms,
            precomp_ms,
            total_ms: ms(t0),
        }
    }

    /// Recompute the hint of seed `j` from scratch (for checking the incremental update).
    pub fn hint_matches_full(&self, j: usize) -> bool {
        let st = &self.seeds[j];
        self.full_hint(&st.aq, st.seed) == st.hint
    }

    /// Answer a batch. `queries[q] = (seed index j, packed query (rows u64), client's packing keys)`.
    /// Returns per query one response ciphertext (bytes) per output.
    pub fn answer(
        &self,
        queries: &[(usize, &[u64], &[PolyMatrixNTT<'static>])],
    ) -> (Vec<Vec<Vec<u8>>>, AnswerTiming) {
        let t0 = Instant::now();
        let nq = queries.len();
        let qv: Vec<&[u64]> = queries.iter().map(|(_, q, _)| *q).collect();
        let inter = self.first_pass(&qv);
        let first_ms = ms(t0);
        let params = self.params;
        let t1 = Instant::now();
        let n_out = self.shape.instances;
        let flat: Vec<Vec<u8>> = (0..nq * n_out)
            .into_par_iter()
            .map(|x| {
                let (q, i) = (x / n_out, x % n_out);
                let (j, _, pp) = queries[q];
                let (pr, pv) = &self.seeds[j].precomp[i];
                let row1 = pack_row1(params, pp, pv, &self.tables, &self.y_constants);
                pack_finish(params, pr, &row1, &inter[q][i * POLY_LEN..(i + 1) * POLY_LEN])
            })
            .collect();
        let mut out: Vec<Vec<Vec<u8>>> = Vec::with_capacity(nq);
        let mut it = flat.into_iter();
        for _ in 0..nq {
            out.push((&mut it).take(n_out).collect());
        }
        let pack_ms = ms(t1);
        (out, AnswerTiming { first_ms, pack_ms, total_ms: ms(t0) })
    }

    /// Packing work for a client's keys under seed `j` (before its query arrives).
    pub fn prepare(&self, j: usize, pp: &[PolyMatrixNTT<'static>]) -> Prepared {
        let row1 = (0..self.shape.instances)
            .into_par_iter()
            .map(|i| {
                let (_, pv) = &self.seeds[j].precomp[i];
                pack_row1(self.params, pp, pv, &self.tables, &self.y_constants)
            })
            .collect();
        Prepared { seed_j: j, epoch: self.epoch, row1 }
    }

    /// Prepare many (client, seed) pairs at once, parallel over all outputs.
    pub fn prepare_many(&self, reqs: &[(usize, &[PolyMatrixNTT<'static>])]) -> Vec<Prepared> {
        let n_out = self.shape.instances;
        let flat: Vec<PolyMatrixNTT<'static>> = (0..reqs.len() * n_out)
            .into_par_iter()
            .map(|x| {
                let (r, i) = (x / n_out, x % n_out);
                let (j, pp) = reqs[r];
                let (_, pv) = &self.seeds[j].precomp[i];
                pack_row1(self.params, pp, pv, &self.tables, &self.y_constants)
            })
            .collect();
        let mut it = flat.into_iter();
        reqs.iter()
            .map(|(j, _)| Prepared { seed_j: *j, epoch: self.epoch, row1: (&mut it).take(n_out).collect() })
            .collect()
    }

    /// First pass for a batch: per query, `cols` LWE bodies.
    fn first_pass(&self, qv: &[&[u64]]) -> Vec<Vec<u64>> {
        let rows = self.shape.rows;
        let cols = self.shape.cols();
        let nq = qv.len();
        let mut buf = AlignedMemory64::new(nq * rows);
        for (q, v) in qv.iter().enumerate() {
            assert_eq!(v.len(), rows);
            buf.as_mut_slice()[q * rows..(q + 1) * rows].copy_from_slice(v);
        }
        let mut groups = Vec::new();
        let mut g0 = 0;
        for k in [8usize, 4, 2, 1] {
            while nq - g0 >= k {
                groups.push((g0, k));
                g0 += k;
            }
        }
        let tile = 8;
        let db = self.server.db();
        let a_all = buf.as_slice();
        let params = self.params;
        let parts: Vec<Vec<u64>> = (0..cols / tile)
            .into_par_iter()
            .map(|t| {
                let b_t = &db[t * tile * rows..(t + 1) * tile * rows];
                let mut out = vec![0u64; nq * tile];
                for &(g0, k) in &groups {
                    let a = &a_all[g0 * rows..(g0 + k) * rows];
                    let c = &mut out[g0 * tile..(g0 + k) * tile];
                    match k {
                        8 => fast_batched_dot_product_avx512::<8, u16>(params, c, a, rows, b_t, rows, tile),
                        4 => fast_batched_dot_product_avx512::<4, u16>(params, c, a, rows, b_t, rows, tile),
                        2 => fast_batched_dot_product_avx512::<2, u16>(params, c, a, rows, b_t, rows, tile),
                        _ => fast_batched_dot_product_avx512::<1, u16>(params, c, a, rows, b_t, rows, tile),
                    }
                }
                out
            })
            .collect();
        let mut inter = vec![vec![0u64; cols]; nq];
        inter.par_iter_mut().enumerate().for_each(|(q, iv)| {
            for (t, p) in parts.iter().enumerate() {
                iv[t * tile..(t + 1) * tile].copy_from_slice(&p[q * tile..(q + 1) * tile]);
            }
        });
        inter
    }

    /// Online answer for queries whose packing was prepared: one pass plus the LWE bodies.
    pub fn answer_prepared(&self, queries: &[(&Prepared, &[u64])]) -> (Vec<Vec<Vec<u8>>>, AnswerTiming) {
        let t0 = Instant::now();
        for (p, _) in queries {
            assert_eq!(p.epoch, self.epoch, "prepared packing is stale");
        }
        let qv: Vec<&[u64]> = queries.iter().map(|(_, q)| *q).collect();
        let inter = self.first_pass(&qv);
        let first_ms = ms(t0);
        let t1 = Instant::now();
        let n_out = self.shape.instances;
        let flat: Vec<Vec<u8>> = (0..queries.len() * n_out)
            .into_par_iter()
            .map(|x| {
                let (q, i) = (x / n_out, x % n_out);
                let p = queries[q].0;
                let (pr, _) = &self.seeds[p.seed_j].precomp[i];
                pack_finish(self.params, pr, &p.row1[i], &inter[q][i * POLY_LEN..(i + 1) * POLY_LEN])
            })
            .collect();
        let mut out = Vec::with_capacity(queries.len());
        let mut it = flat.into_iter();
        for _ in 0..queries.len() {
            out.push((&mut it).take(n_out).collect());
        }
        (out, AnswerTiming { first_ms, pack_ms: ms(t1), total_ms: ms(t0) })
    }

    /// The reference offline precomputation for seed 0, single-threaded (YPIR's code).
    pub fn offline_reference(&self) -> RefState {
        let t = Instant::now();
        let aq = self.server.generate_pseudorandom_query(0);
        let hint = self.server.multiply_with_db_ring(&aq, 0..self.shape.cols(), 0);
        let hint_ms = ms(t);
        let t = Instant::now();
        let n = POLY_LEN;
        let cols = self.shape.cols();
        let mut precomp = Vec::new();
        for i in 0..self.shape.instances {
            let mut v = Vec::with_capacity((n + 1) * n);
            for r in 0..n {
                v.extend_from_slice(&hint[r * cols + i * n..r * cols + (i + 1) * n]);
            }
            v.extend(std::iter::repeat(0u64).take(n));
            let prep = prep_pack_lwes(self.params, &v, n);
            precomp.push(precompute_pack(
                self.params,
                self.params.poly_len_log2,
                &prep,
                &self.fake_pp,
                &self.y_constants,
            ));
        }
        RefState { hint_ms, precomp_ms: ms(t), precomp }
    }

    /// The reference single-query path (YPIR's `perform_online_computation_simplepir`: one pass
    /// and one packing per query, single-threaded).
    pub fn answer_reference(
        &self,
        st: &RefState,
        query: &[u64],
        pp: &[PolyMatrixNTT<'static>],
    ) -> (Vec<Vec<u8>>, AnswerTiming) {
        let t0 = Instant::now();
        let rows = self.shape.rows;
        let cols = self.shape.cols();
        let mut buf = AlignedMemory64::new(rows);
        buf.as_mut_slice().copy_from_slice(query);
        let mut inter = AlignedMemory64::new(cols);
        fast_batched_dot_product_avx512::<1, u16>(
            self.params,
            inter.as_mut_slice(),
            buf.as_slice(),
            rows,
            self.server.db(),
            rows,
            cols,
        );
        let first_ms = ms(t0);
        let t1 = Instant::now();
        let q1 = self.params.get_q_prime_1();
        let q2 = self.params.get_q_prime_2();
        let out = (0..self.shape.instances)
            .map(|i| {
                let (pr, pv, pt) = &st.precomp[i];
                pack_using_precomp_vals(
                    self.params,
                    self.params.poly_len_log2,
                    pp,
                    &inter.as_slice()[i * POLY_LEN..(i + 1) * POLY_LEN],
                    pr,
                    pv,
                    pt,
                    &self.y_constants,
                )
                .raw()
                .switch(q1, q2)
            })
            .collect();
        let pack_ms = ms(t1);
        (out, AnswerTiming { first_ms, pack_ms, total_ms: ms(t0) })
    }
}

/// `res = sum_k w[k] * g[k]` (no reduction), `w` a condensed 1 x t matrix, `g` compact condensed.
fn mul_condensed(res: &mut PolyMatrixNTT, w: &PolyMatrixNTT, g: &[u64]) {
    let n = POLY_LEN;
    let t = w.cols;
    assert_eq!(g.len(), t * n);
    unsafe {
        let wp = w.as_slice().as_ptr();
        let gp = g.as_ptr();
        let rp = res.as_mut_slice().as_mut_ptr();
        for idx in (0..n).step_by(8) {
            let mut lo = _mm512_setzero_si512();
            let mut hi = _mm512_setzero_si512();
            for k in 0..t {
                let x = _mm512_loadu_si512(wp.add(k * 2 * n + idx) as *const _);
                let y = _mm512_loadu_si512(gp.add(k * n + idx) as *const _);
                lo = _mm512_add_epi64(lo, _mm512_mul_epu32(x, y));
                hi = _mm512_add_epi64(
                    hi,
                    _mm512_mul_epu32(_mm512_srli_epi64(x, 32), _mm512_srli_epi64(y, 32)),
                );
            }
            _mm512_storeu_si512(rp.add(idx) as *mut _, lo);
            _mm512_storeu_si512(rp.add(n + idx) as *mut _, hi);
        }
    }
}

/// YPIR's `precompute_pack` with each level's (independent) merges run in parallel and the
/// automorphism tables left out (they do not depend on the data).
pub fn precompute_pack_par(
    params: &'static Params,
    rlwe_cts: Vec<PolyMatrixNTT<'static>>,
    fake_pub_params: &[PolyMatrixNTT<'static>],
    y_constants: &YConst,
) -> Pc {
    let ell = params.poly_len_log2;
    assert!(fake_pub_params.len() == ell);
    assert_eq!(params.crt_count, 2);
    let mut working_set = rlwe_cts;
    let tn = params.t_exp_left * POLY_LEN;
    let mut res: Vec<u64> = Vec::with_capacity((params.poly_len - 1) * tn);
    for cur_ell in 1..=ell {
        let num_in = 1 << (ell - cur_ell + 1);
        let num_out = num_in >> 1;
        let (first_half, second_half) = (&mut working_set[..num_in]).split_at_mut(num_out);
        let (y, neg_y) = (&y_constants.0[cur_ell - 1], &y_constants.1[cur_ell - 1]);
        let t = (1 << cur_ell) + 1;
        let pub_param = &fake_pub_params[ell - 1 - (cur_ell - 1)];
        let level: Vec<Vec<u64>> = first_half
            .par_iter_mut()
            .zip(second_half.par_iter())
            .map_init(
                || {
                    (
                        PolyMatrixNTT::zero(params, 2, 1),
                        PolyMatrixNTT::zero(params, 2, 1),
                        PolyMatrixNTT::zero(params, 2, 1),
                        PolyMatrixRaw::zero(params, 1, 1),
                        PolyMatrixRaw::zero(params, 1, 1),
                        PolyMatrixRaw::zero(params, params.t_exp_left, 1),
                        PolyMatrixNTT::zero(params, params.t_exp_left, 1),
                        PolyMatrixNTT::zero(params, 1, 1),
                        PolyMatrixNTT::zero(params, 2, 1),
                        PolyMatrixNTT::zero(params, 2, 1),
                    )
                },
                |sc, (ct_even, ct_odd)| {
                    let (
                        y_times_ct_odd,
                        neg_y_times_ct_odd,
                        ct_sum_1,
                        ct_raw,
                        ct_auto,
                        ginv_ct,
                        ginv_ct_ntt,
                        ct_auto_1_ntt,
                        w_times_ginv_ct,
                        scratch,
                    ) = sc;
                    scalar_multiply_avx(y_times_ct_odd, y, ct_odd);
                    scalar_multiply_avx(neg_y_times_ct_odd, neg_y, ct_odd);
                    ct_sum_1.as_mut_slice().copy_from_slice(ct_even.as_slice());
                    add_into(ct_sum_1, neg_y_times_ct_odd);
                    fast_add_into_no_reduce(ct_even, y_times_ct_odd);
                    let scratch_mut_slc = scratch.as_mut_slice();
                    from_ntt_scratch(ct_raw, scratch_mut_slc, ct_sum_1);
                    automorph(ct_auto, ct_raw, t);
                    gadget_invert_rdim(ginv_ct, ct_auto, 1);
                    to_ntt(ginv_ct_ntt, ginv_ct);
                    automorph_poly_uncrtd(params, ct_auto_1_ntt.as_mut_slice(), scratch_mut_slc, t);
                    ntt_forward(params, ct_auto_1_ntt.as_mut_slice());
                    let mut condensed = Vec::with_capacity(tn);
                    for r in 0..params.t_exp_left {
                        let pl = ginv_ct_ntt.get_poly(r, 0);
                        condensed.extend((0..POLY_LEN).map(|z| pl[z] | (pl[z + POLY_LEN] << 32)));
                    }
                    w_times_ginv_ct.as_mut_slice().fill(0);
                    multiply_no_reduce(w_times_ginv_ct, pub_param, ginv_ct_ntt, 0);
                    add_into_at_no_reduce(ct_even, ct_auto_1_ntt, 1, 0);
                    add_into(ct_even, w_times_ginv_ct);
                    condensed
                },
            )
            .collect();
        for v in level {
            res.extend_from_slice(&v);
        }
    }
    (working_set.swap_remove(0), res)
}

type YConst = (Vec<PolyMatrixNTT<'static>>, Vec<PolyMatrixNTT<'static>>);

/// The query-independent part of YPIR's online packing (`pack_using_precomp_vals` up to the
/// point where the LWE bodies enter): it depends only on the client's packing keys and the
/// precomputation, so it can run before the query arrives.
pub fn pack_row1(
    params: &'static Params,
    pub_params: &[PolyMatrixNTT<'static>],
    precomp_vals: &[u64],
    precomp_tables: &[Vec<usize>],
    y_constants: &YConst,
) -> PolyMatrixNTT<'static> {
    let tn = params.t_exp_left * POLY_LEN;
    let ell = params.poly_len_log2;
    let mut working_set = Vec::with_capacity(1 << (ell - 1));
    for _ in 0..(1 << (ell - 1)) {
        working_set.push(PolyMatrixNTT::zero(params, 1, 1));
    }
    let mut y_times_ct_odd = PolyMatrixNTT::zero(params, 1, 1);
    let mut neg_y_times_ct_odd = PolyMatrixNTT::zero(params, 1, 1);
    let mut ct_sum_1 = PolyMatrixNTT::zero(params, 1, 1);
    let mut w_times_ginv_ct = PolyMatrixNTT::zero(params, 1, 1);
    let mut idx_precomp = 0;
    for cur_ell in 1..=ell {
        let mut num_in = 1 << (ell - cur_ell + 1);
        let num_out = num_in >> 1;
        if num_in == params.poly_len {
            num_in = num_out;
        }
        let (first_half, second_half) = (&mut working_set[..num_in]).split_at_mut(num_out);
        for i in 0..num_out {
            let ct_even = &mut first_half[i];
            let (y, neg_y) = (&y_constants.0[cur_ell - 1], &y_constants.1[cur_ell - 1]);
            if cur_ell > 1 {
                let ct_odd = &mut second_half[i];
                scalar_multiply_avx(&mut y_times_ct_odd, y, ct_odd);
                scalar_multiply_avx(&mut neg_y_times_ct_odd, neg_y, ct_odd);
                ct_sum_1.as_mut_slice().copy_from_slice(ct_even.as_slice());
                fast_add_into_no_reduce(&mut ct_sum_1, &neg_y_times_ct_odd);
                fast_add_into_no_reduce(ct_even, &y_times_ct_odd);
            }
            let t = (1 << cur_ell) + 1;
            let g = &precomp_vals[idx_precomp * tn..(idx_precomp + 1) * tn];
            idx_precomp += 1;
            let w = &pub_params[params.poly_len_log2 - 1 - (cur_ell - 1)];
            mul_condensed(&mut w_times_ginv_ct, w, g);
            if cur_ell > 1 {
                apply_automorph_ntt(params, precomp_tables, &ct_sum_1, ct_even, t);
                if i < num_out / 2 && ((cur_ell - 1) % 5 != 0) {
                    fast_add_into_no_reduce(ct_even, &w_times_ginv_ct);
                } else {
                    fast_add_into(ct_even, &w_times_ginv_ct);
                }
            } else if i < num_out / 2 {
                fast_add_into_no_reduce(ct_even, &w_times_ginv_ct);
            } else {
                fast_add_into(ct_even, &w_times_ginv_ct);
            }
        }
    }
    assert_eq!(idx_precomp * tn, precomp_vals.len());
    let mut row1 = working_set.swap_remove(0);
    fast_reduce(&mut row1);
    row1
}

/// The query-dependent rest: put the LWE bodies in and switch moduli.
pub fn pack_finish(
    params: &'static Params,
    precomp_res: &PolyMatrixNTT<'static>,
    row1: &PolyMatrixNTT<'static>,
    b_values: &[u64],
) -> Vec<u8> {
    let mut res = precomp_res.clone();
    res.get_poly_mut(1, 0).copy_from_slice(row1.as_slice());
    let mut out_raw = res.raw();
    for z in 0..params.poly_len {
        let val = barrett_reduction_u128(params, b_values[z] as u128 * params.poly_len as u128);
        let idx = params.poly_len + z;
        out_raw.data[idx] += val;
        if out_raw.data[idx] >= params.modulus {
            out_raw.data[idx] -= params.modulus;
        }
    }
    // the reference converts to NTT and back before switching; the raw form is the same
    out_raw.switch(params.get_q_prime_1(), params.get_q_prime_2())
}

/// A client's packing work for one seed, done before its query arrives.
pub struct Prepared {
    pub seed_j: usize,
    pub epoch: u64,
    row1: Vec<PolyMatrixNTT<'static>>,
}

fn row_bytes<F: Fn(usize) -> Vec<u8>>(s: &Shape, r: usize, record: &F) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.row_bytes());
    for k in 0..s.per_row {
        let idx = r * s.per_row + k;
        if idx < s.records {
            let b = record(idx);
            assert_eq!(b.len(), s.record_bytes);
            out.extend_from_slice(&b);
        } else {
            out.extend(std::iter::repeat(0u8).take(s.record_bytes));
        }
    }
    out
}

/// One client key; queries under it must use distinct seeds.
pub struct YpirClient {
    params: &'static Params,
    shape: Shape,
    y: YClient<'static>,
    pub pp: Vec<PolyMatrixNTT<'static>>,
}

impl YpirClient {
    pub fn new(shape: Shape, params: &'static Params) -> YpirClient {
        let c: &'static mut Client<'static> = Box::leak(Box::new(Client::init(params)));
        c.generate_secret_keys();
        let sk = c.get_sk_reg().clone();
        let pp_full = raw_generate_expansion_params(
            params,
            &sk,
            params.poly_len_log2,
            params.t_exp_left,
            &mut ChaCha20Rng::from_entropy(),
            &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
        );
        let pp = pp_full
            .iter()
            .map(|m| condense_matrix(params, &m.submatrix(1, 0, 1, m.cols)))
            .collect();
        let y = YClient::new(c, params);
        YpirClient { params, shape, y, pp }
    }

    /// Packed query for record `idx` under public seed `seed`.
    pub fn query(&self, seed: u8, idx: usize) -> AlignedMemory64 {
        let row = idx / self.shape.per_row;
        let q = self.y.generate_query(seed, self.params.db_dim_1, true, row);
        pack_query(self.params, &q)
    }

    pub fn decode_row(&self, resp: &[Vec<u8>]) -> Vec<u64> {
        let p = self.params;
        let mut el = Vec::with_capacity(resp.len() * POLY_LEN);
        for ct_bytes in resp {
            let ct = PolyMatrixRaw::recover(p, p.get_q_prime_1(), p.get_q_prime_2(), ct_bytes);
            let dec = self.y.client().decrypt_matrix_reg(&ct.ntt()).raw();
            for z in 0..POLY_LEN {
                el.push(rescale(dec.data[z], p.modulus, p.pt_modulus));
            }
        }
        el
    }

    pub fn decode(&self, idx: usize, resp: &[Vec<u8>]) -> Vec<u8> {
        let el = self.decode_row(resp);
        let row = elems_to_bytes(&el, self.shape.row_bytes());
        let off = (idx % self.shape.per_row) * self.shape.record_bytes;
        row[off..off + self.shape.record_bytes].to_vec()
    }
}
