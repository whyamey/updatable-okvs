//! PIR backends behind one interface, so the UOKVS benchmarks can swap them.
//!
//! The requirements for the PIR in our setting are:
//!   R1 single server; R2 many clients without per-client server state;
//!   R3 the owner updates rows continuously and clients' state is cheap to
//!      refresh; R4 records of a few KB; R5 batches of 4..800 queries.
//! `FrodoPir` wraps the optimized FrodoPIR fork used in the paper. YPIR and
//! other schemes are measured through their own Rust CLIs (see
//! `scripts/run_ypir.sh`) because their APIs do not expose row updates.

use base64::{engine::general_purpose, Engine as _};
use frodo_pir::api::{
    BatchedQuery8, BatchedQueryParams16, BatchedQueryParams32, BatchedQueryParams64, BatchedQueryParams8,
    QueryParams, Response,
};
use frodo_pir::db::{BaseParams, CommonParams, Database};
use rand::Rng;
use std::time::Instant;

pub const LWE_DIM: usize = 1288;
pub const PT_BITS: usize = 9;

/// FrodoPIR matrix width (u32 entries per row) for rows of `elem_bytes`.
pub fn frodo_width(elem_bytes: usize) -> usize {
    (elem_bytes * 8 + PT_BITS - 1) / PT_BITS
}
/// Client hint size (bytes) for rows of `elem_bytes`; independent of the row count.
pub fn frodo_hint_bytes(elem_bytes: usize) -> usize {
    LWE_DIM * frodo_width(elem_bytes) * 4
}
/// Bytes a client downloads to refresh its hint for one changed row.
pub fn frodo_row_delta_bytes(elem_bytes: usize, packed: bool) -> usize {
    let w = frodo_width(elem_bytes);
    if packed {
        (w * (PT_BITS + 1) + 7) / 8 + 8
    } else {
        w * 4 + 8
    }
}

/// Body of a batched query of width N (8 is `query_batch`).
macro_rules! batched_impl {
    ($self:ident, $idx:ident, $n:expr, $bqp:ident, $eval:ident) => {{
        let mut c = QueryCost { queries: $idx.len(), ..Default::default() };
        let rows = $self.rows;
        let elem_bytes = $self.elem_bits / 8;
        $self.common();
        let cp = $self.cp.as_ref().unwrap();
        let mut outs = Vec::with_capacity($idx.len());
        for chunk in $idx.chunks($n) {
            let t = Instant::now();
            let mut bqp = $bqp::new(cp, &$self.bp).expect("bqp");
            let mut qps: Vec<QueryParams> = (0..$n).map(|i| bqp.extract_query_params(i).unwrap()).collect();
            c.client_prep_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut queries = Vec::with_capacity($n);
            for k in 0..$n {
                let i = *chunk.get(k).unwrap_or(&chunk[0]);
                let q = match qps[k].generate_query(i) {
                    Ok(q) => q,
                    Err(_) => loop {
                        let t2 = Instant::now();
                        let mut fresh = QueryParams::new(cp, &$self.bp).expect("qp");
                        c.client_prep_ms += t2.elapsed().as_secs_f64() * 1e3;
                        if let Ok(q) = fresh.generate_query(i) {
                            qps[k] = fresh;
                            break q;
                        }
                    },
                };
                queries.push(q);
            }
            let interleaved: Vec<[u32; $n]> = (0..rows)
                .map(|r| {
                    let mut v = [0u32; $n];
                    for j in 0..$n {
                        v[j] = queries[j].as_slice()[r];
                    }
                    v
                })
                .collect();
            c.client_query_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let res = $self.db.$eval(&interleaved);
            c.server_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            for k in 0..chunk.len() {
                let single = Response(res.iter().map(|r| r[k]).collect());
                let mut b = single.parse_output_as_bytes(&qps[k]);
                b.truncate(elem_bytes);
                outs.push(b);
            }
            c.client_decode_ms += t.elapsed().as_secs_f64() * 1e3;
            c.upload_bytes += chunk.len() * $self.query_bytes();
            c.download_bytes += chunk.len() * $self.response_bytes();
        }
        (c, outs)
    }};
}

#[derive(Clone, Debug, Default)]
pub struct UpdateCost {
    pub rows: usize,
    pub server_db_ms: f64,
    pub server_hint_ms: f64,
    /// bytes a client must download to refresh its hint if the server sends
    /// the raw row differences (width x u32 per row), as the fork does
    pub client_bytes_raw: usize,
    /// same, with differences packed into ceil(log2(2^pt)+1) bits per entry
    pub client_bytes_packed: usize,
    /// client CPU to apply the refresh (it recomputes A's column per row)
    pub client_apply_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct QueryCost {
    pub queries: usize,
    /// per-query client preprocessing (s*A, s*hint); can be done offline
    pub client_prep_ms: f64,
    pub client_query_ms: f64,
    pub server_ms: f64,
    pub client_decode_ms: f64,
    pub upload_bytes: usize,
    pub download_bytes: usize,
}

pub struct FrodoPir {
    pub db: Database,
    pub bp: BaseParams,
    pub cp: Option<CommonParams>,
    pub rows: usize,
    pub elem_bits: usize,
}

impl FrodoPir {
    /// Builds the database and the hint; returns the server setup time (ms).
    pub fn new(rows: &[Vec<u8>], elem_bytes: usize) -> (Self, f64) {
        let t = Instant::now();
        let strs: Vec<String> = rows.iter().map(|r| {
            let mut b = r.clone();
            b.resize(elem_bytes, 0);
            general_purpose::STANDARD.encode(b)
        }).collect();
        let elem_bits = elem_bytes * 8;
        let db = Database::new(&strs, rows.len(), elem_bits, PT_BITS).expect("db");
        let bp = BaseParams::new(&db, LWE_DIM);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (Self { db, bp, cp: None, rows: rows.len(), elem_bits }, ms)
    }

    pub fn width(&self) -> usize {
        self.db.get_matrix_width_self()
    }
    /// Client hint (the matrix M = A * D): dim x width u32.
    pub fn hint_bytes(&self) -> usize {
        LWE_DIM * self.width() * 4
    }
    /// Size of one query (b~ in Z_q^m) and one response (Z_q^width).
    pub fn query_bytes(&self) -> usize {
        self.rows * 4
    }
    pub fn response_bytes(&self) -> usize {
        self.width() * 4
    }

    /// Applies a batch of row updates (one UOKVS operation's accesses).
    pub fn update_rows(&mut self, rows: &[(usize, Vec<u8>)]) -> UpdateCost {
        let t = Instant::now();
        let mut pend = Vec::with_capacity(rows.len());
        for (i, bytes) in rows {
            let old = self.db.update_row_from_bytes(*i, bytes);
            let new = self.db.get_row(*i);
            pend.push((*i, old, new));
        }
        let db_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        self.bp.update_hints_batched(&pend);
        let hint_ms = t.elapsed().as_secs_f64() * 1e3;
        let w = self.width();
        let packed_bits = PT_BITS + 1;
        UpdateCost {
            rows: rows.len(),
            server_db_ms: db_ms,
            server_hint_ms: hint_ms,
            client_bytes_raw: rows.len() * (w * 4 + 8),
            client_bytes_packed: rows.len() * ((w * packed_bits + 7) / 8 + 8),
            // a client applies the same computation as the server's hint update
            client_apply_ms: hint_ms,
        }
    }

    fn common(&mut self) -> &CommonParams {
        if self.cp.is_none() {
            self.cp = Some(CommonParams::from(&self.bp));
        }
        self.cp.as_ref().unwrap()
    }

    /// Runs PIR queries in groups of `w` (8, 16, 32 or 64) per database pass,
    /// using the fork's `BatchedQueryParams{w}` and `eval_batched_{w}`.
    pub fn query_batch_w(&mut self, idx: &[usize], w: usize) -> (QueryCost, Vec<Vec<u8>>) {
        if idx.is_empty() {
            return (QueryCost::default(), vec![]);
        }
        match w {
            8 => self.query_batch(idx),
            16 => batched_impl!(self, idx, 16, BatchedQueryParams16, eval_batched_16),
            32 => batched_impl!(self, idx, 32, BatchedQueryParams32, eval_batched_32),
            64 => batched_impl!(self, idx, 64, BatchedQueryParams64, eval_batched_64),
            _ => panic!("batch width must be 8, 16, 32 or 64"),
        }
    }

    /// Runs a batch of PIR queries in groups of 8 and returns the rows.
    pub fn query_batch(&mut self, idx: &[usize]) -> (QueryCost, Vec<Vec<u8>>) {
        let mut c = QueryCost { queries: idx.len(), ..Default::default() };
        let rows = self.rows;
        let elem_bytes = self.elem_bits / 8;
        self.common();
        let cp = self.cp.as_ref().unwrap();
        let mut outs = Vec::with_capacity(idx.len());
        for chunk in idx.chunks(8) {
            let t = Instant::now();
            let mut bqp = BatchedQueryParams8::new(cp, &self.bp).expect("bqp");
            let mut qps: Vec<QueryParams> = (0..8).map(|i| bqp.extract_query_params(i).unwrap()).collect();
            c.client_prep_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let mut queries = Vec::with_capacity(8);
            for k in 0..8 {
                let i = *chunk.get(k).unwrap_or(&chunk[0]);
                // FrodoPIR rejects a query whose indicator addition overflows;
                // like the original bench, retry with fresh parameters.
                let q = match qps[k].generate_query(i) {
                    Ok(q) => q,
                    Err(_) => loop {
                        let t2 = Instant::now();
                        let mut fresh = QueryParams::new(cp, &self.bp).expect("qp");
                        c.client_prep_ms += t2.elapsed().as_secs_f64() * 1e3;
                        if let Ok(q) = fresh.generate_query(i) {
                            qps[k] = fresh;
                            break q;
                        }
                    },
                };
                queries.push(q);
            }
            let interleaved: Vec<[u32; 8]> = (0..rows)
                .map(|r| {
                    let mut v = [0u32; 8];
                    for j in 0..8 {
                        v[j] = queries[j].as_slice()[r];
                    }
                    v
                })
                .collect();
            let bq = BatchedQuery8 { interleaved };
            c.client_query_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            let res = self.db.eval_batched_8(&bq.interleaved);
            c.server_ms += t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            for k in 0..chunk.len() {
                let single = Response(res.iter().map(|r| r[k]).collect());
                let mut b = single.parse_output_as_bytes(&qps[k]);
                b.truncate(elem_bytes);
                outs.push(b);
            }
            c.client_decode_ms += t.elapsed().as_secs_f64() * 1e3;
            c.upload_bytes += chunk.len() * self.query_bytes();
            c.download_bytes += chunk.len() * self.response_bytes();
        }
        (c, outs)
    }
}


/// Random database rows for microbenchmarks.
pub fn random_rows<R: Rng>(n: usize, elem_bytes: usize, rng: &mut R) -> Vec<Vec<u8>> {
    (0..n)
        .map(|_| {
            let mut v = vec![0u8; elem_bytes];
            rng.fill(&mut v[..]);
            v
        })
        .collect()
}


// ---------------------------------------------------------------- Spiral

/// Spiral (hintless) through our faster server (`spiral_fast`) and the Blyss
/// client. Rows of up to 2 KB use n = 1, up to 8 KB n = 2; the table is padded
/// to 2^(9 + nu2) records. The client keeps only its public parameters, so a
/// row update costs it nothing.
pub struct SpiralPir {
    pub params: &'static spiral_rs::params::Params,
    client: spiral_rs::client::Client<'static>,
    pp: spiral_rs::client::PublicParameters<'static>,
    pub db: spiral_fast::DenseDb,
    pub rows: usize,
    pub elem_bytes: usize,
    pub pp_bytes: usize,
}

impl SpiralPir {
    pub fn new(rows: &[Vec<u8>], elem_bytes: usize) -> (Self, f64) {
        let t = Instant::now();
        let n = if elem_bytes <= 2048 { 1 } else if elem_bytes <= 8192 { 2 } else { panic!("Spiral rows up to 8 KB") };
        let total = rows.len().next_power_of_two().max(1 << 12);
        let nu2 = total.trailing_zeros() as usize - 9;
        let params: &'static spiral_rs::params::Params = Box::leak(Box::new(spiral_fast::table_params(n, 9, nu2)));
        let mut client = spiral_rs::client::Client::init(params);
        let pp = client.generate_keys();
        let pp_bytes = pp.serialize().len();
        let mut db = spiral_fast::DenseDb::zero(params);
        for (i, r) in rows.iter().enumerate() {
            db.update_item(params, i, &r[..r.len().min(elem_bytes)]);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        (Self { params, client, pp, db, rows: rows.len(), elem_bytes, pp_bytes }, ms)
    }

    pub fn update_rows(&mut self, rows: &[(usize, Vec<u8>)]) -> UpdateCost {
        let t = Instant::now();
        for (i, b) in rows {
            self.db.update_item(self.params, *i, &b[..b.len().min(self.elem_bytes)]);
        }
        UpdateCost { rows: rows.len(), server_db_ms: t.elapsed().as_secs_f64() * 1e3, ..Default::default() }
    }

    /// Answers queries in batches of `w` that share one pass over the database.
    pub fn query_batch_w(&mut self, idx: &[usize], w: usize) -> (QueryCost, Vec<Vec<u8>>) {
        let mut c = QueryCost { queries: idx.len(), ..Default::default() };
        let mut outs = Vec::with_capacity(idx.len());
        for chunk in idx.chunks(w.max(1)) {
            let t = Instant::now();
            let qs: Vec<spiral_rs::client::Query> = chunk.iter().map(|&i| self.client.generate_query(i)).collect();
            c.client_query_ms += t.elapsed().as_secs_f64() * 1e3;
            c.upload_bytes += qs.iter().map(|q| q.serialize().len()).sum::<usize>();
            let (resp, tm) = spiral_fast::process_queries(self.params, &self.pp, &qs, &self.db);
            c.server_ms += tm.total_ms;
            let t = Instant::now();
            for r in &resp {
                c.download_bytes += r.len();
                let mut b = self.client.decode_response(r);
                b.truncate(self.elem_bytes);
                outs.push(b);
            }
            c.client_decode_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        (c, outs)
    }
}

/// One PIR database of either kind.
pub enum Pir {
    Frodo(FrodoPir),
    Spiral(SpiralPir),
}

impl Pir {
    pub fn new(kind: &str, rows: &[Vec<u8>], elem_bytes: usize) -> (Self, f64) {
        match kind {
            "spiral" => {
                let (p, ms) = SpiralPir::new(rows, elem_bytes);
                (Pir::Spiral(p), ms)
            }
            _ => {
                let (p, ms) = FrodoPir::new(rows, elem_bytes);
                (Pir::Frodo(p), ms)
            }
        }
    }
    pub fn update_rows(&mut self, rows: &[(usize, Vec<u8>)]) -> UpdateCost {
        match self {
            Pir::Frodo(p) => p.update_rows(rows),
            Pir::Spiral(p) => p.update_rows(rows),
        }
    }
    pub fn query_batch_w(&mut self, idx: &[usize], w: usize) -> (QueryCost, Vec<Vec<u8>>) {
        match self {
            Pir::Frodo(p) => p.query_batch_w(idx, w),
            Pir::Spiral(p) => p.query_batch_w(idx, w),
        }
    }
    /// Client state that depends on the database (FrodoPIR hint) or not (Spiral public parameters).
    pub fn client_state_bytes(&self) -> usize {
        match self {
            Pir::Frodo(p) => p.hint_bytes(),
            Pir::Spiral(p) => p.pp_bytes,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Pir::Frodo(_) => "frodo",
            Pir::Spiral(_) => "spiral",
        }
    }
}
