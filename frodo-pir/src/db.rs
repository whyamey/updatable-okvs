use std::fs;
use std::io::BufReader;
use std::simd::Simd;

use rand::rngs::StdRng;
use rand_core::{OsRng, RngCore, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::errors::ResultBoxedError;
use crate::utils::format::*;
use crate::utils::matrices::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Database {
  entries: Vec<u32>,
  m: usize,
  width: usize,
  elem_size: usize,
  plaintext_bits: usize,
}

impl Database {
  pub fn new(
    elements: &[String],
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let rows = construct_rows(elements, m, elem_size, plaintext_bits)?;
    let width = if rows.is_empty() { 0 } else { rows[0].len() };

    Ok(Self {
      entries: crate::utils::matrices::flatten_matrix_row_major(&rows),
      m,
      width,
      elem_size,
      plaintext_bits,
    })
  }

  pub fn from_file(
    db_file: &str,
    m: usize,
    elem_size: usize,
    plaintext_bits: usize,
  ) -> ResultBoxedError<Self> {
    let file_contents: String = fs::read_to_string(db_file)?.parse()?;
    let elements: Vec<String> = serde_json::from_str(&file_contents)?;
    Self::new(&elements, m, elem_size, plaintext_bits)
  }

  pub fn vec_mult(&self, row: &[u32], col_idx: usize) -> u32 {
    let mut acc = 0u32;
    for i in 0..self.m {
      acc = acc.wrapping_add(
        row[i].wrapping_mul(self.entries[i * self.width + col_idx]),
      );
    }
    acc
  }

  /// Generic batched matrix multiplication for N interleaved queries.
  /// Because N is a compile-time constant, LLVM will hopefully unroll
  /// the inner loop and auto-vectorize it to AVX/NEON instructions.
  /// Now with explicit std::simd and as a macro.

  pub fn write_to_file(&self, path: &str) -> ResultBoxedError<()> {
    let mut entries_2d = Vec::with_capacity(self.width);
    for col in 0..self.width {
      let mut column = Vec::with_capacity(self.m);
      for row in 0..self.m {
        column.push(self.entries[row * self.width + col]);
      }
      entries_2d.push(column);
    }
    let json = serde_json::json!(entries_2d);
    Ok(serde_json::to_writer(&std::fs::File::create(path)?, &json)?)
  }

  /// Row-Major single query evaluator
  pub fn eval_single(&self, q: &[u32]) -> Vec<u32> {
    assert_eq!(q.len(), self.m);

    let row_chunk_size = 1024;
    (0..self.m)
      .into_par_iter()
      .chunks(row_chunk_size)
      .fold(
        || vec![0u32; self.width],
        |mut local_acc, row_indices| {
          for row in row_indices {
            let q_val = q[row];
            let row_offset = row * self.width;
            let db_row = &self.entries[row_offset..row_offset + self.width];

            for col in 0..self.width {
              local_acc[col] =
                local_acc[col].wrapping_add(q_val.wrapping_mul(db_row[col]));
            }
          }
          local_acc
        },
      )
      .reduce(
        || vec![0u32; self.width],
        |mut acc_a, acc_b| {
          for col in 0..self.width {
            acc_a[col] = acc_a[col].wrapping_add(acc_b[col]);
          }
          acc_a
        },
      )
  }

  /// Updates a single row (bucket) with a new base64 string, returning the old row.
  pub fn update_row(
    &mut self,
    row_idx: usize,
    new_data: &str,
  ) -> ResultBoxedError<Vec<u32>> {
    let old_row = self.get_row(row_idx);

    let bytes = base64::decode(new_data)?;
    let bits = crate::utils::format::bytes_to_bits_le(&bytes);

    let mut new_row = Vec::with_capacity(self.width);
    for i in 0..self.width {
      let end_bound = (i + 1) * self.plaintext_bits;
      if end_bound < bits.len() {
        new_row.push(crate::utils::format::bits_to_u32_le(
          &bits[i * self.plaintext_bits..end_bound],
        )?);
      } else {
        new_row.push(crate::utils::format::bits_to_u32_le(
          &bits[i * self.plaintext_bits..],
        )?);
      }
    }

    let start = row_idx * self.width;
    self.entries[start..start + self.width].copy_from_slice(&new_row);

    Ok(old_row)
  }

  /// Update row without the allocations by directly using raw bytes
  pub fn update_row_from_bytes(
    &mut self,
    row_idx: usize,
    bytes: &[u8],
  ) -> Vec<u32> {
    let old_row = self.get_row(row_idx);

    let mut new_row = Vec::with_capacity(self.width);
    let mut bit_buffer = 0u64;
    let mut bits_in_buffer = 0;
    let mut byte_idx = 0;

    let mask = (1 << self.plaintext_bits) - 1;

    for _ in 0..self.width {
      while bits_in_buffer < self.plaintext_bits {
        if byte_idx < bytes.len() {
          bit_buffer |= (bytes[byte_idx] as u64) << bits_in_buffer;
          byte_idx += 1;
          bits_in_buffer += 8;
        } else {
          bits_in_buffer += self.plaintext_bits;
        }
      }

      new_row.push((bit_buffer & mask) as u32);

      bit_buffer >>= self.plaintext_bits;
      bits_in_buffer -= self.plaintext_bits;
    }

    let start = row_idx * self.width;
    self.entries[start..start + self.width].copy_from_slice(&new_row);

    old_row
  }

  /// Returns the ith row of the DB matrix
  pub fn get_row(&self, i: usize) -> Vec<u32> {
    let start = i * self.width;
    self.entries[start..start + self.width].to_vec()
  }

  /// Returns the ith DB entry as a base64-encoded string
  pub fn get_db_entry(&self, i: usize) -> String {
    base64_from_u32_slice(&self.get_row(i), self.plaintext_bits, self.elem_size)
  }

  /// Returns the width of the DB matrix
  pub fn get_matrix_width(element_size: usize, plaintext_bits: usize) -> usize {
    let mut quo = element_size / plaintext_bits;
    if element_size % plaintext_bits != 0 {
      quo += 1;
    }
    quo
  }

  /// Returns the width of the DB matrix
  pub fn get_matrix_width_self(&self) -> usize {
    self.width
  }

  /// Get the matrix size
  pub fn get_matrix_height(&self) -> usize {
    self.m
  }

  /// Get the element size
  pub fn get_elem_size(&self) -> usize {
    self.elem_size
  }

  /// Get the plaintext bits
  pub fn get_plaintext_bits(&self) -> usize {
    self.plaintext_bits
  }
}

/// The `BaseParams` object allows loading and interacting with params that
/// are used by the client for constructing queries
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BaseParams {
  dim: usize, // the lwe dimension

  m: usize,         // the size of the DB
  elem_size: usize, // the size (in bits) of each element of the DB. Corresponds to `w` in paper.
  plaintext_bits: usize,

  public_seed: [u8; 32],
  rhs: Vec<u32>,
}

impl BaseParams {
  pub fn new(db: &Database, dim: usize) -> Self {
    let public_seed = generate_seed(); // generates the public seed
    Self {
      public_seed,
      rhs: Self::generate_params_rhs(db, public_seed, dim, db.m),
      dim,
      m: db.m,
      elem_size: db.elem_size,
      plaintext_bits: db.plaintext_bits,
    }
  }

  /// Load params from a JSON file
  pub fn load(params_path: &str) -> ResultBoxedError<Self> {
    let reader = BufReader::new(fs::File::open(params_path)?);
    Ok(serde_json::from_reader(reader)?)
  }

  /// Generates the RHS of the params using the database and the seed
  /// for the LHS
  pub fn generate_params_rhs(
    db: &Database,
    public_seed: [u8; 32],
    dim: usize,
    m: usize,
  ) -> Vec<u32> {
    let lhs =
      swap_matrix_fmt(&generate_lwe_matrix_from_seed(public_seed, dim, m));
    let width = db.get_matrix_width_self();

    let mut c_rows: Vec<Vec<u32>> = Vec::with_capacity(dim);
    let mut row_idx = 0;

    // Bulk Processing with 64 as its fastest
    while row_idx + 64 <= dim {
      let mut interleaved = vec![[0u32; 64]; m];
      for i in 0..m {
        for j in 0..64 {
          interleaved[i][j] = lhs[row_idx + j][i];
        }
      }
      let batched_res = db.eval_batched_64(&interleaved);
      for j in 0..64 {
        let mut row = Vec::with_capacity(width);
        for col in 0..width {
          row.push(batched_res[col][j]);
        }
        c_rows.push(row);
      }
      row_idx += 64;
    }

    // Cascade down to 32
    if row_idx + 32 <= dim {
      let mut interleaved = vec![[0u32; 32]; m];
      for i in 0..m {
        for j in 0..32 {
          interleaved[i][j] = lhs[row_idx + j][i];
        }
      }
      let batched_res = db.eval_batched_32(&interleaved);
      for j in 0..32 {
        let mut row = Vec::with_capacity(width);
        for col in 0..width {
          row.push(batched_res[col][j]);
        }
        c_rows.push(row);
      }
      row_idx += 32;
    }

    // Cascade down to 16
    if row_idx + 16 <= dim {
      let mut interleaved = vec![[0u32; 16]; m];
      for i in 0..m {
        for j in 0..16 {
          interleaved[i][j] = lhs[row_idx + j][i];
        }
      }
      let batched_res = db.eval_batched_16(&interleaved);
      for j in 0..16 {
        let mut row = Vec::with_capacity(width);
        for col in 0..width {
          row.push(batched_res[col][j]);
        }
        c_rows.push(row);
      }
      row_idx += 16;
    }

    // Pad remaining into an 16 or 8 evaluator
    if row_idx < dim {
      let leftover = dim - row_idx;

      // If we have 8 or fewer leftovers, use the cheapest 8 block.
      if leftover <= 8 {
        let mut interleaved = vec![[0u32; 8]; m];
        for i in 0..m {
          for j in 0..leftover {
            interleaved[i][j] = lhs[row_idx + j][i];
          }
        }
        let batched_res = db.eval_batched_8(&interleaved);
        for j in 0..leftover {
          // Only extract the valid leftovers!
          let mut row = Vec::with_capacity(width);
          for col in 0..width {
            row.push(batched_res[col][j]);
          }
          c_rows.push(row);
        }
      } else {
        // If we have 9-15 leftovers, pad them into a 16x block.
        let mut interleaved = vec![[0u32; 16]; m];
        for i in 0..m {
          for j in 0..leftover {
            interleaved[i][j] = lhs[row_idx + j][i];
          }
        }
        let batched_res = db.eval_batched_16(&interleaved);
        for j in 0..leftover {
          let mut row = Vec::with_capacity(width);
          for col in 0..width {
            row.push(batched_res[col][j]);
          }
          c_rows.push(row);
        }
      }
    }

    let mut flat_col_major = Vec::with_capacity(width * dim);
    for col in 0..width {
      for row in 0..dim {
        flat_col_major.push(c_rows[row][col]);
      }
    }

    flat_col_major
  }

  /// Writes the params struct as JSON to file
  pub fn write_to_file(&self, path: &str) -> ResultBoxedError<()> {
    let json = json!({
      "public_seed": self.public_seed,
      "rhs": self.rhs,
    });
    Ok(serde_json::to_writer(&fs::File::create(path)?, &json)?)
  }

  /// Computes c = s*(A*DB) using the RHS of the public parameters
  pub fn mult_right(&self, s: &[u32]) -> ResultBoxedError<Vec<u32>> {
    let width = self.rhs.len() / self.dim;
    let res: Vec<u32> = (0..width)
      .into_par_iter()
      .map(|i| {
        let start = i * self.dim;
        let col = &self.rhs[start..start + self.dim];
        vec_mult_u32_u32(s, col).unwrap()
      })
      .collect();
    Ok(res)
  }

  pub fn mult_right_batched_n<const N: usize>(
    &self,
    s_interleaved: &[[u32; N]],
  ) -> ResultBoxedError<Vec<[u32; N]>> {
    let width = self.rhs.len() / self.dim;
    let res: Vec<[u32; N]> = (0..width)
      .into_par_iter()
      .map(|col_idx| {
        let start = col_idx * self.dim;
        let col = &self.rhs[start..start + self.dim];
        assert_eq!(s_interleaved.len(), self.dim);

        let mut acc = [0u32; N];
        for i in 0..self.dim {
          let c = col[i];
          let s = s_interleaved[i];
          for j in 0..N {
            acc[j] = acc[j].wrapping_add(s[j].wrapping_mul(c));
          }
        }
        acc
      })
      .collect();
    Ok(res)
  }

  pub fn update_hint_for_row(
    &mut self,
    row_idx: usize,
    old_row: &[u32],
    new_row: &[u32],
  ) {
    let width = self.rhs.len() / self.dim;
    assert_eq!(old_row.len(), width);
    assert_eq!(new_row.len(), width);

    let mut hasher = Sha256::new();
    hasher.update(self.public_seed);
    hasher.update((row_idx as u64).to_le_bytes());
    let sub_seed: [u8; 32] = hasher.finalize().into();

    let mut rng = StdRng::from_seed(sub_seed);
    let mut a_col = Vec::with_capacity(self.dim);
    for _ in 0..self.dim {
      a_col.push(rng.next_u32());
    }

    let mut delta = Vec::with_capacity(width);
    for j in 0..width {
      delta.push(new_row[j].wrapping_sub(old_row[j]));
    }

    self
      .rhs
      .par_chunks_exact_mut(self.dim)
      .enumerate()
      .for_each(|(j, rhs_col)| {
        let delta_j = delta[j];
        if delta_j == 0 {
          return;
        }

        const LANES: usize = 16;
        let mut i = 0;
        let delta_simd = Simd::<u32, LANES>::splat(delta_j);

        while i + LANES <= self.dim {
          let a_simd = Simd::<u32, LANES>::from_slice(&a_col[i..i + LANES]);
          let rhs_simd = Simd::<u32, LANES>::from_slice(&rhs_col[i..i + LANES]);

          let res = rhs_simd + (a_simd * delta_simd);
          res.copy_to_slice(&mut rhs_col[i..i + LANES]);

          i += LANES;
        }

        for k in i..self.dim {
          rhs_col[k] = rhs_col[k].wrapping_add(a_col[k].wrapping_mul(delta_j));
        }
      });
  }

  /// Updates multiple rows in a single pass over the RHS matrix.
  pub fn update_hints_batched(
    &mut self,
    updates: &[(usize, Vec<u32>, Vec<u32>)],
  ) {
    if updates.is_empty() {
      return;
    }

    let width = self.rhs.len() / self.dim;

    let mut precomputed = Vec::with_capacity(updates.len());
    for (row_idx, old_row, new_row) in updates {
      assert_eq!(old_row.len(), width);
      assert_eq!(new_row.len(), width);

      let mut hasher = Sha256::new();
      hasher.update(self.public_seed);
      hasher.update((*row_idx as u64).to_le_bytes());
      let sub_seed: [u8; 32] = hasher.finalize().into();

      let mut rng = StdRng::from_seed(sub_seed);
      let mut a_col = Vec::with_capacity(self.dim);
      for _ in 0..self.dim {
        a_col.push(rng.next_u32());
      }

      let mut delta = Vec::with_capacity(width);
      for j in 0..width {
        delta.push(new_row[j].wrapping_sub(old_row[j]));
      }
      precomputed.push((a_col, delta));
    }

    self
      .rhs
      .par_chunks_exact_mut(self.dim)
      .enumerate()
      .for_each(|(j, rhs_col)| {
        // Extract only the updates that actually change this specific column
        let mut active_updates = Vec::new();
        for (a_col, delta) in &precomputed {
          if delta[j] != 0 {
            active_updates.push((a_col, delta[j]));
          }
        }

        if active_updates.is_empty() {
          return;
        }

        const LANES: usize = 16;
        let mut i = 0;

        // Apply all 10 active updates simultaneously using SIMD vectors
        while i + LANES <= self.dim {
          let mut rhs_simd =
            Simd::<u32, LANES>::from_slice(&rhs_col[i..i + LANES]);

          for (a_col, delta_j) in &active_updates {
            let delta_simd = Simd::<u32, LANES>::splat(*delta_j);
            let a_simd = Simd::<u32, LANES>::from_slice(&a_col[i..i + LANES]);
            rhs_simd += a_simd * delta_simd;
          }

          rhs_simd.copy_to_slice(&mut rhs_col[i..i + LANES]);
          i += LANES;
        }

        for k in i..self.dim {
          for (a_col, delta_j) in &active_updates {
            rhs_col[k] =
              rhs_col[k].wrapping_add(a_col[k].wrapping_mul(*delta_j));
          }
        }
      });
  }

  pub fn get_total_records(&self) -> usize {
    self.m
  }

  pub fn get_dim(&self) -> usize {
    self.dim
  }

  pub fn get_elem_size(&self) -> usize {
    self.elem_size
  }

  pub fn get_plaintext_bits(&self) -> usize {
    self.plaintext_bits
  }
}

/// `CommonParams` holds the derived uniform matrix that is used for
/// constructing the server's public parameters and the client query.
#[derive(Serialize, Deserialize)]
pub struct CommonParams {
  matrix: Vec<u32>,
  m: usize,
  dim: usize,
}

impl CommonParams {
  // Returns the internal matrix
  pub fn as_matrix_flat(&self) -> &[u32] {
    &self.matrix
  }

  /// Computes b = s*A + e using the seed used to generate the matrix of
  /// the public parameters
  pub fn mult_left(&self, s: &[u32]) -> ResultBoxedError<Vec<u32>> {
    let res: Vec<u32> = (0..self.m)
      .into_par_iter()
      .map(|i| {
        let start = i * self.dim;
        let col = &self.matrix[start..start + self.dim];
        let s_a = vec_mult_u32_u32(s, col).unwrap();
        let e = random_ternary();
        s_a.wrapping_add(e)
      })
      .collect();
    Ok(res)
  }

  pub fn mult_left_batched_n<const N: usize>(
    &self,
    s_interleaved: &[[u32; N]],
  ) -> ResultBoxedError<Vec<[u32; N]>> {
    let res: Vec<[u32; N]> = (0..self.m)
      .into_par_iter()
      .map(|col_idx| {
        let start = col_idx * self.dim;
        let col = &self.matrix[start..start + self.dim];
        assert_eq!(s_interleaved.len(), self.dim);

        let mut acc = [0u32; N];
        for i in 0..self.dim {
          let c = col[i];
          let s = s_interleaved[i];
          for j in 0..N {
            acc[j] = acc[j].wrapping_add(s[j].wrapping_mul(c));
          }
        }
        // Add random ternary error for each batched element
        for j in 0..N {
          acc[j] = acc[j].wrapping_add(random_ternary());
        }
        acc
      })
      .collect();
    Ok(res)
  }
}

impl From<&BaseParams> for CommonParams {
  fn from(params: &BaseParams) -> Self {
    let cols =
      generate_lwe_matrix_from_seed(params.public_seed, params.dim, params.m);
    Self {
      matrix: cols.into_iter().flatten().collect(),
      m: params.m,
      dim: params.dim,
    }
  }
}

fn construct_rows(
  elements: &[String],
  m: usize,
  elem_size: usize,
  plaintext_bits: usize,
) -> ResultBoxedError<Vec<Vec<u32>>> {
  let row_width = Database::get_matrix_width(elem_size, plaintext_bits);

  let result = (0..m)
    .into_par_iter()
    .map(|i| -> ResultBoxedError<Vec<u32>> {
      let mut row = Vec::with_capacity(row_width);
      let data = &elements[i];
      let bytes = base64::decode(data)?;
      let bits = bytes_to_bits_le(&bytes);
      for i in 0..row_width {
        let end_bound = (i + 1) * plaintext_bits;
        if end_bound < bits.len() {
          row.push(bits_to_u32_le(&bits[i * plaintext_bits..end_bound])?);
        } else {
          row.push(bits_to_u32_le(&bits[i * plaintext_bits..])?);
        }
      }
      Ok(row)
    });

  result.collect()
}

/// Generic batched matrix multiplication for N interleaved queries.
/// Because N is a compile-time constant, LLVM will hopefully unroll
/// the inner loop and auto-vectorize it to AVX/NEON instructions.
/// Now with explicit std::simd and as a macro.
macro_rules! impl_db_eval_batched {
  ($n:expr, $method_name:ident) => {
    impl Database {
      pub fn $method_name(
        &self,
        q_interleaved: &[[u32; $n]],
      ) -> Vec<[u32; $n]> {
        assert_eq!(q_interleaved.len(), self.m);

        let row_chunk_size = 1024;

        (0..self.m)
          .into_par_iter()
          .chunks(row_chunk_size)
          .fold(
            || vec![Simd::<u32, $n>::splat(0); self.width],
            |mut local_acc, row_indices| {
              for row in row_indices {
                let q_vec = Simd::<u32, $n>::from_array(q_interleaved[row]);
                let row_offset = row * self.width;
                let db_row = &self.entries[row_offset..row_offset + self.width];

                for col in 0..self.width {
                  let db_val = Simd::<u32, $n>::splat(db_row[col]);
                  local_acc[col] += q_vec * db_val;
                }
              }
              local_acc
            },
          )
          .reduce(
            || vec![Simd::<u32, $n>::splat(0); self.width],
            |mut acc_a, acc_b| {
              for col in 0..self.width {
                acc_a[col] += acc_b[col];
              }
              acc_a
            },
          )
          .into_iter()
          .map(|simd_val| simd_val.to_array())
          .collect()
      }
    }
  };
}

/// Macro for N > 64. Relies on LLVM auto-vectorization to unroll N into
/// multiple AVX-512 registers, since std::simd maxes out at 64 lanes.
macro_rules! impl_db_eval_batched_large {
  ($n:expr, $method_name:ident) => {
    impl Database {
      pub fn $method_name(
        &self,
        q_interleaved: &[[u32; $n]],
      ) -> Vec<[u32; $n]> {
        assert_eq!(q_interleaved.len(), self.m);

        // Keep cache tiling active
        let row_chunk_size = 1024;

        (0..self.m)
          .into_par_iter()
          .chunks(row_chunk_size)
          .fold(
            || vec![[0u32; $n]; self.width],
            |mut local_acc, row_indices| {
              for row in row_indices {
                let q_vec = &q_interleaved[row];
                let row_offset = row * self.width;
                let db_row = &self.entries[row_offset..row_offset + self.width];

                for col in 0..self.width {
                  let db_val = db_row[col];
                  // LLVM will hopefully unroll this inner loop into AVX-512 instructions
                  for j in 0..$n {
                    local_acc[col][j] = local_acc[col][j]
                      .wrapping_add(q_vec[j].wrapping_mul(db_val));
                  }
                }
              }
              local_acc
            },
          )
          .reduce(
            || vec![[0u32; $n]; self.width],
            |mut acc_a, acc_b| {
              for col in 0..self.width {
                for j in 0..$n {
                  acc_a[col][j] = acc_a[col][j].wrapping_add(acc_b[col][j]);
                }
              }
              acc_a
            },
          )
      }
    }
  };
}

impl_db_eval_batched!(2, eval_batched_2);
impl_db_eval_batched!(4, eval_batched_4);
impl_db_eval_batched!(8, eval_batched_8);
impl_db_eval_batched!(16, eval_batched_16);
impl_db_eval_batched!(32, eval_batched_32);
impl_db_eval_batched!(64, eval_batched_64);

impl_db_eval_batched_large!(128, eval_batched_128);
impl_db_eval_batched_large!(256, eval_batched_256);
impl_db_eval_batched_large!(512, eval_batched_512);
impl_db_eval_batched_large!(1024, eval_batched_1024);

fn generate_seed() -> [u8; 32] {
  let mut seed = [0u8; 32];
  OsRng.fill_bytes(&mut seed);
  seed
}
