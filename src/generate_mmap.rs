use multimap::MultiMap;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use base64::{engine::general_purpose, Engine as _};
use rand::Rng;
use rand::RngCore;

use dashmap::DashMap;
use rayon::prelude::*;
use std::sync::Mutex;

use blake3;

use cipher::KeyIvInit;

use chacha20::cipher::StreamCipher;
use chacha20::{ChaCha20, Key, Nonce};

use frodo_pir::db::{BaseParams, CommonParams, Database};
use frodo_pir::utils::lwe::{get_plaintext_size, get_rounding_factor};

use log::{error, info, trace};

use sonic_rs;
use sonic_rs::{JsonContainerTrait, JsonValueTrait};

use crate::dip;
use dip::{BfsDIP, HashDIP, MaskedItem};

pub struct EncryptedFileSystem {
    original_multimap: MultiMap<String, u32>,
    encrypted_multimap: MultiMap<Vec<u8>, Vec<u8>>,
    pub plaintext_encrypted_multimap: MultiMap<String, Vec<u8>>,
    pub single_string_array_docmap: Option<Vec<String>>,
    pub doc_chunk_counts: HashMap<u32, usize>,
    pub docmap_indices: HashMap<String, usize>,
    pub client_keys: HashMap<String, [u8; 32]>,
    pub ddr_server_store: HashMap<String, Vec<u8>>,
    pub ddr_doc_keys: ([u8; 32], [u8; 32]),
    pub ddr_query_seed: [u8; 32],
    iv: [u8; 16],
    encryption_keys: HashMap<Vec<u8>, [u8; 32]>,
    document_keys: HashMap<u32, [u8; 16]>,
    document_nonces: HashMap<u32, [u8; 12]>,
    keyword_count_map: HashMap<String, usize>,
    keyword_value_map: HashMap<String, Vec<u8>>,
    dip_result: Option<(HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>)>,
    dip_result_volmap: Option<(
        HashMap<usize, Vec<Vec<(String, usize)>>>,
        Vec<Vec<(String, usize)>>,
    )>,
    p: usize,
    use_bfs: bool,
    single_string_array: Option<Vec<String>>,
    single_string_array_volmap: Option<Vec<String>>,
    masked_valmap_buckets: Option<HashMap<usize, Vec<MaskedItem<Vec<u8>>>>>,
    masked_valmap_stash: Option<Vec<MaskedItem<Vec<u8>>>>,
    data_dir: String,
}

impl EncryptedFileSystem {
    pub fn new(
        p: usize,
        data_dir: String,
        use_bfs: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let original_multimap = Self::load_data(&data_dir)?;
        let encrypted_multimap = MultiMap::new();
        let plaintext_encrypted_multimap = MultiMap::new();
        let iv = Self::generate_iv();
        let encryption_keys = HashMap::new();
        let document_keys = Self::pre_generate_document_keys(&original_multimap)?;
        let document_nonces = HashMap::new();

        Ok(Self {
            data_dir,
            original_multimap,
            encrypted_multimap,
            plaintext_encrypted_multimap,
            single_string_array_docmap: None,
            doc_chunk_counts: HashMap::new(),
            docmap_indices: HashMap::new(),
            client_keys: HashMap::new(),
            ddr_server_store: HashMap::new(),
            ddr_doc_keys: (Self::generate_aes_key(), Self::generate_aes_key()),
            ddr_query_seed: Self::generate_aes_key(),
            iv,
            encryption_keys,
            document_keys,
            document_nonces,
            keyword_count_map: HashMap::new(),
            keyword_value_map: HashMap::new(),
            dip_result: None,
            dip_result_volmap: None,
            single_string_array: None,
            single_string_array_volmap: None,
            masked_valmap_buckets: None,
            masked_valmap_stash: None,
            p,
            use_bfs,
        })
    }

    fn load_data(data_dir: &str) -> Result<MultiMap<String, u32>, Box<dyn std::error::Error>> {
        let file = File::open(Path::new(data_dir).join("keyword_file_map.json"))?;
        let mut reader = BufReader::new(file);
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;

        let json_value: sonic_rs::Value = sonic_rs::from_str(&buf)?;

        let valid_file_names: std::collections::HashSet<String> =
            std::fs::read_to_string(Path::new(data_dir).join("files.txt"))?
                .lines()
                .filter_map(|loc| {
                    Path::new(loc)
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(String::from)
                })
                .collect();

        let mut multimap = MultiMap::new();

        if let Some(obj) = json_value.as_object() {
            for (keyword, value) in obj {
                if let Some(file_names) = value.as_array() {
                    for file_name_value in file_names {
                        if let Some(file_name) = file_name_value.as_str() {
                            if valid_file_names.contains(file_name) {
                                if let Ok(file_id) = file_name.parse::<u32>() {
                                    multimap.insert(keyword.to_string(), file_id);
                                }
                            } else {
                                error!("Warning: File '{}' not found in file list", file_name);
                            }
                        }
                    }
                }
            }
        }

        Ok(multimap)
    }

    fn generate_aes_key() -> [u8; 32] {
        let mut rng = rand::thread_rng();
        let mut key = [0u8; 32];
        rng.fill(&mut key);
        key
    }

    fn generate_chacha_key() -> [u8; 16] {
        let mut rng = rand::thread_rng();
        let mut key = [0u8; 16];
        rng.fill(&mut key);
        key
    }

    fn generate_nonce() -> [u8; 12] {
        let mut rng = rand::thread_rng();
        let mut nonce = [0u8; 12];
        rng.fill(&mut nonce);
        nonce
    }

    fn generate_iv() -> [u8; 16] {
        let mut rng = rand::thread_rng();
        let mut iv = [0u8; 16];
        rng.fill(&mut iv);
        iv
    }

    pub fn get_original_paths(&self, keyword: &str) -> Option<Vec<u32>> {
        self.original_multimap.get_vec(keyword).cloned()
    }

    fn hash_keyword(keyword: &str) -> Vec<u8> {
        blake3::hash(keyword.as_bytes()).as_bytes().to_vec()
    }

    fn pre_generate_document_keys(
        original_multimap: &MultiMap<String, u32>,
    ) -> Result<HashMap<u32, [u8; 16]>, Box<dyn std::error::Error>> {
        let mut document_keys = HashMap::new();

        for (_, file_paths) in original_multimap.iter_all() {
            for &file_path in file_paths {
                if !document_keys.contains_key(&file_path) {
                    let key = Self::generate_chacha_key();
                    document_keys.insert(file_path, key);
                }
            }
        }

        Ok(document_keys)
    }

    fn encrypt_data(&self, key: &[u8; 32], data: &[u8]) -> Vec<u8> {
        let nonce = Nonce::from_slice(&self.iv[..12]);
        let mut cipher = ChaCha20::new(Key::from_slice(key), nonce);
        let mut buf = data.to_vec();

        cipher.apply_keystream(&mut buf);
        buf
    }

    fn decrypt_data(&self, key: &[u8; 32], encrypted_data: &[u8]) -> Vec<u8> {
        let nonce = Nonce::from_slice(&self.iv[..12]);
        let mut cipher = ChaCha20::new(Key::from_slice(key), nonce);
        let mut buf = encrypted_data.to_vec();

        cipher.apply_keystream(&mut buf);
        buf
    }

    fn encrypt_document(&self, key: &[u8; 16], nonce: &[u8; 12], data: &[u8]) -> Vec<u8> {
        let mut extended_key = [0u8; 32];
        extended_key[..16].copy_from_slice(key);
        extended_key[16..].copy_from_slice(key);

        let key = Key::from_slice(&extended_key);
        let nonce = Nonce::from_slice(nonce);
        let mut cipher = ChaCha20::new(key, nonce);
        let mut buf = data.to_vec();
        cipher.apply_keystream(&mut buf);
        buf
    }

    fn decrypt_document(&self, key: &[u8; 16], nonce: &[u8; 12], encrypted_data: &[u8]) -> Vec<u8> {
        let mut extended_key = [0u8; 32];
        extended_key[..16].copy_from_slice(key);
        extended_key[16..].copy_from_slice(key);

        let key = Key::from_slice(&extended_key);
        let nonce = Nonce::from_slice(nonce);
        let mut cipher = ChaCha20::new(key, nonce);
        let mut buf = encrypted_data.to_vec();
        cipher.apply_keystream(&mut buf);
        buf
    }

    pub fn extract_insert_dataset(&mut self, split_ratio: f64) -> Vec<(String, Vec<u8>)> {
        let mut kv_pairs: Vec<_> = self.keyword_value_map.drain().collect();
        let split_idx = (kv_pairs.len() as f64 * split_ratio) as usize;
        let insert_kv = kv_pairs.split_off(split_idx);
        self.keyword_value_map.extend(kv_pairs);
        insert_kv
    }

    pub fn build_masked_valmap(
        &mut self,
        m: usize,
        p: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (masked_buckets, masked_stash) = if self.use_bfs {
            BfsDIP::<4>::new(m, p).build_stateful_valmap(&self.keyword_value_map)
        } else {
            HashDIP::<4>::new(m, p).build_stateful_valmap(&self.keyword_value_map)
        };

        self.masked_valmap_buckets = Some(masked_buckets);
        self.masked_valmap_stash = Some(masked_stash);
        Ok(())
    }

    pub fn run_incremental_inserts(
        &mut self,
        m: usize,
        p: usize,
        insert_data: Vec<(String, Vec<u8>)>,
    ) -> std::time::Duration {
        let mut buckets = self.masked_valmap_buckets.take().unwrap_or_default();
        let mut stash = self.masked_valmap_stash.take().unwrap_or_default();
        let mut rng = rand::thread_rng();

        let pir_dim = 1288;
        let elem_size = 65536;
        let p_bits = 9;
        let mock_pir_m = m;

        let dummy_elements = vec![
            base64::engine::general_purpose::STANDARD
                .encode(vec![0u8; elem_size]);
            mock_pir_m
        ];
        let mut pir_db = Database::new(&dummy_elements, mock_pir_m, elem_size, p_bits).unwrap();
        let mut pir_hint = BaseParams::new(&pir_db, pir_dim);

        let mut total_pir_update_time = std::time::Duration::new(0, 0);
        let mut last_touched_bucket = 0;

        let start_time = std::time::Instant::now();

        for (kw, val) in insert_data.iter() {
            let touched_buckets = if self.use_bfs {
                let dip = BfsDIP::<4>::new(m, p);
                dip.insert_valmap_oblivious(
                    kw,
                    Arc::new(val.clone()),
                    &mut buckets,
                    &mut stash,
                    &mut rng,
                )
            } else {
                let dip = HashDIP::<4>::new(m, p);
                dip.insert_valmap_oblivious(
                    kw,
                    Arc::new(val.clone()),
                    &mut buckets,
                    &mut stash,
                    &mut rng,
                )
            };

            let pir_start = std::time::Instant::now();
            let db_update_start = std::time::Instant::now();
            let mut pending_hint_updates = Vec::new();

            for &bucket_id in &touched_buckets {
                let pir_row_idx = bucket_id % mock_pir_m;

                let mut serialized_bucket = Vec::new();
                if let Some(bucket) = buckets.get(&bucket_id) {
                    for item in bucket {
                        serialized_bucket.extend_from_slice(item.key.as_bytes());
                        serialized_bucket.extend_from_slice(&*item.val);
                        for &mask_bit in &item.mask {
                            serialized_bucket.push(if mask_bit { 1 } else { 0 });
                        }
                    }
                }

                if serialized_bucket.len() < elem_size {
                    let current_len = serialized_bucket.len();
                    serialized_bucket.resize(elem_size, 0u8);
                    rng.fill(&mut serialized_bucket[current_len..]);
                } else if serialized_bucket.len() > elem_size {
                    serialized_bucket.truncate(elem_size);
                }

                let old_row = pir_db.update_row_from_bytes(pir_row_idx, &serialized_bucket);
                let new_row = pir_db.get_row(pir_row_idx);

                pending_hint_updates.push((pir_row_idx, old_row, new_row));

                last_touched_bucket = pir_row_idx;
            }

            let db_update_time = db_update_start.elapsed();

            let math_start = std::time::Instant::now();
            pir_hint.update_hints_batched(&pending_hint_updates);
            let math_time = math_start.elapsed();

            log::trace!(
                "DB Allocations/Base64 took: {:?}, Math took: {:?}",
                db_update_time,
                math_time
            );

            total_pir_update_time += pir_start.elapsed();
        }

        let elapsed = start_time.elapsed();

        let common_params = CommonParams::from(&pir_hint);
        let s: Vec<u32> = (0..pir_dim).map(|_| rng.gen()).collect();
        let mut query = common_params.mult_left(&s).unwrap();

        let q_factor = get_rounding_factor(p_bits);
        query[last_touched_bucket] = query[last_touched_bucket].wrapping_add(q_factor);

        let ans = pir_db.eval_single(&query);

        let hint_s = pir_hint.mult_right(&s).unwrap();
        let mut decrypted_row = Vec::with_capacity(ans.len());
        let p_size = get_plaintext_size(p_bits);

        for i in 0..ans.len() {
            let val = ans[i].wrapping_sub(hint_s[i]);
            let rounded = val.wrapping_add(q_factor / 2) / q_factor;
            decrypted_row.push(rounded % p_size);
        }

        if decrypted_row == pir_db.get_row(last_touched_bucket) {
            log::info!("PIR Validation Success.");
        } else {
            log::error!("PIR Validation Failed.");
        }

        let mut volmap_stash_additions = 0;
        for (kw, _) in insert_data.iter() {
            let keyword_part = kw.split("||").next().unwrap();
            let base_keyword = keyword_part.trim_end_matches('1');
            if !self.keyword_count_map.contains_key(base_keyword)
                && !self.keyword_count_map.contains_key(keyword_part)
            {
                volmap_stash_additions += 1;
            }
        }

        log::info!(
            "Finished {} oblivious inserts. VolMap simulated stash grew by {}.",
            insert_data.len(),
            volmap_stash_additions
        );
        log::info!("Total PIR Hint Update Time: {:?}", total_pir_update_time);

        self.masked_valmap_buckets = Some(buckets);
        self.masked_valmap_stash = Some(stash);

        elapsed
    }

    pub fn encrypt_files(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all("encrypted_docs")?;

        let unique_file_paths: std::collections::HashSet<u32> = self
            .original_multimap
            .iter_all()
            .flat_map(|(_, paths)| paths.clone())
            .collect();

        info!(
            "Starting parallel encryption of {} unique files...",
            unique_file_paths.len()
        );

        let concurrent_nonces = DashMap::new();

        unique_file_paths.par_iter().for_each(|&file_path| {
            trace!("  Encrypting file path: {}", file_path);

            let document_key = self
                .document_keys
                .get(&file_path)
                .expect("Document key not found");
            let nonce = Self::generate_nonce();
            concurrent_nonces.insert(file_path, nonce);

            let input_path = format!("{}/{}", self.data_dir, file_path);
            if let Ok(file_content) = std::fs::read(&input_path) {
                let encrypted_content = self.encrypt_document(document_key, &nonce, &file_content);

                let encrypted_file_path_output =
                    Path::new("encrypted_docs").join(format!("{}.enc", file_path));
                let nonce_file_path =
                    Path::new("encrypted_docs").join(format!("{}.nonce", file_path));

                std::fs::write(&encrypted_file_path_output, encrypted_content).unwrap();
                std::fs::write(&nonce_file_path, nonce).unwrap();
            } else {
                error!("Failed to read file: {}", input_path);
            }
        });

        self.document_nonces.extend(concurrent_nonces.into_iter());

        info!("File encryption complete. Starting metadata generation...");

        let encrypted_multimap = Mutex::new(MultiMap::new());
        let plaintext_encrypted_multimap = Mutex::new(MultiMap::new());
        let encryption_keys = DashMap::new();
        let keyword_value_map = DashMap::new();
        let keyword_count_map = DashMap::new();
        let shared_keyword_count = DashMap::new();
        let keyword_client_keys = DashMap::new();

        let entries: Vec<(&String, &Vec<u32>)> = self.original_multimap.iter_all().collect();

        entries.par_iter().for_each(|&(keyword, file_paths)| {
            let hashed_keyword = Self::hash_keyword(keyword);
            let random_key = Self::generate_aes_key();

            keyword_client_keys.insert(keyword.clone(), random_key);
            encryption_keys.insert(self.encrypt_data(&random_key, &hashed_keyword), random_key);
            keyword_count_map.insert(keyword.clone(), file_paths.len());

            // edge_index to guarantee absolute uniqueness for map keys
            for (edge_index, &file_path) in file_paths.iter().enumerate() {
                let document_key = self.document_keys.get(&file_path).unwrap();
                let nonce = self.document_nonces.get(&file_path).unwrap();

                let mut combined_path = [0u8; 32];
                combined_path[..4].copy_from_slice(&file_path.to_le_bytes());
                combined_path[4..20].copy_from_slice(document_key);
                combined_path[20..].copy_from_slice(nonce);

                let encrypted_file_path = self.encrypt_data(&random_key, &combined_path);

                encrypted_multimap.lock().unwrap().insert(
                    self.encrypt_data(&random_key, &hashed_keyword),
                    encrypted_file_path.clone(),
                );
                plaintext_encrypted_multimap
                    .lock()
                    .unwrap()
                    .insert(keyword.clone(), encrypted_file_path.clone());

                let mut count_ref = shared_keyword_count.entry(keyword.to_string()).or_insert(0);
                *count_ref += 1;
                let current_count = *count_ref;
                drop(count_ref);

                let modified_keyword = if current_count <= self.p {
                    keyword.to_string()
                } else {
                    let ones_to_append = (current_count - 1) / self.p;
                    let mut mk = keyword.to_string();
                    mk.push_str(&"1".repeat(ones_to_append));
                    mk
                };

                let combined_key = format!(
                    "{}||{:?}||{}",
                    modified_keyword, encrypted_file_path, edge_index
                );
                keyword_value_map.insert(combined_key, encrypted_file_path);
            }
        });

        self.encryption_keys.extend(encryption_keys.into_iter());
        self.keyword_value_map.extend(keyword_value_map.into_iter());
        self.keyword_count_map.extend(keyword_count_map.into_iter());
        self.client_keys.extend(keyword_client_keys.into_iter());
        self.encrypted_multimap = encrypted_multimap.into_inner().unwrap();
        self.plaintext_encrypted_multimap = plaintext_encrypted_multimap.into_inner().unwrap();

        for (kw, total_count) in shared_keyword_count.into_iter() {
            if total_count > self.p {
                let ones_to_append = (total_count - 1) / self.p;
                let new_count = total_count - (ones_to_append * self.p);
                let mut final_kw = kw;
                final_kw.push_str(&"1".repeat(ones_to_append));
                self.keyword_count_map.insert(final_kw, new_count);
            }
        }

        info!("Encryption and metadata generation successful!");
        Ok(())
    }

    pub fn verify_decryption(&self) -> Result<bool, Box<dyn std::error::Error>> {
        for (encrypted_keyword, encrypted_file_paths) in self.encrypted_multimap.iter_all() {
            let random_key = self
                .encryption_keys
                .get(encrypted_keyword)
                .ok_or("Encryption key not found")?;

            let decrypted_hashed_keyword = self.decrypt_data(random_key, encrypted_keyword);
            let decrypted_hashed_keyword_str =
                String::from_utf8_lossy(&decrypted_hashed_keyword).into_owned();

            trace!("Verifying hashed keyword: {}", decrypted_hashed_keyword_str);

            let original_keyword = self
                .original_multimap
                .keys()
                .find(|&k| {
                    let hashed = Self::hash_keyword(k);
                    hashed == decrypted_hashed_keyword
                })
                .ok_or("No matching original keyword found for hash")?;

            trace!("Matched original keyword: {}", original_keyword);

            for encrypted_file_path in encrypted_file_paths {
                let decrypted_combined_path = self.decrypt_data(random_key, encrypted_file_path);

                let file_path =
                    u32::from_le_bytes(decrypted_combined_path[..4].try_into().unwrap());
                let document_key: [u8; 16] = decrypted_combined_path[4..20].try_into().unwrap();

                trace!("  Decrypted file path: {}", file_path);

                if let Some(stored_key) = self.document_keys.get(&file_path) {
                    if *stored_key != document_key {
                        error!("Key mismatch for file: {}", file_path);
                        return Ok(false);
                    }
                } else {
                    error!("No stored key found for file: {}", file_path);
                    return Ok(false);
                }

                let nonce_file_name = format!("{}.nonce", file_path);
                let nonce_file_path = Path::new("encrypted_docs").join(&nonce_file_name);
                let mut nonce_file = File::open(&nonce_file_path)?;
                let mut nonce = [0u8; 12];
                nonce_file.read_exact(&mut nonce)?;

                if let Some(original_paths) = self.original_multimap.get_vec(original_keyword) {
                    if !original_paths.contains(&file_path) {
                        error!("  Original paths for this keyword:");
                        for path in original_paths {
                            error!("    {}", path);
                        }
                        error!("Decryption verification failed for file path");
                        return Ok(false);
                    }
                } else {
                    error!(
                        "  No original paths found for keyword: {}",
                        original_keyword
                    );
                    return Ok(false);
                }

                let encrypted_file_name = format!("{}.enc", file_path);
                let encrypted_file_path = Path::new("encrypted_docs").join(&encrypted_file_name);
                let mut encrypted_file = File::open(&encrypted_file_path)?;
                let mut encrypted_content = Vec::new();
                encrypted_file.read_to_end(&mut encrypted_content)?;

                let decrypted_content =
                    self.decrypt_document(&document_key, &nonce, &encrypted_content);

                let mut original_file = File::open(format!("data/{}", file_path))?;
                let mut original_content = Vec::new();
                original_file.read_to_end(&mut original_content)?;

                if decrypted_content != original_content {
                    error!("Decryption verification failed for file content");
                    error!("  Original file size: {}", original_content.len());
                    error!("  Decrypted file size: {}", decrypted_content.len());
                    return Ok(false);
                }
            }
        }

        info!("Decryption verification successful!");
        Ok(true)
    }

    fn create_modified_keyword_count_map(&self) -> HashMap<String, usize> {
        let mut modified_map = HashMap::new();

        for (keyword, &count) in &self.keyword_count_map {
            let mut current_keyword = keyword.clone();
            let mut remaining_count = count;

            while remaining_count > 0 {
                let current_count = remaining_count.min(self.p);
                *modified_map.entry(current_keyword.clone()).or_insert(0) += current_count;
                remaining_count -= current_count;

                if remaining_count > 0 {
                    current_keyword.push('1');
                }
            }
        }

        modified_map
    }

    pub fn build_ddr_store(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_size = 1024;
        let mut store = HashMap::new();

        for entry in std::fs::read_dir("encrypted_docs")? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("enc") {
                let file_stem = path.file_stem().unwrap().to_str().unwrap();
                let file_id: u32 = file_stem.parse()?;

                let content = std::fs::read(&path)?;
                let mut chunks = content.chunks(chunk_size);
                let mut chunk_idx = 0;

                while let Some(chunk) = chunks.next() {
                    let mut padded_chunk = vec![0u8; chunk_size];
                    padded_chunk[..chunk.len()].copy_from_slice(chunk);

                    let data_to_hash = format!("doc_{}_{}", file_id, chunk_idx);

                    let mut hasher1 = blake3::Hasher::new_keyed(&self.ddr_doc_keys.0);
                    hasher1.update(data_to_hash.as_bytes());
                    let addr1 = base64::engine::general_purpose::STANDARD
                        .encode(hasher1.finalize().as_bytes());

                    let mut hasher2 = blake3::Hasher::new_keyed(&self.ddr_doc_keys.1);
                    hasher2.update(data_to_hash.as_bytes());
                    let addr2 = base64::engine::general_purpose::STANDARD
                        .encode(hasher2.finalize().as_bytes());

                    store.insert(addr1, padded_chunk.clone());
                    store.insert(addr2, padded_chunk);

                    chunk_idx += 1;
                }
                self.doc_chunk_counts.insert(file_id, chunk_idx);
            }
        }

        self.ddr_server_store = store;
        Ok(())
    }

    pub fn apply_dip(
        &mut self,
        m: usize,
        p: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let modified_keyword_count_map = self.create_modified_keyword_count_map();
        let result = if self.use_bfs {
            let dip = BfsDIP::<4>::new(m, p);
            dip.build_valmap(&modified_keyword_count_map, &self.keyword_value_map)
        } else {
            let dip = HashDIP::<4>::new(m, p);
            dip.build_valmap(&modified_keyword_count_map, &self.keyword_value_map)
        };

        self.dip_result = Some(result);
        Ok(())
    }

    pub fn apply_dip_on_volmap(
        &mut self,
        m: usize,
        p: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let keyword_count_as_value_map: HashMap<String, Vec<(String, usize)>> = self
            .keyword_count_map
            .iter()
            .map(|(k, &v)| (k.clone(), vec![(k.clone(), v)]))
            .collect();

        let result = if self.use_bfs {
            let dip = BfsDIP::<4>::new(m, p);
            dip.build_volmap(&self.keyword_count_map, &keyword_count_as_value_map)
        } else {
            let dip = HashDIP::<4>::new(m, p);
            dip.build_volmap(&self.keyword_count_map, &keyword_count_as_value_map)
        };

        self.dip_result_volmap = Some(result);
        Ok(())
    }

    pub fn build_and_apply_docmap(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_size = 1024;
        let mut flat_array = Vec::new();
        let mut docmap_indices = HashMap::new();

        for entry in std::fs::read_dir("encrypted_docs")? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("enc") {
                let file_stem = path.file_stem().unwrap().to_str().unwrap();
                let file_id: u32 = file_stem.parse()?;

                let content = std::fs::read(&path)?;
                let mut chunks = content.chunks(chunk_size);
                let mut chunk_idx = 0;

                while let Some(chunk) = chunks.next() {
                    let mut padded_chunk = vec![0u8; chunk_size];
                    padded_chunk[..chunk.len()].copy_from_slice(chunk);

                    let mut key_str = format!("doc_{}_", file_id);
                    key_str.push_str(&"1".repeat(chunk_idx));

                    let current_index = flat_array.len();
                    flat_array.push(padded_chunk);
                    docmap_indices.insert(key_str, current_index);
                    chunk_idx += 1;
                }
                self.doc_chunk_counts.insert(file_id, chunk_idx);
            }
        }

        self.docmap_indices = docmap_indices;

        let encoded_array: Vec<String> = flat_array
            .into_par_iter()
            .map(|chunk| base64::engine::general_purpose::STANDARD.encode(&chunk))
            .collect();

        self.single_string_array_docmap = Some(encoded_array);
        Ok(())
    }

    pub fn decrypt_data_public(&self, key: &[u8; 32], encrypted_data: &[u8]) -> Vec<u8> {
        self.decrypt_data(key, encrypted_data)
    }

    pub fn get_dip_stats(&self) -> Option<(usize, usize, usize)> {
        self.dip_result.as_ref().map(|(buckets, stash)| {
            let total_buckets = buckets.len();
            let non_empty_buckets = buckets.values().filter(|b| !b.is_empty()).count();
            let stash_size = stash.len();
            (total_buckets, non_empty_buckets, stash_size)
        })
    }

    pub fn get_dip_stats_volmap(&self) -> Option<(usize, usize, usize)> {
        self.dip_result_volmap
            .as_ref()
            .map(|(buckets, stash)| {
                let total_buckets = buckets.len();
                let non_empty_buckets = buckets.values().filter(|b| !b.is_empty()).count();
                let stash_size = stash.len();
                (total_buckets, non_empty_buckets, stash_size)
            })
    }

    pub fn add_padding_to_buckets(&mut self) {
        if let Some((buckets, _)) = self.dip_result.as_mut() {
            let target_size = 8192;
            for bucket in buckets.values_mut() {
                let current_size: usize = bucket.iter().map(|item| item.len()).sum();
                let padding_needed = target_size - current_size;
                let mut padding = vec![0u8; padding_needed];
                rand::thread_rng().fill_bytes(&mut padding);
                bucket.push(padding);
            }
        }
    }

    pub fn generate_single_string_array(&mut self, m: usize) {
        self.single_string_array = self.dip_result.as_ref().map(|(buckets, _)| {
            let mut single_array = vec![String::new(); m];

            let target_size = 8192;
            let b64_len = (target_size * 4 / 3) + 4;

            let encoded_buckets: Vec<(usize, String)> = buckets
                .par_iter()
                .map(|(&bucket_id, bucket_content)| {
                    let combined_bucket_data: Vec<u8> =
                        bucket_content.iter().flatten().cloned().collect();
                    let mut encoded = String::with_capacity(b64_len);
                    general_purpose::STANDARD.encode_string(&combined_bucket_data, &mut encoded);
                    (bucket_id, encoded)
                })
                .collect();

            for (id, enc) in encoded_buckets {
                if id < m {
                    single_array[id] = enc;
                }
            }
            single_array
        });
    }

    pub fn generate_single_string_array_from_volmap(
        &mut self,
        m: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (buckets, _) = self
            .dip_result_volmap
            .as_ref()
            .ok_or("DIP has not been applied")?;

        let elem_size = 256;

        let encode_vec = |v: &Vec<Vec<(String, usize)>>| {
            let mut buffer = vec![0u8; elem_size];
            let mut pos = 0;

            if let Some(pairs) = v.first() {
                if let Some((keyword, count)) = pairs.first() {
                    let keyword_bytes = keyword.as_bytes();
                    let write_len = keyword_bytes.len().min(251);

                    buffer[pos] = write_len as u8;
                    pos += 1;

                    buffer[pos..pos + write_len].copy_from_slice(&keyword_bytes[..write_len]);
                    pos += write_len;

                    buffer[pos..pos + 4].copy_from_slice(&(*count as u32).to_le_bytes());
                }
            }
            general_purpose::STANDARD.encode(&buffer)
        };

        let mut single_array = vec![general_purpose::STANDARD.encode(&vec![0u8; elem_size]); m];

        for (bucket_id, bucket) in buckets {
            if *bucket_id < m {
                single_array[*bucket_id] = encode_vec(bucket);
            }
        }

        self.single_string_array_volmap = Some(single_array);
        Ok(())
    }

    pub fn get_valmap_stash(&self) -> Option<Vec<Vec<u8>>> {
        self.dip_result
            .as_ref()
            .map(|(_, stash)| stash.clone())
    }

    pub fn get_single_string_array_volmap(&self) -> Option<&Vec<String>> {
        self.single_string_array_volmap.as_ref()
    }

    pub fn print_dip_distribution(&self) {
        if let Some((buckets, stash)) = &self.dip_result {
            trace!("\nTethysDIP Distribution:");
            for (bucket_id, values) in buckets {
                trace!("Bucket {}: {} items", bucket_id, values.len());
            }
            trace!("Stash: {} items", stash.len());
        } else {
            trace!("TethysDIP has not been applied yet.");
        }
    }

    pub fn print_volmap_dip_distribution(&self) {
        if let Some((buckets, stash)) = &self.dip_result_volmap {
            trace!("\nTethysDIP Distribution:");
            for (bucket_id, values) in buckets {
                trace!("Bucket {}: {} items", bucket_id, values.len());
            }
            trace!("Stash: {} items", stash.len());
        } else {
            trace!("TethysDIP has not been applied yet.");
        }
    }

    pub fn fill_empty_index(&mut self) {
        if let Some(single_array) = &mut self.single_string_array {
            let non_empty_size = single_array
                .iter()
                .filter(|&s| !s.is_empty())
                .map(|s| s.len())
                .next()
                .unwrap_or(0);

            if non_empty_size == 0 {
                error!("No non-empty indexes found to determine size.");
                return;
            }

            let random_data_size = (non_empty_size * 3) / 4;

            for index in 0..single_array.len() {
                if single_array[index].is_empty() {
                    let mut random_data = vec![0u8; random_data_size];
                    rand::thread_rng().fill_bytes(&mut random_data);
                    single_array[index] = general_purpose::STANDARD.encode(&random_data);

                    if single_array[index].len() > non_empty_size {
                        single_array[index].truncate(non_empty_size);
                    } else {
                        while single_array[index].len() < non_empty_size {
                            single_array[index].push('=');
                        }
                    }
                }
            }
            info!(
                "Empty indexes in ValMap have been filled with random data of size {} bytes.",
                non_empty_size
            );
        } else {
            error!("Single string array not generated yet. Please call generate_single_string_array() first.");
        }
    }

    pub fn fill_empty_index_volmap(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(single_array) = &mut self.single_string_array_volmap {
            let elem_size = 256;

            let keywords_path = Path::new(&self.data_dir).join("keywords.txt");
            let keywords: Vec<String> = std::fs::read_to_string(&keywords_path)
                .map_err(|e| {
                    format!(
                        "Failed to read keywords.txt from {}: {}",
                        keywords_path.display(),
                        e
                    )
                })?
                .lines()
                .map(String::from)
                .collect();

            let mut rng = rand::thread_rng();

            let mut indices_to_fill = Vec::new();
            for (index, value) in single_array.iter().enumerate() {
                if value.is_empty() {
                    indices_to_fill.push(index);
                    continue;
                }

                if let Ok(bytes) = general_purpose::STANDARD.decode(value) {
                    if bytes.iter().all(|&b| b == 0) {
                        indices_to_fill.push(index);
                    }
                } else {
                    indices_to_fill.push(index);
                }
            }

            for index in indices_to_fill {
                let mut buffer = vec![0u8; elem_size];

                let random_keyword = keywords
                    .choose(&mut rng)
                    .ok_or(format!("Keywords list is empty"))?
                    .to_string();

                let keyword_bytes = random_keyword.as_bytes();
                let write_len = keyword_bytes.len().min(251);

                buffer[0] = write_len as u8;
                buffer[1..1 + write_len].copy_from_slice(&keyword_bytes[..write_len]);

                let random_count = rng.gen_range(1..10);
                buffer[1 + write_len..1 + write_len + 4]
                    .copy_from_slice(&(random_count as u32).to_le_bytes());

                single_array[index] = general_purpose::STANDARD.encode(&buffer);
            }

            Ok(())
        } else {
            Err("Single string array for volmap not generated yet.".into())
        }
    }

    pub fn print_single_string_array_volmap(&self) {
        if let Some(single_array) = &self.single_string_array_volmap {
            trace!("\nSingle-Dimensional String Array Representation for Volmap:");
            for (i, bucket_data) in single_array.iter().enumerate() {
                if bucket_data.is_empty() {
                    trace!("Bucket {}: Empty", i);
                } else {
                    trace!("Bucket {}: {} characters", i, bucket_data.len());
                }
            }
        } else {
            error!("Single string array for volmap not generated yet.");
        }
    }

    pub fn get_single_string_array(&self) -> Option<Vec<String>> {
        self.single_string_array.clone()
    }

    pub fn get_p(&self) -> usize {
        self.p
    }

    pub fn print_single_string_array(&self) {
        if let Some(single_array) = &self.single_string_array {
            trace!("\nSingle-Dimensional String Array Representation:");
            for (i, bucket_data) in single_array.iter().enumerate() {
                if bucket_data.is_empty() {
                    trace!("Bucket {}: Empty", i);
                } else {
                    trace!("Bucket {}: {} characters", i, bucket_data.len());
                }
            }
        } else {
            error!("Single string array not generated yet.");
        }
    }

    pub fn decode_base64(&self, encoded: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let decoded = general_purpose::STANDARD.decode(encoded)?;
        Ok(decoded)
    }

    #[allow(dead_code)]
    pub fn print_keyword_count_map(&self) {
        info!("\nKeyword Count Map:");
        for (keyword, count) in &self.keyword_count_map {
            info!("{}: {}", keyword, count);
        }
    }

    #[allow(dead_code)]
    pub fn print_keyword_value_map(&self) {
        info!("\nKeyword Value Map:");
        for (combined_key, value) in &self.keyword_value_map {
            info!("{}: {:?}", combined_key, value);
        }
    }

    #[allow(dead_code)]
    pub fn get_keyword_count_map_size(&self) -> usize {
        self.keyword_count_map.len()
    }

    #[allow(dead_code)]
    pub fn get_keyword_value_map_size(&self) -> usize {
        self.keyword_value_map.len()
    }

    #[allow(dead_code)]
    pub fn print_original_data(&self) {
        info!("Original Multimap:");
        for (keyword, file_paths) in self.original_multimap.iter_all() {
            info!("Keyword: {}", keyword);
            for file_path in file_paths {
                info!("  File Path: {}", file_path);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_encrypted_data(&self) {
        info!("Encrypted Multimap:");
        for (encrypted_keyword, encrypted_file_paths) in self.encrypted_multimap.iter_all() {
            info!("Encrypted Keyword: {:?}", encrypted_keyword);
            for encrypted_file_path in encrypted_file_paths {
                info!("  Encrypted File Path: {:?}", encrypted_file_path);
            }
        }
    }

    #[allow(dead_code)]
    pub fn print_plaintext_encrypted_data(&self) {
        info!("Plaintext-Encrypted Multimap:");
        for (keyword, encrypted_file_paths) in self.plaintext_encrypted_multimap.iter_all() {
            info!("Keyword: {}", keyword);
            for encrypted_file_path in encrypted_file_paths {
                info!("  Encrypted File Path: {:?}", encrypted_file_path);
            }
        }
    }
}
