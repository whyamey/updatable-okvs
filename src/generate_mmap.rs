use multimap::MultiMap;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::Path;

use base64::{engine::general_purpose, Engine as _};
use rand::Rng;
use rand::RngCore;

use blake3;

use cipher::KeyIvInit;

use chacha20::cipher::StreamCipher;
use chacha20::{ChaCha20, Key, Nonce};

use log::{error, info, trace};

use sonic_rs;
use sonic_rs::{JsonContainerTrait, JsonValueTrait};

use crate::dip;
use dip::{BfsDIP, HashDIP};

pub struct EncryptedFileSystem {
    original_multimap: MultiMap<String, u32>,
    encrypted_multimap: MultiMap<Vec<u8>, Vec<u8>>,
    pub plaintext_encrypted_multimap: MultiMap<String, Vec<u8>>,
    iv: [u8; 16],
    encryption_keys: HashMap<Vec<u8>, [u8; 32]>,
    document_keys: HashMap<u32, [u8; 16]>,
    document_nonces: HashMap<u32, [u8; 12]>,
    keyword_count_map: HashMap<String, usize>,
    keyword_value_map: HashMap<String, Vec<u8>>,
    tethys_dip_result: Option<(HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>)>,
    tethys_dip_result_volmap: Option<(
        HashMap<usize, Vec<Vec<(String, usize)>>>,
        Vec<Vec<(String, usize)>>,
    )>,
    p: usize,
    use_bfs: bool,
    single_string_array: Option<Vec<String>>,
    single_string_array_volmap: Option<Vec<String>>,
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
            iv,
            encryption_keys,
            document_keys,
            document_nonces,
            keyword_count_map: HashMap::new(),
            keyword_value_map: HashMap::new(),
            tethys_dip_result: None,
            tethys_dip_result_volmap: None,
            single_string_array: None,
            single_string_array_volmap: None,
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

        let file_locations: Vec<String> =
            std::fs::read_to_string(Path::new(data_dir).join("files.txt"))?
                .lines()
                .map(String::from)
                .collect();

        let mut multimap = MultiMap::new();

        if let Some(obj) = json_value.as_object() {
            for (keyword, value) in obj {
                if let Some(file_names) = value.as_array() {
                    for file_name_value in file_names {
                        if let Some(file_name) = file_name_value.as_str() {
                            if let Some(_) =
                                file_locations.iter().find(|&loc| loc.ends_with(file_name))
                            {
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

    pub fn encrypt_files(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut encrypted_files = std::collections::HashSet::new();
        let mut keyword_count = std::collections::HashMap::new();

        std::fs::create_dir_all("encrypted_docs")?;

        for (keyword, file_paths) in self.original_multimap.iter_all() {
            trace!("Encrypting Keyword: {}", keyword);
            let hashed_keyword = Self::hash_keyword(&keyword);
            trace!("Hashed keyword: {:?}", hashed_keyword);
            let random_key = Self::generate_aes_key();
            let encrypted_keyword = self.encrypt_data(&random_key, &hashed_keyword);

            self.encryption_keys
                .insert(encrypted_keyword.clone(), random_key);

            for &file_path in file_paths {
                trace!("  Encrypting file path: {}", file_path);
                let document_key = self
                    .document_keys
                    .get(&file_path)
                    .ok_or("Document key not found")?;

                let nonce = Self::generate_nonce();
                self.document_nonces.insert(file_path, nonce);

                let mut combined_path = [0u8; 32];
                combined_path[..4].copy_from_slice(&file_path.to_le_bytes());
                combined_path[4..20].copy_from_slice(document_key);
                combined_path[20..].copy_from_slice(&nonce);

                let encrypted_file_path = self.encrypt_data(&random_key, &combined_path);

                if !encrypted_files.contains(&file_path) {
                    let mut file = File::open(format!("{}/{}", self.data_dir, file_path))?;
                    let mut file_content = Vec::new();
                    file.read_to_end(&mut file_content)?;

                    let encrypted_content =
                        self.encrypt_document(document_key, &nonce, &file_content);

                    let encrypted_file_name = format!("{}.enc", file_path);
                    let encrypted_file_path_output =
                        Path::new("encrypted_docs").join(&encrypted_file_name);

                    let mut encrypted_file = File::create(&encrypted_file_path_output)?;
                    encrypted_file.write_all(&encrypted_content)?;

                    let nonce_file_name = format!("{}.nonce", file_path);
                    let nonce_file_path = Path::new("encrypted_docs").join(&nonce_file_name);
                    let mut nonce_file = File::create(&nonce_file_path)?;
                    nonce_file.write_all(&nonce)?;

                    trace!(
                        "  Encrypted file saved as: {:?}",
                        encrypted_file_path_output
                    );

                    encrypted_files.insert(file_path);
                }

                self.encrypted_multimap
                    .insert(encrypted_keyword.clone(), encrypted_file_path.clone());
                self.plaintext_encrypted_multimap
                    .insert(keyword.clone(), encrypted_file_path.clone());

                let modified_keyword = self.get_modified_keyword(keyword, &mut keyword_count);

                let combined_key = format!("{}||{:?}", modified_keyword, encrypted_file_path);
                trace!(
                    "Keyword: {}, Plaintext length: {}, Ciphertext length: {}",
                    modified_keyword,
                    combined_key.len(),
                    encrypted_file_path.len()
                );

                self.keyword_value_map
                    .insert(combined_key, encrypted_file_path);
            }

            self.keyword_count_map
                .insert(keyword.clone(), file_paths.len());
        }

        info!("Encryption successful!");
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

    fn get_modified_keyword(
        &self,
        original_keyword: &str,
        keyword_count: &mut HashMap<String, usize>,
    ) -> String {
        let count = keyword_count
            .entry(original_keyword.to_string())
            .or_insert(0);
        *count += 1;

        if *count <= self.p {
            return original_keyword.to_string();
        }

        let ones_to_append = (*count - 1) / self.p;

        let mut modified_keyword = original_keyword.to_string();
        modified_keyword.push_str(&"1".repeat(ones_to_append));

        let new_count = *count - (ones_to_append * self.p);
        keyword_count.insert(modified_keyword.clone(), new_count);

        modified_keyword
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

    pub fn apply_tethys_dip(
        &mut self,
        m: usize,
        p: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let modified_keyword_count_map = self.create_modified_keyword_count_map();
        let result = if self.use_bfs {
            let dip = BfsDIP::<2>::new(m, p);
            dip.build_valmap(&modified_keyword_count_map, &self.keyword_value_map)
        } else {
            let dip = HashDIP::<2>::new(m, p);
            dip.build_valmap(&modified_keyword_count_map, &self.keyword_value_map)
        };

        self.tethys_dip_result = Some(result);
        Ok(())
    }

    pub fn apply_tethys_dip_on_volmap(
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
            let dip = BfsDIP::<2>::new(m, p);
            dip.build_volmap(&self.keyword_count_map, &keyword_count_as_value_map)
        } else {
            let dip = HashDIP::<2>::new(m, p);
            dip.build_volmap(&self.keyword_count_map, &keyword_count_as_value_map)
        };

        self.tethys_dip_result_volmap = Some(result);
        Ok(())
    }

    pub fn get_tethys_dip_stats(&self) -> Option<(usize, usize, usize)> {
        self.tethys_dip_result.as_ref().map(|(buckets, stash)| {
            let total_buckets = buckets.len();
            let non_empty_buckets = buckets.values().filter(|b| !b.is_empty()).count();
            let stash_size = stash.len();
            (total_buckets, non_empty_buckets, stash_size)
        })
    }

    pub fn get_tethys_dip_stats_volmap(&self) -> Option<(usize, usize, usize)> {
        self.tethys_dip_result_volmap
            .as_ref()
            .map(|(buckets, stash)| {
                let total_buckets = buckets.len();
                let non_empty_buckets = buckets.values().filter(|b| !b.is_empty()).count();
                let stash_size = stash.len();
                (total_buckets, non_empty_buckets, stash_size)
            })
    }

    pub fn add_padding_to_buckets(&mut self) {
        if let Some((buckets, _)) = self.tethys_dip_result.as_mut() {
            let target_size = 1024;
            for bucket in buckets.values_mut() {
                let current_size: usize = bucket.iter().map(|item| item.len()).sum();
                let padding_needed = target_size - current_size;
                let mut padding = vec![0u8; padding_needed];
                rand::thread_rng().fill_bytes(&mut padding);
                bucket.push(padding);
            }
        }
    }

    pub fn generate_single_string_array(&mut self) {
        self.single_string_array = self.tethys_dip_result.as_ref().map(|(buckets, _)| {
            let max_bucket_id = buckets.keys().max().unwrap_or(&0);
            let mut single_array = vec![String::new(); max_bucket_id + 1];

            let target_size = 1024;
            let b64_len = (target_size * 4 / 3) + 4;

            for (&bucket_id, bucket_content) in buckets {
                let combined_bucket_data: Vec<u8> =
                    bucket_content.iter().flatten().cloned().collect();

                let mut encoded = String::with_capacity(b64_len);
                general_purpose::STANDARD.encode_string(&combined_bucket_data, &mut encoded);

                single_array[bucket_id] = encoded;
            }

            single_array
        });
    }

    pub fn generate_single_string_array_from_volmap(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (buckets, _) = self
            .tethys_dip_result_volmap
            .as_ref()
            .ok_or("TethysDIP has not been applied")?;

        let elem_size = 256;

        let encode_vec = |v: &Vec<Vec<(String, usize)>>| {
            let mut buffer = vec![0u8; elem_size];
            let mut pos = 0;

            if let Some(pairs) = v.first() {
                if let Some((keyword, count)) = pairs.first() {
                    buffer[pos] = keyword.len() as u8;
                    pos += 1;

                    let keyword_bytes = keyword.as_bytes();
                    buffer[pos..pos + keyword_bytes.len()].copy_from_slice(keyword_bytes);
                    pos += keyword_bytes.len();

                    buffer[pos..pos + 4].copy_from_slice(&(*count as u32).to_le_bytes());
                }
            }

            general_purpose::STANDARD.encode(&buffer)
        };

        let max_bucket_index = buckets.keys().max().unwrap_or(&0);
        let encoded_buckets: Vec<String> = (0..=*max_bucket_index)
            .map(|i| {
                buckets.get(&i).map_or(
                    general_purpose::STANDARD.encode(&vec![0u8; elem_size]),
                    |b| encode_vec(b),
                )
            })
            .collect();

        self.single_string_array_volmap = Some(encoded_buckets);

        Ok(())
    }

    pub fn get_valmap_stash(&self) -> Option<Vec<Vec<u8>>> {
        self.tethys_dip_result
            .as_ref()
            .map(|(_, stash)| stash.clone())
    }

    pub fn get_single_string_array_volmap(&self) -> Option<&Vec<String>> {
        self.single_string_array_volmap.as_ref()
    }

    pub fn print_tethys_dip_distribution(&self) {
        if let Some((buckets, stash)) = &self.tethys_dip_result {
            trace!("\nTethysDIP Distribution:");
            for (bucket_id, values) in buckets {
                trace!("Bucket {}: {} items", bucket_id, values.len());
            }
            trace!("Stash: {} items", stash.len());
        } else {
            trace!("TethysDIP has not been applied yet.");
        }
    }

    pub fn print_volmap_tethys_dip_distribution(&self) {
        if let Some((buckets, stash)) = &self.tethys_dip_result_volmap {
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
                    .ok_or(format!(
                        "Keywords list is empty in {}",
                        keywords_path.display()
                    ))?
                    .to_string();

                buffer[0] = random_keyword.len() as u8;
                buffer[1..1 + random_keyword.len()].copy_from_slice(random_keyword.as_bytes());

                let random_count = rng.gen_range(1..10);
                buffer[1 + random_keyword.len()..1 + random_keyword.len() + 4]
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
