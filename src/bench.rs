use crate::dip::HashDIP;
use crate::generate_mmap::EncryptedFileSystem;

use frodo_pir::api::*;
use log::{error, info};

use rand::seq::SliceRandom;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

pub struct Benchmarker {
    encrypted_fs: EncryptedFileSystem,
    keywords_file: Option<String>,
    data_dir: String,
    epsilon: f64,
    run_decryption_test: bool,
    volmap_keyword_counts: Vec<u32>,
    selected_keywords: Vec<String>,
    valmap_m: usize,
    volmap_m: usize,
    setup_time: Duration,
    volmap_setup_time: Duration,
    volmap_query_time: Duration,
    valmap_setup_time: Duration,
    valmap_query_time: Duration,
    run_plaintext_retrieval: bool,
}

impl Benchmarker {
    pub fn new(
        p: usize,
        epsilon: f64,
        run_decryption_test: bool,
        keywords_file: Option<String>,
        data_dir: String,
        use_bfs: bool,
        run_plaintext_retrieval: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encrypted_fs = EncryptedFileSystem::new(p, data_dir.clone(), use_bfs)?;
        Ok(Benchmarker {
            encrypted_fs,
            keywords_file,
            data_dir,
            epsilon,
            run_decryption_test,
            volmap_keyword_counts: Vec::new(),
            selected_keywords: Vec::new(),
            valmap_m: 0,
            volmap_m: 0,
            setup_time: Duration::new(0, 0),
            volmap_setup_time: Duration::new(0, 0),
            volmap_query_time: Duration::new(0, 0),
            valmap_setup_time: Duration::new(0, 0),
            valmap_query_time: Duration::new(0, 0),
            run_plaintext_retrieval,
        })
    }

    pub fn run_benchmarks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let total_start = Instant::now();

        self.setup()?;
        self.select_random_keywords()?;
        self.run_volmap_benchmark()?;
        self.run_valmap_benchmark()?;

        let num_keywords = self.selected_keywords.len() as f64;

        let total_time = total_start.elapsed();
        let total_setup_time = self.setup_time + self.volmap_setup_time + self.valmap_setup_time;
        let total_query_time = self.volmap_query_time + self.valmap_query_time;
        let query_time_per_keyword = total_query_time.div_f64(num_keywords);

        info!("Timing Results:");
        info!("Setup Time: {:?}", self.setup_time);
        info!("VolMap Setup Time: {:?}", self.volmap_setup_time);
        info!("VolMap Query Time: {:?}", self.volmap_query_time);
        info!("ValMap Setup Time: {:?}", self.valmap_setup_time);
        info!("ValMap Query Time: {:?}", self.valmap_query_time);
        info!("Total Setup Time: {:?}", total_setup_time);
        info!(
            "Total Query Time: {:?} (avg per keyword: {:?})",
            total_query_time, query_time_per_keyword
        );
        info!("Total Execution Time: {:?}", total_time);

        // Check if we retrieve the right plaintext after decoding volmap
        if self.run_plaintext_retrieval {
            if let Some(test_kw) = self.selected_keywords.first().cloned() {
                self.test_specific_keyword_retrieval(&test_kw)?;
            }
        }
        Ok(())
    }

    fn setup(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let setup_start = Instant::now();

        self.encrypted_fs.encrypt_files()?;

        if self.run_decryption_test {
            info!("Running decryption test...");
            match self.encrypted_fs.verify_decryption() {
                Ok(true) => info!("Decryption verification passed successfully."),
                Ok(false) => error!("Decryption verification failed."),
                Err(e) => error!("Error during decryption verification: {}", e),
            }
        } else {
            info!("Skipping decryption test.");
        }

        let n = self.encrypted_fs.get_keyword_value_map_size();
        let p = self.encrypted_fs.get_p();
        self.valmap_m = ((1.8 + self.epsilon) * n as f64 / p as f64).ceil() as usize;

        let volmap_n = self.encrypted_fs.get_keyword_count_map_size();
        self.volmap_m = 10 * volmap_n;

        info!("TethysDIP Parameters:");
        info!("p (bucket capacity): {}", p);
        info!("n (keyword value map size): {}", n);
        info!("m (number of nodes): {}", self.valmap_m);
        info!("epsilon: {}", self.epsilon);

        self.encrypted_fs
            .apply_tethys_dip_on_volmap(self.volmap_m, 1)?;
        self.encrypted_fs.print_volmap_tethys_dip_distribution();
        self.encrypted_fs
            .generate_single_string_array_from_volmap()?;
        self.encrypted_fs.fill_empty_index_volmap()?;
        self.encrypted_fs.print_single_string_array_volmap();

        self.encrypted_fs.apply_tethys_dip(self.valmap_m, p)?;

        if let Some((total_buckets, non_empty_buckets, stash_size)) =
            self.encrypted_fs.get_tethys_dip_stats_volmap()
        {
            info!("\nTethysDIP Statistics for VolMap:");
            info!("Total buckets built: {}", total_buckets);
            info!("Non-empty buckets: {}", non_empty_buckets);
            info!("Stash size: {}", stash_size);
        }

        if let Some((total_buckets, non_empty_buckets, stash_size)) =
            self.encrypted_fs.get_tethys_dip_stats()
        {
            info!("\nTethysDIP Statistics for ValMap:");
            info!("Total buckets built: {}", total_buckets);
            info!("Non-empty buckets: {}", non_empty_buckets);
            info!("Stash size: {}", stash_size);
        }

        self.encrypted_fs.print_tethys_dip_distribution();
        self.encrypted_fs.add_padding_to_buckets();
        self.encrypted_fs.generate_single_string_array();
        self.encrypted_fs.fill_empty_index();
        self.encrypted_fs.print_single_string_array();

        self.setup_time = setup_start.elapsed();

        Ok(())
    }

    fn select_random_keywords(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // If we have a keywords file, use that instead of random selection
        if let Some(ref keywords_path) = self.keywords_file {
            let file = File::open(keywords_path)
                .map_err(|e| format!("Failed to open keywords file {}: {}", keywords_path, e))?;
            let reader = BufReader::new(file);
            self.selected_keywords = reader.lines().filter_map(Result::ok).collect();
            info!(
                "Using {} keywords from file {}",
                self.selected_keywords.len(),
                keywords_path
            );
            return Ok(());
        }

        // Otherwise randomly select
        let file = File::open(Path::new(&self.data_dir).join("keywords.txt"))
            .expect("Failed to open keywords.txt");
        let reader = BufReader::new(file);
        let keywords: Vec<String> = reader.lines().filter_map(Result::ok).collect();

        let mut rng = rand::thread_rng();

        self.selected_keywords = if keywords.len() <= 1 {
            keywords
        } else {
            keywords.choose_multiple(&mut rng, 1).cloned().collect()
        };

        info!(
            "Randomly selected {} keywords",
            self.selected_keywords.len()
        );
        Ok(())
    }

    fn get_indexes_for_keywords(&self, keywords: &[String], m: usize) -> HashSet<usize> {
        let mut indexes = HashSet::new();
        let dip_hasher = HashDIP::<4>::new(m, 1);

        for keyword in keywords {
            let hashes = dip_hasher.get_hashes(keyword);
            for h in hashes {
                indexes.insert(h);
            }
        }

        indexes
    }

    fn run_volmap_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Running FrodoPIR for VolMap.");
        let volmap_setup_start = Instant::now();

        let volmap_db_array = self.encrypted_fs.get_single_string_array_volmap().unwrap();
        let m = volmap_db_array.len() as usize;
        let elem_size = 2u32.pow(8) as usize;
        let plaintext_bits = 10usize;
        let lwe_dim = 1572;
        let shard =
            Shard::from_base64_strings(&volmap_db_array, lwe_dim, m, elem_size, plaintext_bits)
                .unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);

        self.volmap_setup_time = volmap_setup_start.elapsed();

        let volmap_query_start = Instant::now();

        let indexes_to_query =
            self.get_indexes_for_keywords(&self.selected_keywords, self.volmap_m);
        self.volmap_keyword_counts.clear();

        for &index in &indexes_to_query {
            let mut qp = QueryParams::new(&cp, bp).unwrap();
            let q = match qp.generate_query(index) {
                Ok(query) => query,
                Err(_) => continue,
            };
            let d_resp = shard.respond(&q).unwrap();
            let resp: Response = bincode::deserialize(&d_resp).unwrap();
            let output = resp.parse_output_as_base64(&qp);
            let decoded = self.encrypted_fs.decode_base64(&output)?;

            let keyword_size = decoded[0] as usize;
            if keyword_size > 0 && decoded.len() >= 1 + keyword_size + 4 {
                if let Ok(keyword) = String::from_utf8(decoded[1..1 + keyword_size].to_vec()) {
                    if let Ok(count_bytes) =
                        decoded[1 + keyword_size..1 + keyword_size + 4].try_into()
                    {
                        let count = u32::from_le_bytes(count_bytes);

                        if self.selected_keywords.contains(&keyword) {
                            self.volmap_keyword_counts.push(count);
                        }
                    }
                }
            }
        }

        self.volmap_query_time = volmap_query_start.elapsed();
        info!("FrodoPIR query finished executing.");

        Ok(())
    }

    fn generate_valmap_keywords(&self) -> Vec<String> {
        let mut valmap_keywords = Vec::new();
        for (keyword, &count) in self
            .selected_keywords
            .iter()
            .zip(self.volmap_keyword_counts.iter())
        {
            let num_entries = (count as f64 / self.encrypted_fs.get_p() as f64).ceil() as usize;
            for i in 0..num_entries {
                let mut modified_keyword = keyword.clone();
                modified_keyword.push_str(&"1".repeat(i));
                valmap_keywords.push(modified_keyword);
            }
        }
        valmap_keywords
    }

    fn run_valmap_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Running FrodoPIR for ValMap.");
        let valmap_start_time = Instant::now();

        let db_array = self.encrypted_fs.get_single_string_array().unwrap();
        let m = db_array.len() as usize;
        let elem_size = 2u32.pow(13) as usize;
        let plaintext_bits = 10usize;
        let lwe_dim = 1572;
        let shard =
            Shard::from_base64_strings(&db_array, lwe_dim, m, elem_size, plaintext_bits).unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);
        self.valmap_setup_time = valmap_start_time.elapsed();

        let valmap_query_time = Instant::now();
        let valmap_keywords = self.generate_valmap_keywords();
        let indexes_to_query = self.get_indexes_for_keywords(&valmap_keywords, self.valmap_m);

        for &index in &indexes_to_query {
            let mut qp = QueryParams::new(&cp, bp).unwrap();
            let q = match qp.generate_query(index) {
                Ok(query) => query,
                Err(_) => continue,
            };
            let d_resp = shard.respond(&q).unwrap();
            let resp: Response = bincode::deserialize(&d_resp).unwrap();
            let _ = resp.parse_output_as_base64(&qp);
        }

        self.valmap_query_time = valmap_query_time.elapsed();
        info!("FrodoPIR query for ValMap finished executing.");

        Ok(())
    }

    pub fn test_specific_keyword_retrieval(
        &mut self,
        target_keyword: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!(
            "--- Running End-to-End Retrieval Test for '{}' ---",
            target_keyword
        );

        let volmap_db_array = self.encrypted_fs.get_single_string_array_volmap().unwrap();
        let m_vol = volmap_db_array.len() as usize;
        let vol_shard = Shard::from_base64_strings(&volmap_db_array, 1572, m_vol, 256, 10).unwrap();
        let bp_vol = vol_shard.get_base_params();
        let cp_vol = CommonParams::from(bp_vol);

        let volmap_indexes =
            self.get_indexes_for_keywords(&[target_keyword.to_string()], self.volmap_m);
        let mut retrieved_count = 0;

        for &index in &volmap_indexes {
            let mut qp = QueryParams::new(&cp_vol, bp_vol).unwrap();
            if let Ok(query) = qp.generate_query(index) {
                let d_resp = vol_shard.respond(&query).unwrap();
                let resp: Response = bincode::deserialize(&d_resp).unwrap();
                let output = resp.parse_output_as_base64(&qp);
                let decoded = self.encrypted_fs.decode_base64(&output)?;

                let keyword_size = decoded[0] as usize;
                if keyword_size > 0 && decoded.len() >= 1 + keyword_size + 4 {
                    if let Ok(keyword) = String::from_utf8(decoded[1..1 + keyword_size].to_vec()) {
                        if keyword == target_keyword {
                            let count_bytes: [u8; 4] = decoded
                                [1 + keyword_size..1 + keyword_size + 4]
                                .try_into()
                                .unwrap();
                            retrieved_count = u32::from_le_bytes(count_bytes);
                            break;
                        }
                    }
                }
            }
        }

        info!("> VolMap returned count: {}", retrieved_count);
        if retrieved_count == 0 {
            info!("> No documents found. Ending test.");
            return Ok(());
        }

        let p = self.encrypted_fs.get_p();
        let num_entries = (retrieved_count as f64 / p as f64).ceil() as usize;
        let mut valmap_keywords = Vec::new();

        for i in 0..num_entries {
            let mut modified_keyword = target_keyword.to_string();
            modified_keyword.push_str(&"1".repeat(i));
            valmap_keywords.push(modified_keyword);
        }

        info!(
            "> Generating {} queries for ValMap...",
            valmap_keywords.len()
        );

        let valmap_db_array = self.encrypted_fs.get_single_string_array().unwrap();
        let m_val = valmap_db_array.len() as usize;
        let val_shard =
            Shard::from_base64_strings(&valmap_db_array, 1572, m_val, 8192, 10).unwrap();
        let bp_val = val_shard.get_base_params();
        let cp_val = CommonParams::from(bp_val);

        let valmap_indexes = self.get_indexes_for_keywords(&valmap_keywords, self.valmap_m);
        let mut retrieved_encrypted_paths = Vec::new();

        for &index in &valmap_indexes {
            let mut qp = QueryParams::new(&cp_val, bp_val).unwrap();
            if let Ok(query) = qp.generate_query(index) {
                let d_resp = val_shard.respond(&query).unwrap();
                let resp: Response = bincode::deserialize(&d_resp).unwrap();
                let output = resp.parse_output_as_base64(&qp);
                let decoded_bucket = self.encrypted_fs.decode_base64(&output)?;

                // Extract the 32-byte AES chunks generated by `encrypt_data`
                for chunk in decoded_bucket.chunks_exact(32) {
                    retrieved_encrypted_paths.push(chunk.to_vec());
                }
            }
        }

        info!(
            "> Retrieved {} potential 48-byte encrypted payloads from ValMap.",
            retrieved_encrypted_paths.len()
        );

        let stash = self.encrypted_fs.get_valmap_stash().unwrap_or_default();
        info!(
            "> Retrieved {} payloads from the ValMap Stash.",
            stash.len()
        );

        if let Some(expected_paths) = self
            .encrypted_fs
            .plaintext_encrypted_multimap
            .get_vec(target_keyword)
        {
            let mut matched_in_db = 0;
            let mut matched_in_stash = 0;

            for expected_path in expected_paths {
                if retrieved_encrypted_paths.contains(expected_path) {
                    matched_in_db += 1;
                } else if stash.contains(expected_path) {
                    matched_in_stash += 1;
                }
            }

            let total_matched = matched_in_db + matched_in_stash;

            if total_matched == expected_paths.len() {
                info!(
                    "> Success! Verified {}/{} document payloads linked to '{}'.",
                    total_matched,
                    expected_paths.len(),
                    target_keyword
                );
                if matched_in_stash > 0 {
                    info!(">    - {} found via PIR (Buckets)", matched_in_db);
                    info!(
                        ">    - {} found via Direct Download (Stash)",
                        matched_in_stash
                    );
                }
            } else {
                error!("> Failed: Only found {}/{} document payloads for '{}'. ({} in DB, {} in Stash)",
                            total_matched, expected_paths.len(), target_keyword, matched_in_db, matched_in_stash);
            }
        }

        Ok(())
    }
}
