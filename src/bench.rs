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

use base64::Engine;
use rayon::prelude::*;

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
    run_docmap_test: bool,
    docmap_m: usize,
    docmap_setup_time: Duration,
    docmap_query_time: Duration,
    run_ddr_test: bool,
    run_insert_test: bool,
    insert_split: f64,
    ddr_setup_time: Duration,
    ddr_query_time: Duration,
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
        run_docmap_test: bool,
        run_ddr_test: bool,
        run_insert_test: bool,
        insert_split: f64,
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
            run_docmap_test,
            docmap_m: 0,
            docmap_setup_time: Duration::new(0, 0),
            docmap_query_time: Duration::new(0, 0),
            run_ddr_test,
            ddr_setup_time: Duration::new(0, 0),
            ddr_query_time: Duration::new(0, 0),
            run_insert_test,
            insert_split,
        })
    }

    pub fn run_benchmarks(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let total_start = Instant::now();

        if self.run_insert_test {
            log::info!(
                "--- Running Incremental Insert Benchmark (Split: {}) ---",
                self.insert_split
            );
            self.encrypted_fs.encrypt_files()?;

            let mut insert_data = self.encrypted_fs.extract_insert_dataset(self.insert_split);
            insert_data.truncate(1000);
            log::info!(
                "Initial Build Dataset: {}, Insert Dataset: {}",
                self.encrypted_fs.get_keyword_value_map_size(),
                insert_data.len()
            );

            let n = self.encrypted_fs.get_keyword_value_map_size() + insert_data.len();
            let p = self.encrypted_fs.get_p();
            self.valmap_m = ((1.8 + self.epsilon) * n as f64 / p as f64).ceil() as usize;

            let num_inserts = insert_data.len();
            self.encrypted_fs.build_masked_valmap(self.valmap_m, p)?;

            let insert_time =
                self.encrypted_fs
                    .run_incremental_inserts(self.valmap_m, p, insert_data);
            let avg_insert_time = insert_time.div_f64(num_inserts as f64);

            log::info!(
                "Total Time for {} Oblivious Inserts: {:?}",
                num_inserts,
                insert_time
            );
            log::info!("Average Time per Insert: {:?}", avg_insert_time);

            return Ok(());
        } else {
            self.setup()?;
        }

        self.select_random_keywords()?;
        self.run_volmap_benchmark()?;
        self.run_valmap_benchmark()?;

        if self.run_docmap_test {
            self.run_docmap_benchmark()?;
        }

        if self.run_ddr_test {
            self.run_ddr_benchmark()?;
        }

        let num_keywords = self.selected_keywords.len() as f64;

        let total_time = total_start.elapsed();

        let total_setup_time = self.setup_time
            + self.volmap_setup_time
            + self.valmap_setup_time
            + self.docmap_setup_time
            + self.ddr_setup_time;

        let total_query_time = self.volmap_query_time
            + self.valmap_query_time
            + self.docmap_query_time
            + self.ddr_query_time;

        let query_time_per_keyword = total_query_time.div_f64(num_keywords);

        info!("Timing Results:");
        info!("Setup Time: {:?}", self.setup_time);
        info!("VolMap Setup Time: {:?}", self.volmap_setup_time);
        info!("VolMap Query Time: {:?}", self.volmap_query_time);
        info!("ValMap Setup Time: {:?}", self.valmap_setup_time);
        info!("ValMap Query Time: {:?}", self.valmap_query_time);
        info!("DocMap Setup Time: {:?}", self.docmap_setup_time);
        info!("DocMap Query Time: {:?}", self.docmap_query_time);
        info!("DDR Setup Time: {:?}", self.ddr_setup_time);
        info!("DDR Query Time: {:?}", self.ddr_query_time);
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
        self.volmap_m = 5 * volmap_n;

        info!("DIP Parameters:");
        info!("p (bucket capacity): {}", p);
        info!("n (keyword value map size): {}", n);
        info!("m (number of nodes): {}", self.valmap_m);
        info!("epsilon: {}", self.epsilon);

        self.encrypted_fs
            .apply_dip_on_volmap(self.volmap_m, 1)?;
        self.encrypted_fs.print_volmap_dip_distribution();
        self.encrypted_fs
            .generate_single_string_array_from_volmap(self.volmap_m)?;
        self.encrypted_fs.fill_empty_index_volmap()?;
        self.encrypted_fs.print_single_string_array_volmap();

        self.encrypted_fs.apply_dip(self.valmap_m, p)?;

        if let Some((total_buckets, non_empty_buckets, stash_size)) =
            self.encrypted_fs.get_dip_stats_volmap()
        {
            info!("\nDIP Statistics for VolMap:");
            info!("Total buckets built: {}", total_buckets);
            info!("Non-empty buckets: {}", non_empty_buckets);
            info!("Stash size: {}", stash_size);
        }

        if let Some((total_buckets, non_empty_buckets, stash_size)) =
            self.encrypted_fs.get_dip_stats()
        {
            info!("\nDIP Statistics for ValMap:");
            info!("Total buckets built: {}", total_buckets);
            info!("Non-empty buckets: {}", non_empty_buckets);
            info!("Stash size: {}", stash_size);
        }

        self.encrypted_fs.print_dip_distribution();
        self.encrypted_fs.add_padding_to_buckets();
        self.encrypted_fs
            .generate_single_string_array(self.valmap_m);
        self.encrypted_fs.fill_empty_index();
        self.encrypted_fs.print_single_string_array();

        if self.run_docmap_test {
            info!("Setting up DocMap...");
            let docmap_setup_start = Instant::now();
            self.encrypted_fs.build_and_apply_docmap()?;

            if let Some(arr) = &self.encrypted_fs.single_string_array_docmap {
                self.docmap_m = arr.len();
                info!("\nFlat Array Statistics for DocMap:");
                info!("Total chunks built: {}", self.docmap_m);
            }
            self.docmap_setup_time = docmap_setup_start.elapsed();
        }

        if self.run_ddr_test {
            info!("Setting up DDR-SSE Document Store...");
            let ddr_setup_start = Instant::now();
            self.encrypted_fs.build_ddr_store()?;
            info!("\nDDR Store Statistics:");
            info!(
                "Total KV pairs stored: {}",
                self.encrypted_fs.ddr_server_store.len()
            );
            self.ddr_setup_time = ddr_setup_start.elapsed();
        }

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
        let elem_size = 2u32.pow(11) as usize;
        let plaintext_bits = 10usize;
        let lwe_dim = 1288;
        let shard =
            Shard::from_base64_strings(&volmap_db_array, lwe_dim, m, elem_size, plaintext_bits)
                .unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);

        self.volmap_setup_time = volmap_setup_start.elapsed();

        let volmap_query_start = Instant::now();
        let indexes_to_query =
            self.get_indexes_for_keywords(&self.selected_keywords, self.volmap_m);

        let mut indices: Vec<usize> = indexes_to_query.into_iter().collect();
        while indices.len() % 8 != 0 {
            indices.push(0);
        }

        self.volmap_keyword_counts.clear();

        let counts: Vec<u32> = indices
            .par_chunks(8)
            .flat_map(|chunk| {
                let mut qps = Vec::with_capacity(8);
                let mut queries = Vec::with_capacity(8);

                let mut batched_qps = BatchedQueryParams8::new(&cp, bp).unwrap();

                for i in 0..8 {
                    let mut success = false;

                    let mut qp = batched_qps.extract_query_params(i).unwrap();
                    if let Ok(q) = qp.generate_query(chunk[i]) {
                        qps.push(qp);
                        queries.push(q);
                        success = true;
                    }

                    if !success {
                        loop {
                            let mut fallback_qp = QueryParams::new(&cp, bp).unwrap();
                            if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                                qps.push(fallback_qp);
                                queries.push(q);
                                break;
                            }
                        }
                    }
                }
                let interleaved: Vec<[u32; 8]> = (0..m)
                    .map(|row| {
                        let mut vals = [0u32; 8];
                        for j in 0..8 {
                            vals[j] = queries[j].as_slice()[row];
                        }
                        vals
                    })
                    .collect();

                let bq = BatchedQuery8 { interleaved };
                let d_resp = shard.respond_batched_8(&bq).unwrap();
                let resp: BatchedResponse8 = bincode::deserialize(&d_resp).unwrap();

                let width = shard.get_db().get_matrix_width_self();
                let mut chunk_counts = Vec::new();

                for j in 0..8 {
                    let mut single_resp = Vec::with_capacity(width);
                    for w in 0..width {
                        single_resp.push(resp.results[w][j]);
                    }

                    let parsed_resp = Response(single_resp);

                    let output = parsed_resp.parse_output_as_base64(&qps[j]);
                    if let Ok(decoded) = self.encrypted_fs.decode_base64(&output) {
                        let keyword_size = decoded[0] as usize;
                        if keyword_size > 0 && decoded.len() >= 1 + keyword_size + 4 {
                            let retrieved_bytes = &decoded[1..1 + keyword_size];

                            if let Ok(count_bytes) =
                                decoded[1 + keyword_size..1 + keyword_size + 4].try_into()
                            {
                                let count = u32::from_le_bytes(count_bytes);

                                if self.selected_keywords.iter().any(|k| {
                                    let k_bytes = k.as_bytes();
                                    let trunc_len = k_bytes.len().min(251);
                                    &k_bytes[..trunc_len] == retrieved_bytes
                                }) {
                                    chunk_counts.push(count);
                                }
                            }
                        }
                    }
                }
                chunk_counts
            })
            .collect();

        self.volmap_keyword_counts.extend(counts);
        self.volmap_query_time = volmap_query_start.elapsed();
        info!("FrodoPIR batched query finished executing.");

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
        let elem_size = 2u32.pow(16) as usize;
        let plaintext_bits = 9usize;
        let lwe_dim = 1288;
        let shard =
            Shard::from_base64_strings(&db_array, lwe_dim, m, elem_size, plaintext_bits).unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);
        self.valmap_setup_time = valmap_start_time.elapsed();

        let valmap_query_time = Instant::now();
        let valmap_keywords = self.generate_valmap_keywords();
        let indexes_to_query = self.get_indexes_for_keywords(&valmap_keywords, self.valmap_m);

        let mut indices: Vec<usize> = indexes_to_query.into_iter().collect();
        while indices.len() % 8 != 0 {
            indices.push(0);
        }

        indices.par_chunks(8).for_each(|chunk| {
            let mut qps = Vec::with_capacity(8);
            let mut queries = Vec::with_capacity(8);

            let mut batched_qps = BatchedQueryParams8::new(&cp, bp).unwrap();

            for i in 0..8 {
                let mut success = false;
                let mut qp = batched_qps.extract_query_params(i).unwrap();
                if let Ok(q) = qp.generate_query(chunk[i]) {
                    qps.push(qp);
                    queries.push(q);
                    success = true;
                }

                if !success {
                    loop {
                        let mut fallback_qp = QueryParams::new(&cp, bp).unwrap();
                        if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                            qps.push(fallback_qp);
                            queries.push(q);
                            break;
                        }
                    }
                }
            }
            let interleaved: Vec<[u32; 8]> = (0..m)
                .map(|row| {
                    let mut vals = [0u32; 8];
                    for j in 0..8 {
                        vals[j] = queries[j].as_slice()[row];
                    }
                    vals
                })
                .collect();

            let bq = BatchedQuery8 { interleaved };
            let d_resp = shard.respond_batched_8(&bq).unwrap();
            let resp: BatchedResponse8 = bincode::deserialize(&d_resp).unwrap();

            let width = shard.get_db().get_matrix_width_self();

            for j in 0..8 {
                let mut single_resp = Vec::with_capacity(width);
                for w in 0..width {
                    single_resp.push(resp.results[w][j]);
                }

                let parsed_resp = Response(single_resp);

                let _ = parsed_resp.parse_output_as_base64(&qps[j]);
            }
        });

        self.valmap_query_time = valmap_query_time.elapsed();
        info!("FrodoPIR batched query for ValMap finished executing.");

        Ok(())
    }

    fn run_ddr_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Running DDR-SSE Document Retrieval Benchmark.");
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let query_start = Instant::now();
        let mut ddr_addresses = Vec::new();

        for kw in &self.selected_keywords {
            if let Some(paths) = self.encrypted_fs.get_original_paths(kw) {
                let mut seed_val = 0u64;
                for (i, &b) in self.encrypted_fs.ddr_query_seed[..8].iter().enumerate() {
                    seed_val |= (b as u64) << (8 * i);
                }
                for &b in kw.as_bytes() {
                    seed_val = seed_val.wrapping_add(b as u64);
                }

                let mut rng = StdRng::seed_from_u64(seed_val);
                let midpoint = rng.gen_range(0..=paths.len());
                let side = rng.gen_range(0..2);

                for (ii, &doc_id) in paths.iter().enumerate() {
                    let chunks = *self
                        .encrypted_fs
                        .doc_chunk_counts
                        .get(&doc_id)
                        .unwrap_or(&1);

                    let key_to_use =
                        if (ii < midpoint && side == 0) || (ii >= midpoint && side == 1) {
                            &self.encrypted_fs.ddr_doc_keys.0
                        } else {
                            &self.encrypted_fs.ddr_doc_keys.1
                        };

                    for chunk_idx in 0..chunks {
                        let data_to_hash = format!("doc_{}_{}", doc_id, chunk_idx);
                        let mut hasher = blake3::Hasher::new_keyed(key_to_use);
                        hasher.update(data_to_hash.as_bytes());
                        let addr = base64::engine::general_purpose::STANDARD
                            .encode(hasher.finalize().as_bytes());
                        ddr_addresses.push(addr);
                    }
                }
            }
        }

        info!(
            "Querying {} 4KB chunks from DDR Server Store.",
            ddr_addresses.len()
        );

        let mut retrieved_chunks = 0;
        for addr in ddr_addresses {
            if self.encrypted_fs.ddr_server_store.contains_key(&addr) {
                retrieved_chunks += 1;
            }
        }

        self.ddr_query_time = query_start.elapsed();
        info!(
            "DDR batched query finished executing. Retrieved {} chunks.",
            retrieved_chunks
        );
        Ok(())
    }

    fn run_docmap_benchmark(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        info!("Running FrodoPIR for DocMap.");
        let query_start = Instant::now();

        let db_array = self
            .encrypted_fs
            .single_string_array_docmap
            .as_ref()
            .unwrap();
        let m = db_array.len();
        let elem_size = 2u32.pow(13) as usize;
        let plaintext_bits = 9usize;
        let lwe_dim = 1288;
        let shard =
            Shard::from_base64_strings(db_array, lwe_dim, m, elem_size, plaintext_bits).unwrap();

        let bp = shard.get_base_params();
        let cp = CommonParams::from(bp);

        let mut target_docs = HashSet::new();
        for kw in &self.selected_keywords {
            if let Some(paths) = self.encrypted_fs.get_original_paths(kw) {
                for p in paths {
                    target_docs.insert(p);
                }
            }
        }

        let mut docmap_keywords = Vec::new();
        for &doc_id in &target_docs {
            let chunks = *self
                .encrypted_fs
                .doc_chunk_counts
                .get(&doc_id)
                .unwrap_or(&1);
            for i in 0..chunks {
                let mut modified_keyword = format!("doc_{}_", doc_id);
                modified_keyword.push_str(&"1".repeat(i));
                docmap_keywords.push(modified_keyword);
            }
        }

        info!("Querying {} 4KB chunks from DocMap.", docmap_keywords.len());

        let mut indices: Vec<usize> = docmap_keywords
            .iter()
            .filter_map(|kw| self.encrypted_fs.docmap_indices.get(kw).copied())
            .collect();

        while indices.len() % 8 != 0 {
            indices.push(0);
        }

        indices.par_chunks(8).for_each(|chunk| {
            let mut qps = Vec::with_capacity(8);
            let mut queries = Vec::with_capacity(8);
            let mut batched_qps = BatchedQueryParams8::new(&cp, bp).unwrap();

            for i in 0..8 {
                let mut success = false;
                let mut qp = batched_qps.extract_query_params(i).unwrap();
                if let Ok(q) = qp.generate_query(chunk[i]) {
                    qps.push(qp);
                    queries.push(q);
                    success = true;
                }

                if !success {
                    loop {
                        let mut fallback_qp = QueryParams::new(&cp, bp).unwrap();
                        if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                            qps.push(fallback_qp);
                            queries.push(q);
                            break;
                        }
                    }
                }
            }

            let interleaved: Vec<[u32; 8]> = (0..m)
                .map(|row| {
                    let mut vals = [0u32; 8];
                    for j in 0..8 {
                        vals[j] = queries[j].as_slice()[row];
                    }
                    vals
                })
                .collect();

            let bq = BatchedQuery8 { interleaved };
            let d_resp = shard.respond_batched_8(&bq).unwrap();
            let resp: BatchedResponse8 = bincode::deserialize(&d_resp).unwrap();
            let width = shard.get_db().get_matrix_width_self();

            for j in 0..8 {
                let mut single_resp = Vec::with_capacity(width);
                for w in 0..width {
                    single_resp.push(resp.results[w][j]);
                }
                let parsed_resp = Response(single_resp);
                let _ = parsed_resp.parse_output_as_base64(&qps[j]);
            }
        });

        self.docmap_query_time = query_start.elapsed();
        info!("FrodoPIR batched query for DocMap finished executing.");
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
        let vol_shard =
            Shard::from_base64_strings(&volmap_db_array, 1288, m_vol, 2048, 10).unwrap();
        let bp_vol = vol_shard.get_base_params();
        let cp_vol = CommonParams::from(bp_vol);

        let volmap_indexes_set =
            self.get_indexes_for_keywords(&[target_keyword.to_string()], self.volmap_m);
        let mut vol_indices: Vec<usize> = volmap_indexes_set.into_iter().collect();
        let original_volmap_len = vol_indices.len();

        while vol_indices.len() % 64 != 0 {
            vol_indices.push(0);
        }

        let fs = &self.encrypted_fs;

        let mut retrieved_counts: Vec<u32> = vol_indices
            .par_chunks(64)
            .flat_map(|chunk| {
                let mut local_counts = Vec::new();
                let mut qps = Vec::with_capacity(64);
                let mut queries = Vec::with_capacity(64);

                let mut batched_qps = BatchedQueryParams64::new(&cp_vol, bp_vol).unwrap();

                for i in 0..64 {
                    let mut success = false;
                    let mut qp = batched_qps.extract_query_params(i).unwrap();

                    if let Ok(q) = qp.generate_query(chunk[i]) {
                        qps.push(qp);
                        queries.push(q);
                        success = true;
                    }

                    if !success {
                        loop {
                            let mut fallback_qp = QueryParams::new(&cp_vol, bp_vol).unwrap();
                            if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                                qps.push(fallback_qp);
                                queries.push(q);
                                break;
                            }
                        }
                    }
                }

                let interleaved: Vec<[u32; 64]> = (0..m_vol)
                    .map(|row| {
                        let mut vals = [0u32; 64];
                        for j in 0..64 {
                            vals[j] = queries[j].as_slice()[row];
                        }
                        vals
                    })
                    .collect();

                let bq = BatchedQuery64 { interleaved };
                let d_resp = vol_shard.respond_batched_64(&bq).unwrap();
                let resp: BatchedResponse64 = bincode::deserialize(&d_resp).unwrap();
                let width = vol_shard.get_db().get_matrix_width_self();

                for j in 0..64 {
                    let mut single_resp = Vec::with_capacity(width);
                    for w in 0..width {
                        single_resp.push(resp.results[w][j]);
                    }
                    let parsed_resp = Response(single_resp);
                    let output = parsed_resp.parse_output_as_base64(&qps[j]);

                    let mut local_count = 0;
                    if let Ok(decoded) = fs.decode_base64(&output) {
                        let keyword_size = decoded[0] as usize;
                        if keyword_size > 0 && decoded.len() >= 1 + keyword_size + 4 {
                            let retrieved_bytes = &decoded[1..1 + keyword_size];
                            let target_bytes = target_keyword.as_bytes();
                            let trunc_len = target_bytes.len().min(251);

                            if retrieved_bytes == &target_bytes[..trunc_len] {
                                let count_bytes: [u8; 4] = decoded
                                    [1 + keyword_size..1 + keyword_size + 4]
                                    .try_into()
                                    .unwrap();
                                local_count = u32::from_le_bytes(count_bytes);
                            }
                        }
                    }
                    local_counts.push(local_count);
                }
                local_counts
            })
            .collect();

        retrieved_counts.truncate(original_volmap_len);
        let retrieved_count = retrieved_counts.into_iter().max().unwrap_or(0);

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
            Shard::from_base64_strings(&valmap_db_array, 1288, m_val, 65536, 9).unwrap();
        let bp_val = val_shard.get_base_params();
        let cp_val = CommonParams::from(bp_val);

        let valmap_indexes_set = self.get_indexes_for_keywords(&valmap_keywords, self.valmap_m);
        let mut valmap_indexes: Vec<usize> = valmap_indexes_set.into_iter().collect();
        let original_valmap_len = valmap_indexes.len();

        while valmap_indexes.len() % 64 != 0 {
            valmap_indexes.push(0);
        }

        let mut retrieved_buckets: Vec<Vec<u8>> = valmap_indexes
            .par_chunks(64)
            .flat_map(|chunk| {
                let mut local_buckets = Vec::new();
                let mut qps = Vec::with_capacity(64);
                let mut queries = Vec::with_capacity(64);

                let mut batched_qps = BatchedQueryParams64::new(&cp_val, bp_val).unwrap();

                for i in 0..64 {
                    let mut success = false;
                    let mut qp = batched_qps.extract_query_params(i).unwrap();

                    if let Ok(q) = qp.generate_query(chunk[i]) {
                        qps.push(qp);
                        queries.push(q);
                        success = true;
                    }

                    if !success {
                        loop {
                            let mut fallback_qp = QueryParams::new(&cp_val, bp_val).unwrap();
                            if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                                qps.push(fallback_qp);
                                queries.push(q);
                                break;
                            }
                        }
                    }
                }

                let interleaved: Vec<[u32; 64]> = (0..m_val)
                    .map(|row| {
                        let mut vals = [0u32; 64];
                        for j in 0..64 {
                            vals[j] = queries[j].as_slice()[row];
                        }
                        vals
                    })
                    .collect();

                let bq = BatchedQuery64 { interleaved };
                let d_resp = val_shard.respond_batched_64(&bq).unwrap();
                let resp: BatchedResponse64 = bincode::deserialize(&d_resp).unwrap();
                let width = val_shard.get_db().get_matrix_width_self();

                for j in 0..64 {
                    let mut single_resp = Vec::with_capacity(width);
                    for w in 0..width {
                        single_resp.push(resp.results[w][j]);
                    }
                    let parsed_resp = Response(single_resp);
                    let output = parsed_resp.parse_output_as_base64(&qps[j]);

                    if let Ok(decoded_bucket) = fs.decode_base64(&output) {
                        local_buckets.push(decoded_bucket);
                    } else {
                        local_buckets.push(Vec::new());
                    }
                }
                local_buckets
            })
            .collect();

        retrieved_buckets.truncate(original_valmap_len);
        let retrieved_encrypted_paths: Vec<Vec<u8>> = retrieved_buckets
            .into_iter()
            .flat_map(|bucket| {
                bucket
                    .chunks_exact(32)
                    .map(|c| c.to_vec())
                    .collect::<Vec<_>>()
            })
            .collect();

        let stash = self.encrypted_fs.get_valmap_stash().unwrap_or_default();
        let all_payloads: Vec<_> = retrieved_encrypted_paths
            .iter()
            .chain(stash.iter())
            .collect();

        let client_key = self
            .encrypted_fs
            .client_keys
            .get(target_keyword)
            .expect("Client key not found");

        let mut valid_payloads = Vec::new();
        let mut valid_doc_ids = std::collections::HashSet::new();
        let mut failed_decrypts = 0;

        if let Some(expected_paths) = self
            .encrypted_fs
            .plaintext_encrypted_multimap
            .get_vec(target_keyword)
        {
            for payload in all_payloads {
                if payload.len() != 32 {
                    failed_decrypts += 1;
                    continue;
                }
                let decrypted = self.encrypted_fs.decrypt_data_public(client_key, payload);
                let file_id = u32::from_le_bytes(decrypted[..4].try_into().unwrap());

                if expected_paths.contains(payload) {
                    valid_payloads.push(payload.clone());
                    valid_doc_ids.insert(file_id);
                } else {
                    failed_decrypts += 1;
                }
            }

            info!(
                "> Attempting to decrypt {} payloads using client key...",
                retrieved_encrypted_paths.len() + stash.len()
            );
            info!(
                "> Decryption failed on {} payloads (belonged to other buckets/padding).",
                failed_decrypts
            );
            info!(
                "> Decryption succeeded on {} payloads! Retrieved Document IDs: {:?}",
                valid_payloads.len(),
                valid_doc_ids
            );

            let total_matched = valid_payloads.len();
            if total_matched == expected_paths.len() {
                info!(
                    "> Success! Verified {}/{} document payloads linked to '{}'.",
                    total_matched,
                    expected_paths.len(),
                    target_keyword
                );

                let mut ordered_docs: Vec<u32> = valid_doc_ids.iter().copied().collect();
                ordered_docs.sort_unstable();
                let mut expected_file_chunks = Vec::new();

                for &doc_id in &ordered_docs {
                    let chunks = *self
                        .encrypted_fs
                        .doc_chunk_counts
                        .get(&doc_id)
                        .unwrap_or(&1);
                    let enc_path = format!("encrypted_docs/{}.enc", doc_id);
                    let enc_content = std::fs::read(&enc_path).unwrap_or_default();

                    for i in 0..chunks {
                        let start = i * 1024;
                        let end = ((i + 1) * 1024).min(enc_content.len());
                        let mut padded_chunk = vec![0u8; 1024];
                        if start < enc_content.len() {
                            padded_chunk[..end - start].copy_from_slice(&enc_content[start..end]);
                        }
                        expected_file_chunks.push(padded_chunk);
                    }
                }

                if self.run_docmap_test {
                    info!("> Proceeding to DocMap retrieval test for the discovered documents...");

                    let mut docmap_keywords = Vec::new();
                    for &doc_id in &ordered_docs {
                        let chunks = *self
                            .encrypted_fs
                            .doc_chunk_counts
                            .get(&doc_id)
                            .unwrap_or(&1);
                        for i in 0..chunks {
                            let mut modified_keyword = format!("doc_{}_", doc_id);
                            modified_keyword.push_str(&"1".repeat(i));
                            docmap_keywords.push(modified_keyword);
                        }
                    }

                    info!(
                        "> Generating {} queries for DocMap to fetch {} documents...",
                        docmap_keywords.len(),
                        valid_doc_ids.len()
                    );

                    let docmap_db_array = self
                        .encrypted_fs
                        .single_string_array_docmap
                        .as_ref()
                        .expect("DocMap not initialized");
                    let m_doc = docmap_db_array.len() as usize;
                    let doc_shard =
                        Shard::from_base64_strings(&docmap_db_array, 1288, m_doc, 8192, 9).unwrap();
                    let bp_doc = doc_shard.get_base_params();
                    let cp_doc = CommonParams::from(bp_doc);

                    let mut docmap_indexes: Vec<usize> = docmap_keywords
                        .iter()
                        .filter_map(|kw| self.encrypted_fs.docmap_indices.get(kw).copied())
                        .collect();

                    let original_docmap_len = docmap_indexes.len();

                    // Pad to 64
                    while docmap_indexes.len() % 64 != 0 {
                        docmap_indexes.push(0);
                    }

                    // Chunk by 64
                    let mut retrieved_doc_chunks: Vec<Vec<u8>> = docmap_indexes
                        .par_chunks(64)
                        .flat_map(|chunk| {
                            let mut local_chunks = Vec::new();
                            let mut qps = Vec::with_capacity(64);
                            let mut queries = Vec::with_capacity(64);
                            let mut batched_qps =
                                BatchedQueryParams64::new(&cp_doc, bp_doc).unwrap();

                            for i in 0..64 {
                                let mut success = false;
                                let mut qp = batched_qps.extract_query_params(i).unwrap();

                                if let Ok(q) = qp.generate_query(chunk[i]) {
                                    qps.push(qp);
                                    queries.push(q);
                                    success = true;
                                }

                                if !success {
                                    loop {
                                        let mut fallback_qp =
                                            QueryParams::new(&cp_doc, bp_doc).unwrap();
                                        if let Ok(q) = fallback_qp.generate_query(chunk[i]) {
                                            qps.push(fallback_qp);
                                            queries.push(q);
                                            break;
                                        }
                                    }
                                }
                            }

                            let interleaved: Vec<[u32; 64]> = (0..m_doc)
                                .map(|row| {
                                    let mut vals = [0u32; 64];
                                    for j in 0..64 {
                                        vals[j] = queries[j].as_slice()[row];
                                    }
                                    vals
                                })
                                .collect();

                            let bq = BatchedQuery64 { interleaved };
                            let d_resp = doc_shard.respond_batched_64(&bq).unwrap();
                            let resp: BatchedResponse64 = bincode::deserialize(&d_resp).unwrap();
                            let width = doc_shard.get_db().get_matrix_width_self();

                            for j in 0..64 {
                                let mut single_resp = Vec::with_capacity(width);
                                for w in 0..width {
                                    single_resp.push(resp.results[w][j]);
                                }
                                let parsed_resp = Response(single_resp);
                                let output = parsed_resp.parse_output_as_base64(&qps[j]);

                                if let Ok(decoded_chunk) = fs.decode_base64(&output) {
                                    local_chunks.push(decoded_chunk);
                                } else {
                                    local_chunks.push(vec![0u8; 1024]);
                                }
                            }
                            local_chunks
                        })
                        .collect();

                    retrieved_doc_chunks.truncate(original_docmap_len);

                    let mut docmap_success = true;
                    for (i, (retrieved, expected)) in retrieved_doc_chunks
                        .iter()
                        .zip(expected_file_chunks.iter())
                        .enumerate()
                    {
                        if retrieved != expected {
                            error!("> DocMap chunk mismatch at query index {}", i);
                            docmap_success = false;
                        }
                    }

                    if docmap_success {
                        info!("> Success! All {} DocMap chunks exactly match the original encrypted file chunks on disk.", retrieved_doc_chunks.len());
                    } else {
                        error!("> DocMap chunk verification against disk failed!");
                    }
                    info!("> End-to-End DocMap Test Complete.");
                }
            } else {
                error!(
                    "> Failed: Only found {}/{} document payloads for '{}'.",
                    total_matched,
                    expected_paths.len(),
                    target_keyword
                );
            }
        }

        Ok(())
    }
}
