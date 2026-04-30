use crate::generate_mmap::EncryptedFileSystem;
use crate::tethys_dip::TethysDIP;

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
}

impl Benchmarker {
    pub fn new(
        p: usize,
        epsilon: f64,
        run_decryption_test: bool,
        keywords_file: Option<String>,
        data_dir: String,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encrypted_fs = EncryptedFileSystem::new(p, data_dir.clone())?;
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

        for keyword in keywords {
            let (base_keyword, num_ones) = TethysDIP::parse_keyword(keyword);

            if num_ones == 0 {
                indexes.insert(self.hash1(base_keyword, m));
                indexes.insert(self.hash2(base_keyword, m));
            } else {
                let h1 = (self.hash1(base_keyword, m) + self.hash1(&num_ones.to_string(), m)) % m;
                let h2 = (self.hash2(base_keyword, m) + self.hash2(&num_ones.to_string(), m)) % m;

                indexes.insert(h1);
                indexes.insert(h2);
            }
        }

        indexes
    }

    fn hash1(&self, key: &str, m: usize) -> usize {
        let hash = blake3::hash(key.as_bytes());
        let hash_bytes = hash.as_bytes();
        let hash_value = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        (hash_value as usize) % m
    }

    fn hash2(&self, key: &str, m: usize) -> usize {
        let mut keyed_hasher = blake3::Hasher::new_keyed(b"different key for second hashing");
        keyed_hasher.update(key.as_bytes());
        let hash = keyed_hasher.finalize();

        let hash_bytes = hash.as_bytes();
        let hash_value = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        (hash_value as usize) % m
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
            if keyword_size > 0 {
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
}
