use blake3;
use rand::Rng;
use std::collections::HashMap;

const MAX_EVICTIONS: usize = 100;

pub struct TethysDIP {
    m: usize,
    p: usize,
}

impl TethysDIP {
    pub fn new(m: usize, p: usize) -> Self {
        TethysDIP { m, p }
    }

    /// Tabulation Hashing is just (hash(kwd) + hash(counter)) mod m
    fn get_hashes(&self, keyword: &str) -> (usize, usize) {
        let (base_keyword, num_ones) = Self::parse_keyword(keyword);

        if num_ones == 0 {
            (self.hash1(base_keyword), self.hash2(base_keyword))
        } else {
            let counter_str = num_ones.to_string();
            let h1 = (self.hash1(base_keyword) + self.hash1(&counter_str)) % self.m;
            let h2 = (self.hash2(base_keyword) + self.hash2(&counter_str)) % self.m;
            (h1, h2)
        }
    }

    pub fn build_valmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>) {
        // Map to find the keyword associated with a value during eviction
        let value_to_key_map: HashMap<Vec<u8>, String> = keyword_value_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        let mut buckets: HashMap<usize, Vec<Vec<u8>>> =
            (0..self.m).map(|i| (i, Vec::new())).collect();
        let mut stash = Vec::new();
        let mut rng = rand::thread_rng();

        'outer: for (combined_key, value) in keyword_value_map.iter() {
            let keyword_part = combined_key.split("||").next().unwrap();
            let mut current_val = value.clone();
            let mut current_key = keyword_part.to_string();

            for _ in 0..MAX_EVICTIONS {
                let (h1, h2) = self.get_hashes(&current_key);

                let b1 = buckets.get_mut(&h1).unwrap();
                if b1.len() < self.p {
                    b1.push(current_val);
                    continue 'outer;
                }

                let b2 = buckets.get_mut(&h2).unwrap();
                if b2.len() < self.p {
                    b2.push(current_val);
                    continue 'outer;
                }

                // Evict randomly from one of the two buckets
                let target_hash = if rng.gen_bool(0.5) { h1 } else { h2 };
                let bucket = buckets.get_mut(&target_hash).unwrap();

                let idx = rng.gen_range(0..bucket.len());
                let evicted_val = bucket.remove(idx);
                bucket.push(current_val);

                current_val = evicted_val;
                let full_evicted_key = value_to_key_map.get(&current_val).unwrap();
                current_key = full_evicted_key.split("||").next().unwrap().to_string();
            }

            stash.push(current_val);
        }

        (buckets, stash)
    }

    pub fn build_volmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<(String, usize)>>,
    ) -> (
        HashMap<usize, Vec<Vec<(String, usize)>>>,
        Vec<Vec<(String, usize)>>,
    ) {
        let value_to_key_map: HashMap<Vec<(String, usize)>, String> = keyword_value_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        let mut buckets: HashMap<usize, Vec<Vec<(String, usize)>>> =
            (0..self.m).map(|i| (i, Vec::new())).collect();
        let mut stash = Vec::new();
        let mut rng = rand::thread_rng();

        'outer: for (key, value) in keyword_value_map.iter() {
            let mut current_val = value.clone();
            let mut current_key = key.clone();

            for _ in 0..10000 {
                let (h1, h2) = self.get_hashes(&current_key);

                let b1 = buckets.get_mut(&h1).unwrap();
                if b1.len() < self.p {
                    b1.push(current_val);
                    continue 'outer;
                }

                let b2 = buckets.get_mut(&h2).unwrap();
                if b2.len() < self.p {
                    b2.push(current_val);
                    continue 'outer;
                }

                let target_hash = if rng.gen_bool(0.5) { h1 } else { h2 };
                let bucket = buckets.get_mut(&target_hash).unwrap();

                let idx = rng.gen_range(0..bucket.len());
                let evicted_val = bucket.remove(idx);
                bucket.push(current_val);

                current_val = evicted_val;
                current_key = value_to_key_map.get(&current_val).unwrap().clone();
            }

            stash.push(current_val);
        }

        (buckets, stash)
    }

    pub fn parse_keyword(keyword: &str) -> (&str, usize) {
        let mut count = 0;
        let mut base_keyword = keyword;

        while base_keyword.ends_with('1') {
            if base_keyword.len() <= 1 {
                break;
            }
            base_keyword = &base_keyword[..base_keyword.len() - 1];
            count += 1;
        }
        (base_keyword, count)
    }

    fn hash1(&self, key: &str) -> usize {
        let hash = blake3::hash(key.as_bytes());
        let hash_value = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }

    fn hash2(&self, key: &str) -> usize {
        let mut keyed_hasher = blake3::Hasher::new_keyed(b"different key for second hashing");
        keyed_hasher.update(key.as_bytes());
        let hash = keyed_hasher.finalize();
        let hash_value = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }
}
