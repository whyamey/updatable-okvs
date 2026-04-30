use blake3;
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

    pub fn build_valmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>) {
        let value_to_key_map: HashMap<Vec<u8>, String> = keyword_value_map
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        let mut buckets: HashMap<usize, Vec<Vec<u8>>> =
            (0..self.m).map(|i| (i, Vec::new())).collect();
        let mut stash = Vec::new();

        'outer: for (key, value) in keyword_value_map.iter() {
            let mut current_key = key.clone();
            let mut current_val = value.clone();

            for _ in 0..MAX_EVICTIONS {
                let h1 = self.hash1(&current_key);
                if buckets.get(&h1).map_or(0, |b| b.len()) < self.p {
                    buckets.entry(h1).or_default().push(current_val);
                    continue 'outer;
                }

                let bucket1 = buckets.get_mut(&h1).unwrap();
                let evicted_val = bucket1.swap_remove(0);
                bucket1.push(current_val);

                let evicted_key = value_to_key_map.get(&evicted_val).unwrap().clone();

                let h2 = self.hash2(&evicted_key);
                if buckets.get(&h2).map_or(0, |b| b.len()) < self.p {
                    buckets.entry(h2).or_default().push(evicted_val);
                    continue 'outer;
                }

                let bucket2 = buckets.get_mut(&h2).unwrap();
                current_val = bucket2.swap_remove(0);
                bucket2.push(evicted_val);

                current_key = value_to_key_map.get(&current_val).unwrap().clone();
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

        'outer: for (key, value) in keyword_value_map.iter() {
            let mut current_key = key.clone();
            let mut current_val = value.clone();

            for _ in 0..10000 {
                let h1 = self.hash1(&current_key);
                if buckets.get(&h1).map_or(0, |b| b.len()) < self.p {
                    buckets.entry(h1).or_default().push(current_val);
                    continue 'outer;
                }

                let bucket1 = buckets.get_mut(&h1).unwrap();
                let evicted_val = bucket1.swap_remove(0);
                bucket1.push(current_val);

                let evicted_key = value_to_key_map.get(&evicted_val).unwrap().clone();

                let h2 = self.hash2(&evicted_key);
                if buckets.get(&h2).map_or(0, |b| b.len()) < self.p {
                    buckets.entry(h2).or_default().push(evicted_val);
                    continue 'outer;
                }

                let bucket2 = buckets.get_mut(&h2).unwrap();
                current_val = bucket2.swap_remove(0);
                bucket2.push(evicted_val);

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
        let hash_bytes = hash.as_bytes();
        let hash_value = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }

    fn hash2(&self, key: &str) -> usize {
        let mut keyed_hasher = blake3::Hasher::new_keyed(b"different key for second hashing");
        keyed_hasher.update(key.as_bytes());
        let hash = keyed_hasher.finalize();

        let hash_bytes = hash.as_bytes();
        let hash_value = u64::from_le_bytes(hash_bytes[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }
}
