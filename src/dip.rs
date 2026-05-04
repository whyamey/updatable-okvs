use blake3;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};

const MAX_EVICTIONS: usize = 100;

pub struct HashDIP<const N: usize = 2> {
    m: usize,
    p: usize,
}

impl<const N: usize> HashDIP<N> {
    pub fn new(m: usize, p: usize) -> Self {
        assert!(N >= 1 && N <= 4);
        HashDIP { m, p }
    }

    fn hash(&self, idx: usize, key: &str) -> usize {
        let hash = match idx {
            0 => blake3::hash(key.as_bytes()),
            1 => {
                let mut hasher = blake3::Hasher::new_keyed(b"different key for second hashing");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            2 => {
                let mut hasher = blake3::Hasher::new_keyed(b"completely new key for 3rd hash!");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            3 => {
                let mut hasher = blake3::Hasher::new_keyed(b"and a fourth key just in case!!!");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            _ => unreachable!(),
        };
        let hash_value = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }

    /// Tabulation Hashing is just (hash(kwd) + hash(counter)) mod m
    fn get_hashes(&self, keyword: &str) -> [usize; N] {
        let (base_keyword, num_ones) = Self::parse_keyword(keyword);
        let mut hashes = [0; N];

        if num_ones == 0 {
            for i in 0..N {
                hashes[i] = self.hash(i, base_keyword);
            }
        } else {
            let counter_str = num_ones.to_string();
            for i in 0..N {
                hashes[i] = (self.hash(i, base_keyword) + self.hash(i, &counter_str)) % self.m;
            }
        }
        hashes
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
        let mut rng = rand::thread_rng();

        'outer: for (combined_key, value) in keyword_value_map.iter() {
            let keyword_part = combined_key.split("||").next().unwrap();
            let mut current_val = value.clone();
            let mut current_key = keyword_part.to_string();

            for _ in 0..MAX_EVICTIONS {
                let hashes = self.get_hashes(&current_key);

                let mut placed = false;
                for &h in &hashes {
                    let bucket = buckets.get_mut(&h).unwrap();
                    if bucket.len() < self.p {
                        bucket.push(current_val.clone());
                        placed = true;
                        break;
                    }
                }
                if placed {
                    continue 'outer;
                }

                let target_hash = hashes[rng.gen_range(0..N)];
                let bucket = buckets.get_mut(&target_hash).unwrap();

                let idx = rng.gen_range(0..bucket.len());
                let evicted_val = bucket.swap_remove(idx);
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

            for _ in 0..MAX_EVICTIONS {
                let hashes = self.get_hashes(&current_key);

                let mut placed = false;
                for &h in &hashes {
                    let bucket = buckets.get_mut(&h).unwrap();
                    if bucket.len() < self.p {
                        bucket.push(current_val.clone());
                        placed = true;
                        break;
                    }
                }
                if placed {
                    continue 'outer;
                }

                let target_hash = hashes[rng.gen_range(0..N)];
                let bucket = buckets.get_mut(&target_hash).unwrap();

                let idx = rng.gen_range(0..bucket.len());
                let evicted_val = bucket.swap_remove(idx);
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
}

pub struct BfsDIP<const N: usize = 2> {
    m: usize,
    p: usize,
}

impl<const N: usize> BfsDIP<N> {
    pub fn new(m: usize, p: usize) -> Self {
        assert!(N >= 1 && N <= 4);
        BfsDIP { m, p }
    }

    fn hash(&self, idx: usize, key: &str) -> usize {
        let hash = match idx {
            0 => blake3::hash(key.as_bytes()),
            1 => {
                let mut hasher = blake3::Hasher::new_keyed(b"different key for second hashing");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            2 => {
                let mut hasher = blake3::Hasher::new_keyed(b"completely new key for 3rd hash!");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            3 => {
                let mut hasher = blake3::Hasher::new_keyed(b"and a fourth key just in case!!!");
                hasher.update(key.as_bytes());
                hasher.finalize()
            }
            _ => unreachable!(),
        };
        let hash_value = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        (hash_value as usize) % self.m
    }

    fn get_hashes(&self, keyword: &str) -> [usize; N] {
        let (base_keyword, num_ones) = Self::parse_keyword(keyword);
        let mut hashes = [0; N];

        if num_ones == 0 {
            for i in 0..N {
                hashes[i] = self.hash(i, base_keyword);
            }
        } else {
            let counter_str = num_ones.to_string();
            for i in 0..N {
                hashes[i] = (self.hash(i, base_keyword) + self.hash(i, &counter_str)) % self.m;
            }
        }
        hashes
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

        for (combined_key, value) in keyword_value_map.iter() {
            let keyword_part = combined_key.split("||").next().unwrap();
            let current_val = value.clone();
            let current_key = keyword_part.to_string();

            let hashes = self.get_hashes(&current_key);
            let mut placed = false;
            for &h in &hashes {
                let bucket = buckets.get_mut(&h).unwrap();
                if bucket.len() < self.p {
                    bucket.push(current_val.clone());
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            let mut queue: VecDeque<(usize, Vec<u8>, Vec<(usize, Vec<u8>, Vec<u8>)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();

            for &h in &hashes {
                queue.push_back((h, current_val.clone(), Vec::new()));
                visited_buckets.insert(h);
            }

            let mut success = false;
            let mut nodes_expanded = 0;

            'bfs: while let Some((curr_b, val_to_insert, path)) = queue.pop_front() {
                if nodes_expanded >= 5000 {
                    break;
                }
                nodes_expanded += 1;

                let current_bucket_items = buckets.get(&curr_b).unwrap().clone();

                for (i, item) in current_bucket_items.iter().enumerate() {
                    let full_key = value_to_key_map.get(item).unwrap();
                    let b_key = full_key.split("||").next().unwrap();
                    let item_hashes = self.get_hashes(b_key);

                    for &alt_b in &item_hashes {
                        if alt_b == curr_b {
                            continue;
                        }

                        if buckets.get(&alt_b).unwrap().len() < self.p {
                            buckets.get_mut(&alt_b).unwrap().push(item.clone());
                            let cb = buckets.get_mut(&curr_b).unwrap();
                            cb.remove(i);
                            cb.push(val_to_insert.clone());

                            for (b, v_in, v_out) in path.iter().rev() {
                                let bb = buckets.get_mut(b).unwrap();
                                if let Some(pos) = bb.iter().position(|x| x == v_out) {
                                    bb.remove(pos);
                                }
                                bb.push(v_in.clone());
                            }

                            success = true;
                            break 'bfs;
                        }

                        if !visited_buckets.contains(&alt_b) {
                            visited_buckets.insert(alt_b);
                            let mut new_path = path.clone();
                            new_path.push((curr_b, val_to_insert.clone(), item.clone()));
                            queue.push_back((alt_b, item.clone(), new_path));
                        }
                    }
                }
            }

            if !success {
                stash.push(current_val);
            }
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

        for (key, value) in keyword_value_map.iter() {
            let current_val = value.clone();
            let current_key = key.clone();

            let hashes = self.get_hashes(&current_key);
            let mut placed = false;
            for &h in &hashes {
                let bucket = buckets.get_mut(&h).unwrap();
                if bucket.len() < self.p {
                    bucket.push(current_val.clone());
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            let mut queue: VecDeque<(
                usize,
                Vec<(String, usize)>,
                Vec<(usize, Vec<(String, usize)>, Vec<(String, usize)>)>,
            )> = VecDeque::new();
            let mut visited_buckets = HashSet::new();

            for &h in &hashes {
                queue.push_back((h, current_val.clone(), Vec::new()));
                visited_buckets.insert(h);
            }

            let mut success = false;
            let mut nodes_expanded = 0;

            'bfs: while let Some((curr_b, val_to_insert, path)) = queue.pop_front() {
                if nodes_expanded >= 10000 {
                    break;
                }
                nodes_expanded += 1;

                let current_bucket_items = buckets.get(&curr_b).unwrap().clone();

                for (i, item) in current_bucket_items.iter().enumerate() {
                    let b_key = value_to_key_map.get(item).unwrap();
                    let item_hashes = self.get_hashes(b_key);

                    for &alt_b in &item_hashes {
                        if alt_b == curr_b {
                            continue;
                        }

                        if buckets.get(&alt_b).unwrap().len() < self.p {
                            buckets.get_mut(&alt_b).unwrap().push(item.clone());
                            let cb = buckets.get_mut(&curr_b).unwrap();
                            cb.remove(i);
                            cb.push(val_to_insert.clone());

                            for (b, v_in, v_out) in path.iter().rev() {
                                let bb = buckets.get_mut(b).unwrap();
                                if let Some(pos) = bb.iter().position(|x| x == v_out) {
                                    bb.remove(pos);
                                }
                                bb.push(v_in.clone());
                            }

                            success = true;
                            break 'bfs;
                        }

                        if !visited_buckets.contains(&alt_b) {
                            visited_buckets.insert(alt_b);
                            let mut new_path = path.clone();
                            new_path.push((curr_b, val_to_insert.clone(), item.clone()));
                            queue.push_back((alt_b, item.clone(), new_path));
                        }
                    }
                }
            }

            if !success {
                stash.push(current_val);
            }
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
}
