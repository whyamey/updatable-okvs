use std::sync::Arc;

use blake3;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};

const MAX_EVICTIONS: usize = 5;

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
    pub fn get_hashes(&self, keyword: &str) -> [usize; N] {
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
        let mut buckets: Vec<Vec<(String, Arc<Vec<u8>>)>> =
            (0..self.m).map(|_| Vec::with_capacity(self.p)).collect();

        let mut stash = Vec::new();
        let mut rng = rand::thread_rng();

        let mut precomputed_hashes: HashMap<String, [usize; N]> =
            HashMap::with_capacity(keyword_value_map.len());
        for combined_key in keyword_value_map.keys() {
            let keyword_part = combined_key.split("||").next().unwrap();
            precomputed_hashes.insert(keyword_part.to_string(), self.get_hashes(keyword_part));
        }

        'outer: for (combined_key, value) in keyword_value_map.iter() {
            let keyword_part = combined_key.split("||").next().unwrap();
            let mut current_val = Arc::new(value.clone());
            let mut current_key = keyword_part.to_string();

            for _ in 0..MAX_EVICTIONS {
                let hashes = precomputed_hashes.get(&current_key).unwrap();

                let mut placed = false;
                for &h in hashes {
                    let bucket = &mut buckets[h];
                    if bucket.len() < self.p {
                        bucket.push((current_key.clone(), Arc::clone(&current_val)));
                        placed = true;
                        break;
                    }
                }
                if placed {
                    continue 'outer;
                }

                let target_hash = hashes[rng.gen_range(0..N)];
                let bucket = &mut buckets[target_hash];

                let idx = rng.gen_range(0..bucket.len());
                let evicted = bucket.swap_remove(idx);
                bucket.push((current_key.clone(), current_val));

                current_key = evicted.0;
                current_val = evicted.1;
            }

            stash.push(current_val);
        }

        let final_buckets: HashMap<usize, Vec<Vec<u8>>> = buckets
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v.into_iter().map(|(_, arc)| (*arc).clone()).collect()))
            .collect();

        let final_stash = stash.into_iter().map(|arc| (*arc).clone()).collect();

        (final_buckets, final_stash)
    }

    pub fn build_volmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<(String, usize)>>,
    ) -> (
        HashMap<usize, Vec<Vec<(String, usize)>>>,
        Vec<Vec<(String, usize)>>,
    ) {
        let mut buckets: Vec<Vec<(String, Arc<Vec<(String, usize)>>)>> =
            (0..self.m).map(|_| Vec::with_capacity(self.p)).collect();

        let mut stash = Vec::new();
        let mut rng = rand::thread_rng();

        let mut precomputed_hashes: HashMap<String, [usize; N]> =
            HashMap::with_capacity(keyword_value_map.len());
        for key in keyword_value_map.keys() {
            precomputed_hashes.insert(key.clone(), self.get_hashes(key));
        }

        'outer: for (key, value) in keyword_value_map.iter() {
            let mut current_val = Arc::new(value.clone());
            let mut current_key = key.clone();

            for _ in 0..MAX_EVICTIONS {
                let hashes = precomputed_hashes.get(&current_key).unwrap();

                let mut placed = false;
                for &h in hashes {
                    let bucket = &mut buckets[h];
                    if bucket.len() < self.p {
                        bucket.push((current_key.clone(), Arc::clone(&current_val)));
                        placed = true;
                        break;
                    }
                }
                if placed {
                    continue 'outer;
                }

                let target_hash = hashes[rng.gen_range(0..N)];
                let bucket = &mut buckets[target_hash];

                let idx = rng.gen_range(0..bucket.len());
                let evicted = bucket.swap_remove(idx);
                bucket.push((current_key.clone(), current_val));

                current_key = evicted.0;
                current_val = evicted.1;
            }

            stash.push(current_val);
        }

        let final_buckets: HashMap<usize, Vec<Vec<(String, usize)>>> = buckets
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v.into_iter().map(|(_, arc)| (*arc).clone()).collect()))
            .collect();

        let final_stash = stash.into_iter().map(|arc| (*arc).clone()).collect();

        (final_buckets, final_stash)
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

    pub fn get_hashes(&self, keyword: &str) -> [usize; N] {
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
        let mut buckets: Vec<Vec<(String, Arc<Vec<u8>>)>> =
            (0..self.m).map(|_| Vec::with_capacity(self.p)).collect();
        let mut stash = Vec::new();

        let mut precomputed_hashes: HashMap<String, [usize; N]> =
            HashMap::with_capacity(keyword_value_map.len());
        for combined_key in keyword_value_map.keys() {
            let keyword_part = combined_key.split("||").next().unwrap();
            precomputed_hashes.insert(keyword_part.to_string(), self.get_hashes(keyword_part));
        }

        for (combined_key, value) in keyword_value_map.iter() {
            let keyword_part = combined_key.split("||").next().unwrap();
            let current_val = Arc::new(value.clone());
            let current_key = keyword_part.to_string();

            let hashes = precomputed_hashes.get(&current_key).unwrap();
            let mut placed = false;
            for &h in hashes {
                let bucket = &mut buckets[h];
                if bucket.len() < self.p {
                    bucket.push((current_key.clone(), Arc::clone(&current_val)));
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            type TupleType = (String, Arc<Vec<u8>>);
            let mut queue: VecDeque<(usize, TupleType, Vec<(usize, TupleType, TupleType)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();

            let insert_tuple = (current_key.clone(), Arc::clone(&current_val));

            for &h in hashes {
                queue.push_back((h, insert_tuple.clone(), Vec::new()));
                visited_buckets.insert(h);
            }

            let mut success = false;
            let mut nodes_expanded = 0;

            'bfs: while let Some((curr_b, val_to_insert, path)) = queue.pop_front() {
                if nodes_expanded >= 5000 {
                    break;
                }
                nodes_expanded += 1;

                let current_bucket_items = buckets[curr_b].clone();

                for (i, item) in current_bucket_items.iter().enumerate() {
                    let b_key = &item.0;
                    let item_hashes = precomputed_hashes.get(b_key).unwrap();

                    for &alt_b in item_hashes {
                        if alt_b == curr_b {
                            continue;
                        }

                        if buckets[alt_b].len() < self.p {
                            buckets[alt_b].push(item.clone());
                            let cb = &mut buckets[curr_b];
                            cb.remove(i);
                            cb.push(val_to_insert.clone());

                            for (b, v_in, v_out) in path.iter().rev() {
                                let bb = &mut buckets[*b];
                                if let Some(pos) = bb
                                    .iter()
                                    .position(|x| Arc::ptr_eq(&x.1, &v_out.1) || x.1 == v_out.1)
                                {
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

        let final_buckets: HashMap<usize, Vec<Vec<u8>>> = buckets
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v.into_iter().map(|(_, arc)| (*arc).clone()).collect()))
            .collect();
        let final_stash = stash.into_iter().map(|arc| (*arc).clone()).collect();

        (final_buckets, final_stash)
    }

    pub fn build_volmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<(String, usize)>>,
    ) -> (
        HashMap<usize, Vec<Vec<(String, usize)>>>,
        Vec<Vec<(String, usize)>>,
    ) {
        let mut buckets: Vec<Vec<(String, Arc<Vec<(String, usize)>>)>> =
            (0..self.m).map(|_| Vec::with_capacity(self.p)).collect();
        let mut stash = Vec::new();

        let mut precomputed_hashes: HashMap<String, [usize; N]> =
            HashMap::with_capacity(keyword_value_map.len());
        for key in keyword_value_map.keys() {
            precomputed_hashes.insert(key.clone(), self.get_hashes(key));
        }

        for (key, value) in keyword_value_map.iter() {
            let current_val = Arc::new(value.clone());
            let current_key = key.clone();

            let hashes = precomputed_hashes.get(&current_key).unwrap();
            let mut placed = false;
            for &h in hashes {
                let bucket = &mut buckets[h];
                if bucket.len() < self.p {
                    bucket.push((current_key.clone(), Arc::clone(&current_val)));
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            type TupleType = (String, Arc<Vec<(String, usize)>>);
            let mut queue: VecDeque<(usize, TupleType, Vec<(usize, TupleType, TupleType)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();

            let insert_tuple = (current_key.clone(), Arc::clone(&current_val));

            for &h in hashes {
                queue.push_back((h, insert_tuple.clone(), Vec::new()));
                visited_buckets.insert(h);
            }

            let mut success = false;
            let mut nodes_expanded = 0;

            'bfs: while let Some((curr_b, val_to_insert, path)) = queue.pop_front() {
                if nodes_expanded >= 10000 {
                    break;
                }
                nodes_expanded += 1;

                let current_bucket_items = buckets[curr_b].clone();

                for (i, item) in current_bucket_items.iter().enumerate() {
                    let b_key = &item.0;
                    let item_hashes = precomputed_hashes.get(b_key).unwrap();

                    for &alt_b in item_hashes {
                        if alt_b == curr_b {
                            continue;
                        }

                        if buckets[alt_b].len() < self.p {
                            buckets[alt_b].push(item.clone());
                            let cb = &mut buckets[curr_b];
                            cb.remove(i);
                            cb.push(val_to_insert.clone());

                            for (b, v_in, v_out) in path.iter().rev() {
                                let bb = &mut buckets[*b];
                                if let Some(pos) = bb
                                    .iter()
                                    .position(|x| Arc::ptr_eq(&x.1, &v_out.1) || x.1 == v_out.1)
                                {
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

        let final_buckets: HashMap<usize, Vec<Vec<(String, usize)>>> = buckets
            .into_iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| (k, v.into_iter().map(|(_, arc)| (*arc).clone()).collect()))
            .collect();
        let final_stash = stash.into_iter().map(|arc| (*arc).clone()).collect();

        (final_buckets, final_stash)
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
