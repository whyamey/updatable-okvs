use std::sync::Arc;

use blake3;
use rand::Rng;
use std::collections::{HashMap, HashSet, VecDeque};

const MAX_EVICTIONS: usize = 10;
const MAX_BFS_NODES: usize = 20;
const MAX_DEPTH: usize = 5;

#[derive(Clone, Debug)]
pub struct MaskedItem<T> {
    pub key: Arc<String>,
    pub val: Arc<T>,
    // We are already decided on 4 hashes, so this is hard-coded for now and I am lazy.
    pub mask: [bool; 4],
}

impl<T> MaskedItem<T> {
    pub fn new(key: Arc<String>, val: Arc<T>) -> Self {
        MaskedItem {
            key,
            val,
            mask: [false; 4],
        }
    }
}

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

    pub fn get_hashes(&self, keyword: &str) -> [usize; N] {
        let mut hashes = [0; N];
        for i in 0..N {
            hashes[i] = self.hash(i, keyword);
        }
        hashes
    }

    pub fn build_valmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>) {
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<u8>>)>> =
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
            let mut current_key = Arc::new(keyword_part.to_string());

            for _ in 0..MAX_EVICTIONS {
                let hashes = precomputed_hashes.get(current_key.as_str()).unwrap();
                let mut placed = false;

                for &h in hashes {
                    let bucket = &mut buckets[h];
                    if bucket.len() < self.p {
                        bucket.push((Arc::clone(&current_key), Arc::clone(&current_val)));
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

                bucket.push((Arc::clone(&current_key), current_val));
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
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<(String, usize)>>)>> =
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
            let mut current_key = Arc::new(key.clone());

            for _ in 0..MAX_EVICTIONS {
                let hashes = precomputed_hashes.get(current_key.as_str()).unwrap();
                let mut placed = false;

                for &h in hashes {
                    let bucket = &mut buckets[h];
                    if bucket.len() < self.p {
                        bucket.push((Arc::clone(&current_key), Arc::clone(&current_val)));
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

                bucket.push((Arc::clone(&current_key), current_val));
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

    pub fn build_stateful_valmap(
        &self,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (
        HashMap<usize, Vec<MaskedItem<Vec<u8>>>>,
        Vec<MaskedItem<Vec<u8>>>,
    ) {
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<u8>>)>> =
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
            let mut current_key = Arc::new(keyword_part.to_string());

            for _ in 0..MAX_EVICTIONS {
                let hashes = precomputed_hashes
                    .get(current_key.as_str())
                    .copied()
                    .unwrap_or_else(|| self.get_hashes(&current_key));
                let mut placed = false;

                for &h in &hashes {
                    if buckets[h].len() < self.p {
                        buckets[h].push((Arc::clone(&current_key), Arc::clone(&current_val)));
                        placed = true;
                        break;
                    }
                }

                if placed {
                    continue 'outer;
                }

                let target_hash = hashes[rng.gen_range(0..N)];
                let idx = rng.gen_range(0..buckets[target_hash].len());
                let evicted = buckets[target_hash].swap_remove(idx);

                buckets[target_hash].push((Arc::clone(&current_key), current_val));
                current_key = evicted.0;
                current_val = evicted.1;
            }

            stash.push((current_key, current_val));
        }

        let mut masked_buckets = HashMap::new();
        for (b_id, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut masked_bucket = Vec::with_capacity(self.p);
            for (k, v) in bucket {
                let mut item = MaskedItem::new(k.clone(), v);
                let hashes = precomputed_hashes
                    .get(k.as_str())
                    .copied()
                    .unwrap_or_else(|| self.get_hashes(&k));
                if let Some(hash_idx) = hashes.iter().position(|&h| h == b_id) {
                    item.mask[hash_idx] = true;
                }
                masked_bucket.push(item);
            }
            masked_buckets.insert(b_id, masked_bucket);
        }

        let masked_stash = stash
            .into_iter()
            .map(|(k, v)| MaskedItem::new(k, v))
            .collect();
        (masked_buckets, masked_stash)
    }

    pub fn insert_valmap_oblivious(
        &self,
        keyword: &str,
        val: Arc<Vec<u8>>,
        buckets: &mut HashMap<usize, Vec<MaskedItem<Vec<u8>>>>,
        stash: &mut Vec<MaskedItem<Vec<u8>>>,
        rng: &mut impl rand::Rng,
    ) -> Vec<usize> {
        let mut current_item = MaskedItem::new(Arc::new(keyword.to_string()), val);
        let mut success = false;
        let mut touched_buckets = Vec::new();

        for _step in 0..MAX_EVICTIONS {
            if !success {
                let hashes = self.get_hashes(&current_item.key);
                let mut placed = false;

                for (i, &h) in hashes.iter().enumerate() {
                    let bucket = buckets
                        .entry(h)
                        .or_insert_with(|| Vec::with_capacity(self.p));

                    if bucket.len() < self.p {
                        current_item.mask = [false; 4];
                        current_item.mask[i] = true;
                        bucket.push(current_item.clone());

                        success = true;
                        placed = true;
                        touched_buckets.push(h);
                        break;
                    }
                }

                if !placed {
                    let target_idx = rng.gen_range(0..N);
                    let target_hash = hashes[target_idx];
                    let bucket = buckets
                        .entry(target_hash)
                        .or_insert_with(|| Vec::with_capacity(self.p));

                    let evict_idx = rng.gen_range(0..bucket.len());
                    let evicted = bucket.swap_remove(evict_idx);

                    current_item.mask = [false; 4];
                    current_item.mask[target_idx] = true;
                    bucket.push(current_item.clone());

                    touched_buckets.push(target_hash);

                    current_item = evicted;
                }
            } else {
                let dummy_hash = rng.gen_range(0..self.m);
                let bucket = buckets
                    .entry(dummy_hash)
                    .or_insert_with(|| Vec::with_capacity(self.p));

                if !bucket.is_empty() {
                    let dummy = bucket.swap_remove(0);
                    bucket.push(dummy);
                }
                touched_buckets.push(dummy_hash);
            }
        }

        if !success {
            stash.push(current_item);
        }

        touched_buckets.sort_unstable();
        touched_buckets.dedup();

        const EXACT_TOUCHES: usize = MAX_EVICTIONS;

        while touched_buckets.len() < EXACT_TOUCHES {
            let dummy_hash = rng.gen_range(0..self.m);
            if !touched_buckets.contains(&dummy_hash) {
                let bucket = buckets
                    .entry(dummy_hash)
                    .or_insert_with(|| Vec::with_capacity(self.p));

                if !bucket.is_empty() {
                    let dummy = bucket.swap_remove(0);
                    bucket.push(dummy);
                }
                touched_buckets.push(dummy_hash);
            }
        }

        touched_buckets
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
        let mut hashes = [0; N];
        for i in 0..N {
            hashes[i] = self.hash(i, keyword);
        }
        hashes
    }

    pub fn build_valmap(
        &self,
        _keyword_count_map: &HashMap<String, usize>,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (HashMap<usize, Vec<Vec<u8>>>, Vec<Vec<u8>>) {
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<u8>>)>> =
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
            let current_key = Arc::new(keyword_part.to_string());
            let hashes = precomputed_hashes.get(current_key.as_str()).unwrap();
            let mut placed = false;

            for &h in hashes {
                let bucket = &mut buckets[h];
                if bucket.len() < self.p {
                    bucket.push((Arc::clone(&current_key), Arc::clone(&current_val)));
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            type TupleType = (Arc<String>, Arc<Vec<u8>>);
            let mut queue: VecDeque<(usize, TupleType, Vec<(usize, TupleType, TupleType)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();
            let insert_tuple = (Arc::clone(&current_key), Arc::clone(&current_val));

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
                    let b_key = item.0.as_str();
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
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<(String, usize)>>)>> =
            (0..self.m).map(|_| Vec::with_capacity(self.p)).collect();
        let mut stash = Vec::new();
        let mut precomputed_hashes: HashMap<String, [usize; N]> =
            HashMap::with_capacity(keyword_value_map.len());

        for key in keyword_value_map.keys() {
            precomputed_hashes.insert(key.clone(), self.get_hashes(key));
        }

        for (key, value) in keyword_value_map.iter() {
            let current_val = Arc::new(value.clone());
            let current_key = Arc::new(key.clone());
            let hashes = precomputed_hashes.get(current_key.as_str()).unwrap();
            let mut placed = false;

            for &h in hashes {
                let bucket = &mut buckets[h];
                if bucket.len() < self.p {
                    bucket.push((Arc::clone(&current_key), Arc::clone(&current_val)));
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            type TupleType = (Arc<String>, Arc<Vec<(String, usize)>>);
            let mut queue: VecDeque<(usize, TupleType, Vec<(usize, TupleType, TupleType)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();
            let insert_tuple = (Arc::clone(&current_key), Arc::clone(&current_val));

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
                    let b_key = item.0.as_str();
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

    pub fn build_stateful_valmap(
        &self,
        keyword_value_map: &HashMap<String, Vec<u8>>,
    ) -> (
        HashMap<usize, Vec<MaskedItem<Vec<u8>>>>,
        Vec<MaskedItem<Vec<u8>>>,
    ) {
        let mut buckets: Vec<Vec<(Arc<String>, Arc<Vec<u8>>)>> =
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
            let current_key = Arc::new(keyword_part.to_string());
            let hashes = precomputed_hashes.get(current_key.as_str()).unwrap();
            let mut placed = false;

            for &h in hashes {
                if buckets[h].len() < self.p {
                    buckets[h].push((Arc::clone(&current_key), Arc::clone(&current_val)));
                    placed = true;
                    break;
                }
            }
            if placed {
                continue;
            }

            type TupleType = (Arc<String>, Arc<Vec<u8>>);
            let mut queue: VecDeque<(usize, TupleType, Vec<(usize, TupleType, TupleType)>)> =
                VecDeque::new();
            let mut visited_buckets = HashSet::new();
            let insert_tuple = (Arc::clone(&current_key), Arc::clone(&current_val));

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
                    let b_key = item.0.as_str();
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
                stash.push(insert_tuple);
            }
        }

        let mut masked_buckets = HashMap::new();
        for (b_id, bucket) in buckets.into_iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let mut masked_bucket = Vec::with_capacity(self.p);
            for (k, v) in bucket {
                let mut item = MaskedItem::new(k.clone(), v);
                let hashes = precomputed_hashes
                    .get(k.as_str())
                    .copied()
                    .unwrap_or_else(|| self.get_hashes(&k));
                if let Some(hash_idx) = hashes.iter().position(|&h| h == b_id) {
                    item.mask[hash_idx] = true;
                }
                masked_bucket.push(item);
            }
            masked_buckets.insert(b_id, masked_bucket);
        }

        let masked_stash = stash
            .into_iter()
            .map(|(k, v)| MaskedItem::new(k, v))
            .collect();
        (masked_buckets, masked_stash)
    }

    /// Oblivious, constant-time bounded BFS insert
    pub fn insert_valmap_oblivious(
        &self,
        keyword: &str,
        val: Arc<Vec<u8>>,
        buckets: &mut HashMap<usize, Vec<MaskedItem<Vec<u8>>>>,
        stash: &mut Vec<MaskedItem<Vec<u8>>>,
        rng: &mut impl rand::Rng,
    ) -> Vec<usize> {
        let mut to_place_item = MaskedItem::new(Arc::new(keyword.to_string()), val);
        let mut success = false;
        let initial_hashes = self.get_hashes(keyword);
        let mut touched_buckets = Vec::new();

        for (i, &h) in initial_hashes.iter().enumerate() {
            let bucket = buckets
                .entry(h)
                .or_insert_with(|| Vec::with_capacity(self.p));
            if bucket.len() < self.p {
                to_place_item.mask[i] = true;
                bucket.push(to_place_item.clone());
                success = true;
                touched_buckets.push(h);
                break;
            }
        }

        type BfsPath = Vec<((usize, usize), MaskedItem<Vec<u8>>, MaskedItem<Vec<u8>>)>;
        let mut queue: VecDeque<((usize, usize), MaskedItem<Vec<u8>>, BfsPath, usize)> =
            VecDeque::new();
        let mut visited = HashSet::new();

        if !success {
            for (i, &h) in initial_hashes.iter().enumerate() {
                queue.push_back(((h, i), to_place_item.clone(), Vec::new(), 0));
                visited.insert(h);
            }
        }

        let mut nodes_expanded = 0;

        while nodes_expanded < MAX_BFS_NODES {
            nodes_expanded += 1;

            if !success {
                // Randomized beam search because 256*4*depth is too much.
                if let Some(((curr_b, hash_idx_of_item), item_to_insert, path, cur_depth)) =
                    queue.pop_front()
                {
                    let current_bucket_items = buckets.get(&curr_b).cloned().unwrap_or_default();

                    use rand::seq::SliceRandom;
                    let mut indices: Vec<usize> = (0..current_bucket_items.len()).collect();
                    indices.shuffle(rng);

                    let mut branches_explored = 0;
                    const MAX_BRANCHES_PER_NODE: usize = 2;

                    'item_loop: for &i in &indices {
                        if branches_explored >= MAX_BRANCHES_PER_NODE {
                            break;
                        }

                        let bk_item = &current_bucket_items[i];
                        let item_hashes = self.get_hashes(&bk_item.key);

                        let mut alt_indices: Vec<usize> = (0..item_hashes.len()).collect();
                        alt_indices.shuffle(rng);

                        for &alt_idx in &alt_indices {
                            let alt_b = item_hashes[alt_idx];

                            if alt_b == curr_b || bk_item.mask[alt_idx] {
                                continue;
                            }

                            let alt_bucket = buckets
                                .entry(alt_b)
                                .or_insert_with(|| Vec::with_capacity(self.p));

                            if alt_bucket.len() < self.p {
                                alt_bucket.push(bk_item.clone());
                                touched_buckets.push(alt_b);

                                let cb = buckets.get_mut(&curr_b).unwrap();
                                cb.remove(i);
                                cb.push(item_to_insert.clone());
                                touched_buckets.push(curr_b);

                                let mut updated_bk = bk_item.clone();
                                updated_bk.mask[alt_idx] = true;

                                for ((b, _), v_in, v_out) in path.iter().rev() {
                                    let bb = buckets.get_mut(b).unwrap();
                                    if let Some(pos) =
                                        bb.iter().position(|x| Arc::ptr_eq(&x.val, &v_out.val))
                                    {
                                        bb.remove(pos);
                                    }
                                    bb.push(v_in.clone());
                                    touched_buckets.push(*b);
                                }
                                success = true;
                                break 'item_loop;
                            }

                            if cur_depth < MAX_DEPTH && !visited.contains(&alt_b) {
                                visited.insert(alt_b);
                                let mut new_path = path.clone();
                                new_path.push((
                                    (curr_b, hash_idx_of_item),
                                    item_to_insert.clone(),
                                    bk_item.clone(),
                                ));
                                queue.push_back((
                                    (alt_b, alt_idx),
                                    bk_item.clone(),
                                    new_path,
                                    cur_depth + 1,
                                ));
                                branches_explored += 1;
                                break;
                            }
                        }
                    }
                }
            } else {
                if !queue.is_empty() {
                    queue.pop_front();
                }

                let dummy_hash = rng.gen_range(0..self.m);
                if let Some(bucket) = buckets.get_mut(&dummy_hash) {
                    if !bucket.is_empty() {
                        let dummy = bucket.swap_remove(0);
                        bucket.push(dummy);
                    }
                }
                touched_buckets.push(dummy_hash);
            }
        }

        if !success {
            stash.push(to_place_item);
        }

        touched_buckets.sort_unstable();
        touched_buckets.dedup();

        touched_buckets.sort_unstable();
        touched_buckets.dedup();

        const EXACT_TOUCHES: usize = MAX_BFS_NODES + MAX_DEPTH + 2;

        while touched_buckets.len() < EXACT_TOUCHES {
            let dummy_hash = rng.gen_range(0..self.m);

            if !touched_buckets.contains(&dummy_hash) {
                let bucket = buckets
                    .entry(dummy_hash)
                    .or_insert_with(|| Vec::with_capacity(self.p));

                if !bucket.is_empty() {
                    let dummy = bucket.swap_remove(0);
                    bucket.push(dummy);
                }

                touched_buckets.push(dummy_hash);
            }
        }

        touched_buckets
    }
}
