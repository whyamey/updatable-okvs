import json
import argparse
from pathlib import Path
import math
import random

def count_mappings(keyword_map):
    """Count total number of mappings in the keyword map"""
    return sum(len(docs) for docs in keyword_map.values())

def get_all_files(keyword_map):
    """Get unique files from keyword map"""
    return sorted(set(file for files in keyword_map.values() for file in files))

def save_files_list(files, output_dir, shard_num):
    output_path = Path(output_dir) / f"files_{shard_num}.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(files))

def save_keywords_list(keywords, output_dir, shard_num):
    output_path = Path(output_dir) / f"keywords_{shard_num}.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(sorted(keywords)))

def sample_keywords(keyword_map, num_keywords=100):
    keywords = list(keyword_map.keys())
    sampled = random.sample(keywords, min(num_keywords, len(keywords)))
    return sampled

def save_sample_keywords(keywords, output_dir):
    output_path = Path(output_dir) / "keywords_to_query"
    with open(output_path, 'w') as f:
        f.write('\n'.join(keywords))

def create_shards(input_path, output_dir, num_shards, sample_size=100):
    print(f"Loading keyword map from {input_path}...")
    with open(input_path, 'r') as f:
        full_map = json.load(f)
    
    if sample_size:
        sampled_keywords = sample_keywords(full_map, sample_size)
        save_sample_keywords(sampled_keywords, output_dir)
        print(f"Saved {len(sampled_keywords)} sample keywords to {output_dir}/keywords_to_query")
    
    # Sort keywords by mapping count in descending order
    sorted_keywords = sorted(full_map.items(), key=lambda x: len(x[1]), reverse=True)
    total_mappings = sum(len(docs) for _, docs in sorted_keywords)
    target_per_shard = total_mappings / num_shards
    
    print(f"Original map has {total_mappings:,} total mappings")
    print(f"Target mappings per shard: {target_per_shard:,}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shards = [dict() for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    
    # Distribute keywords using greedy approach
    for keyword, docs in sorted_keywords:
        # Find shard with minimum current size
        min_shard = min(range(num_shards), key=lambda i: shard_sizes[i])
        shards[min_shard][keyword] = docs
        shard_sizes[min_shard] += len(docs)
    
    # Save shards
    for i, shard in enumerate(shards):
        output_path = Path(output_dir) / f"shard_{i}.json"
        print(f"\nSaving shard {i}:")
        print(f"- Keywords: {len(shard):,}")
        print(f"- Mappings: {shard_sizes[i]:,}")
        print(f"- Path: {output_path}")
        
        files = get_all_files(shard)
        save_files_list(files, output_dir, i)
        save_keywords_list(shard.keys(), output_dir, i)
        
        with open(output_path, 'w') as f:
            json.dump(shard, f)

def main():
    parser = argparse.ArgumentParser(description='Split keyword map into shards')
    parser.add_argument('input_file', help='Input keyword map path')
    parser.add_argument('output_dir', help='Output directory for shards')
    parser.add_argument('num_shards', type=int, help='Number of shards to create')
    parser.add_argument('--sample_size', type=int, default=100, help='Number of keywords to sample (default: 100)')
    
    args = parser.parse_args()
    create_shards(args.input_file, args.output_dir, args.num_shards, args.sample_size)

if __name__ == "__main__":
    main()

