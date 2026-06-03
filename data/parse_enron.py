import os
import json
import shutil
import string
import re
import nltk
import multiprocessing

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words


nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)

ENGLISH_WORDS = set(words.words())
STOP_WORDS = set(stopwords.words("english"))


def get_file_paths(root_dir):
    """Recursively get all file paths"""
    file_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return sorted(file_paths)


def is_valid_word(word):
    """Check if a word is valid"""
    return word.isalpha() and len(word) > 3 and word.lower() in ENGLISH_WORDS


def extract_keywords(args):
    """Extract keywords from a single file"""
    file_path, doc_id = args
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().lower()

        text = re.sub(r"^.*?:\s.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^>.*?$", "", text, flags=re.MULTILINE)
        text = re.sub(r"[^\w\s]", " ", text)

        words = word_tokenize(text)
        words = {
            word.lower()
            for word in words
            if word.lower() not in STOP_WORDS and is_valid_word(word)
        }

        return doc_id, list(words)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, []


def create_flattened_dataset(file_paths, output_dir):
    """Create flattened dataset with renamed files"""
    os.makedirs(output_dir, exist_ok=True)

    print("Copying files...")
    for i, src_path in enumerate(tqdm(file_paths), 1):
        dst_path = os.path.join(output_dir, str(i))
        shutil.copy2(src_path, dst_path)

    return len(file_paths)


def merge_results(results):
    """Merge results from parallel processing"""
    keyword_map = defaultdict(list)
    all_keywords = set()

    for doc_id, keywords in results:
        if doc_id is not None:
            all_keywords.update(keywords)
            for keyword in keywords:
                keyword_map[keyword].append(doc_id)

    return {k: sorted(set(v)) for k, v in keyword_map.items()}, sorted(
        list(all_keywords)
    )


def process_files(root_dir, output_dir):
    """Process all files in parallel"""
    print("Getting file paths...")
    file_paths = get_file_paths(root_dir)

    print("Creating flattened dataset...")
    total_files = create_flattened_dataset(file_paths, output_dir)

    print("Processing files for keywords...")
    process_args = [
        (os.path.join(output_dir, str(i)), i) for i in range(1, total_files + 1)
    ]

    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} CPU cores...")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(extract_keywords, process_args, chunksize=100),
                total=len(process_args),
                desc="Extracting keywords",
            )
        )

    print("Merging results...")
    keyword_map, all_keywords = merge_results(results)
    return keyword_map, all_keywords, total_files


def save_files(keyword_map, keywords_list, total_files, output_dir):
    """Save all output files"""
    print("Saving keyword map...")
    string_keyword_map = {k: [str(v) for v in vals] for k, vals in keyword_map.items()}

    with open(os.path.join(output_dir, "keyword_file_map.json"), "w") as f:
        json.dump(string_keyword_map, f, indent=2)

    print("Saving keywords list...")
    with open(os.path.join(output_dir, "keywords.txt"), "w") as f:
        for keyword in keywords_list:
            f.write(f"{keyword}\n")

    print("Saving files list...")
    with open(os.path.join(output_dir, "files.txt"), "w") as f:
        for i in range(1, total_files + 1):
            f.write(f"{i}\n")


def main():
    input_dir = "maildir"
    output_dir = "data"

    print("Starting processing...")
    keyword_map, keywords_list, total_files = process_files(input_dir, output_dir)

    save_files(keyword_map, keywords_list, total_files, output_dir)

    print(f"\nProcessing complete:")
    print(f"- Total files processed: {total_files}")
    print(f"- Total unique keywords: {len(keywords_list)}")
    print(f"- Output directory: {output_dir}")
    print(f"- Files created: keyword_map.json, keywords.txt, files.txt")


if __name__ == "__main__":
    main()
