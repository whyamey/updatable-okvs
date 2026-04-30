import os
import json
import nltk
import random

import numpy as np
import multiprocessing as mp

nltk.download("words")
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import words


def power_law_distribution(min_value, max_value, alpha, size):
    """Generate a power-law distribution."""
    x = np.random.power(alpha, size)
    return (x * (max_value - min_value) + min_value).astype(int)


def create_file(file_name):
    with open(file_name, "w") as f:
        for _ in range(300 * 1024 // 50):
            f.write(
                "".join(
                    random.choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", k=50
                    )
                )
                + "\n"
            )


def generate_random_data(num_keywords, num_files, alpha):
    all_words = words.words()

    keywords = random.sample(all_words, num_keywords)
    files = [f"{i}" for i in range(num_files)]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(create_file, files)

    files_per_keyword = power_law_distribution(1, num_files // 10, alpha, num_keywords)

    keyword_file_map = {}
    for keyword, num_files_for_keyword in zip(keywords, files_per_keyword):
        keyword_file_map[keyword] = random.sample(
            files, min(num_files_for_keyword, len(files))
        )

    return keywords, files, keyword_file_map


def save_data(keywords, files, keyword_file_map):
    with open("keywords.txt", "w") as f:
        for keyword in keywords:
            f.write(keyword + "\n")

    with open("files.txt", "w") as f:
        for file in files:
            f.write(file + "\n")

    with open("keyword_file_map.json", "w") as f:
        json.dump(keyword_file_map, f, indent=4)


if __name__ == "__main__":
    num_keywords = 128
    num_files = 12000
    alpha = 2.5

    keywords, files, keyword_file_map = generate_random_data(
        num_keywords, num_files, alpha
    )
    save_data(keywords, files, keyword_file_map)

    total_associations = sum(len(v) for v in keyword_file_map.values())
    avg_files_per_keyword = total_associations / len(keyword_file_map)

    print(f"Number of keywords: {len(keywords)}")
    print(f"Number of files: {len(files)}")
    print(f"Average number of files per keyword: {avg_files_per_keyword:.2f}")
    print(f"Max files for a keyword: {max(len(v) for v in keyword_file_map.values())}")
    print(f"Min files for a keyword: {min(len(v) for v in keyword_file_map.values())}")
    print(
        f"Total size of the dataset (total keyword-file associations): {total_associations}"
    )
