import random
import os
import json
import nltk

import multiprocessing as mp

from functools import partial

nltk.download("words")
nltk.download("punkt")
nltk.download("stopwords")

from nltk.corpus import words


def generate_file_content(file):
    with open(file, "w") as f:
        for _ in range(300 * 1024 // 50):
            f.write(
                "".join(
                    random.choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", k=50
                    )
                )
                + "\n"
            )


def generate_random_data(num_keywords, num_files, avg_files_per_keyword):
    all_words = words.words()
    keywords = random.sample(all_words, num_keywords)
    files = [f"{i}" for i in range(num_files)]

    with mp.Pool() as pool:
        pool.map(generate_file_content, files)

    keyword_file_map = {}
    for keyword in keywords:
        num_files_for_keyword = max(
            1, int(random.gauss(avg_files_per_keyword, avg_files_per_keyword / 4))
        )
        keyword_file_map[keyword] = random.choices(files, k=num_files_for_keyword)

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
    num_keywords = 512
    num_files = 12000
    avg_files_per_keyword = 128

    keywords, files, keyword_file_map = generate_random_data(
        num_keywords, num_files, avg_files_per_keyword
    )
    save_data(keywords, files, keyword_file_map)

    print(f"Number of keywords: {len(keywords)}")
    print(f"Number of files: {len(files)}")
    print(
        f"Average number of files per keyword: {sum(len(v) for v in keyword_file_map.values()) / len(keyword_file_map):.2f}"
    )
