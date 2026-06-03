import json
import argparse
import re


def parse_size(size_str):
    power_pattern = re.compile(r"2\^(\d+)")
    match = power_pattern.match(size_str)
    return 2 ** int(match.group(1)) if match else int(size_str)


def generate_query_keywords(input_path, output_path, target_doc_count, num_keywords=1):
    with open(input_path) as f:
        full_map = json.load(f)

    keyword_counts = [(k, len(docs)) for k, docs in full_map.items()]

    margin = target_doc_count * 0.1
    eligible_keywords = [
        k for k, count in keyword_counts if abs(count - target_doc_count) <= margin
    ]

    if len(eligible_keywords) < num_keywords:
        keyword_counts.sort(key=lambda x: abs(x[1] - target_doc_count))
        eligible_keywords = [k for k, _ in keyword_counts[:num_keywords]]
        print(
            f"Warning: Could not find enough keywords with approximately {target_doc_count} documents (±10%)."
        )
        print("Selected closest matching keywords instead:")
        for k in eligible_keywords:
            print(f"  {k}: {len(full_map[k])} documents")

    selected = eligible_keywords[:num_keywords]

    with open(output_path, "w") as f:
        for keyword in selected:
            f.write(f"{keyword}\n")

    return selected


def create_subset_map(input_path, output_path, target_size, keywords_to_keep):
    with open(input_path) as f:
        full_map = json.load(f)

    subset_map = {k: full_map[k] for k in keywords_to_keep if k in full_map}
    if len(subset_map) != len(keywords_to_keep):
        missing = set(keywords_to_keep) - set(subset_map.keys())
        raise ValueError(
            f"Could not find all required keywords in input map. Missing: {missing}"
        )

    preserved_size = sum(len(docs) for docs in subset_map.values())

    remaining_items = [(k, full_map[k]) for k in full_map if k not in keywords_to_keep]
    remaining_items.sort(key=lambda x: len(x[1]))

    current_size = preserved_size
    for keyword, doc_list in remaining_items:
        if current_size + len(doc_list) > target_size:
            continue
        subset_map[keyword] = doc_list
        current_size += len(doc_list)

    with open(output_path, "w") as f:
        json.dump(subset_map, f)

    return subset_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", help="Input JSON file containing the full keyword map"
    )
    parser.add_argument(
        "--output-file",
        help="Output file for the subset map (required when using --load-keywords)",
    )
    parser.add_argument("--target-size", type=parse_size)
    parser.add_argument(
        "--generate-keywords",
        nargs=2,
        metavar=("DOC_COUNT", "NUM_KEYWORDS"),
        help="Generate keywords with specified document count. Format: DOC_COUNT NUM_KEYWORDS",
    )
    parser.add_argument("--load-keywords", type=str)
    parser.add_argument("--keywords-output", type=str, default="keywords_to_query")

    args = parser.parse_args()

    if args.generate_keywords:
        doc_count = int(args.generate_keywords[0])
        num_keywords = int(args.generate_keywords[1])
        generate_query_keywords(
            args.input_file, args.keywords_output, doc_count, num_keywords
        )
    elif args.load_keywords:
        if not args.output_file:
            parser.error("--output-file is required when using --load-keywords")
        if not args.target_size:
            parser.error("--target-size is required when using --load-keywords")
        with open(args.load_keywords) as f:
            keywords = [line.split("\t")[0].strip() for line in f]
        create_subset_map(args.input_file, args.output_file, args.target_size, keywords)


if __name__ == "__main__":
    main()
