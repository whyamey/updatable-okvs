import os
import argparse
import multiprocessing
import shutil
import statistics
import brotli
from tqdm import tqdm

META_FILES = {"keywords.txt", "keyword_file_map.json", "files.txt"}


def get_target_files(input_dir):
    file_paths = []
    for filename in os.listdir(input_dir):
        if filename not in META_FILES:
            full_path = os.path.join(input_dir, filename)
            if os.path.isfile(full_path):
                file_paths.append(full_path)
    return file_paths


def compress_single_file(args):
    input_path, output_dir = args
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    try:
        with open(input_path, "rb") as f_in:
            data = f_in.read()

        compressed_data = brotli.compress(data, quality=9)
        compressed_size = len(compressed_data)

        with open(output_path, "wb") as f_out:
            f_out.write(compressed_data)

        return (True, compressed_size)
    except Exception as e:
        return (False, f"Failed {filename}: {str(e)}")


def format_bytes(b):
    if b < 1024.0:
        return f"{b:.2f} B"
    return f"{b / 1024.0:.2f} KB"


def main():
    parser = argparse.ArgumentParser(description="Brotli compression.")
    parser.add_argument("input_dir", help="Directory containing the uncompressed files")
    parser.add_argument("output_dir", help="Directory to save the compressed files")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print("Copying metadata files...")
    for meta_file in META_FILES:
        src = os.path.join(args.input_dir, meta_file)
        dst = os.path.join(args.output_dir, meta_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {meta_file}")
        else:
            print(f"Warning: {meta_file} not found in input directory.")

    print(f"\nScanning '{args.input_dir}' for files to compress...")
    file_paths = get_target_files(args.input_dir)
    total_files = len(file_paths)

    if total_files == 0:
        print("No files found to compress.")
        return

    print(f"Found {total_files} files.")

    process_args = [(path, args.output_dir) for path in file_paths]
    num_workers = multiprocessing.cpu_count()
    print(f"Starting compression across {num_workers} CPU cores...\n")

    compressed_sizes = []
    failed_files = []

    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(compress_single_file, process_args, chunksize=50),
            total=total_files,
            desc="Compressing",
        ):
            success, val = result
            if success:
                compressed_sizes.append(val)
            else:
                failed_files.append(val)

    print("\n" + "=" * 50)
    print("COMPRESSION COMPLETE")
    print("=" * 50)
    print(f"Successfully processed: {len(compressed_sizes)} files")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")

    if compressed_sizes:
        avg_size = statistics.mean(compressed_sizes)
        med_size = statistics.median(compressed_sizes)
        print("-" * 50)
        print(
            f"Average Compressed Size : {avg_size:.2f} bytes ({format_bytes(avg_size)})"
        )
        print(
            f"Median Compressed Size  : {med_size:.2f} bytes ({format_bytes(med_size)})"
        )
        print("-" * 50)

    if failed_files:
        print(f"\nWarning: {len(failed_files)} files failed to compress.")
        for err in failed_files[:5]:
            print(f"  {err}")
        if len(failed_files) > 5:
            print("  ... and others.")


if __name__ == "__main__":
    main()
