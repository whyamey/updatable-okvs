import math
import os
import sys

p_val = 256
val_element_size = 32
val_bucket_size = 8192
vol_bucket_size = 256
doc_chunk_size = 1024

n = 1572
q = 32
rho = 9
tabulation = 0

def calculate_exact_dataset_metrics(data_dir, chunk_size=1024):
    total_chunks = 0
    if not os.path.exists(data_dir):
        return 0
    for filename in os.listdir(data_dir):
        if filename.isdigit():
            filepath = os.path.join(data_dir, filename)
            size = os.path.getsize(filepath)
            total_chunks += math.ceil(size / chunk_size)
    return total_chunks

def bits_to_kb(bits):
    return bits / (8 * 1024)

def bits_to_mb(bits):
    return bits / (8 * 1024 * 1024)

def bits_to_gb(bits):
    return bits / (8 * 1024 * 1024 * 1024)

def compute_batch_bandwidth(volbw, valquerybw, valresponse_bw, batch_size, shards=1):
    chunks = math.ceil(batch_size / p_val)
    num_bucket_fetches = 4 * chunks

    if tabulation:
        return (volbw * shards) + (4 * valquerybw) + math.ceil(valresponse_bw * num_bucket_fetches * shards)
    else:
        return (volbw * shards) + math.ceil((valquerybw + valresponse_bw) * num_bucket_fetches * shards)

def compute_storage(vol, val, stash, pair, shards=1):
    vol = vol / shards
    val = val / shards

    valm = math.ceil((val * 2.0) / p_val)
    volm = math.ceil(vol * 2.0)

    valomega = math.ceil(val_bucket_size * 8 / rho)
    volomega = math.ceil(vol_bucket_size * 8 / rho)

    valquery_bw = math.ceil(valm * q)
    volquery_bw = math.ceil(volm * q)

    valresponse_bw = math.ceil(valomega * q)
    volresponse_bw = math.ceil(volomega * q)

    volbw = volquery_bw + volresponse_bw

    storage = ((valomega * valm) + (volomega * volm)) * q

    bw = volbw + valquery_bw + valresponse_bw

    bw_100 = compute_batch_bandwidth(volbw, valquery_bw, valresponse_bw, 100, shards=shards)
    bw_500 = compute_batch_bandwidth(volbw, valquery_bw, valresponse_bw, 500, shards=shards)
    bw_1000 = compute_batch_bandwidth(volbw, valquery_bw, valresponse_bw, 1000, shards=shards)

    storage_kb = bits_to_kb(storage)
    storage_mb = bits_to_mb(storage)
    storage_gb = bits_to_gb(storage)

    bw_results = {
        'base': (bw, bits_to_kb(bw), bits_to_mb(bw)),
        '100': (bw_100, bits_to_kb(bw_100), bits_to_mb(bw_100)),
        '500': (bw_500, bits_to_kb(bw_500), bits_to_mb(bw_500)),
        '1000': (bw_1000, bits_to_kb(bw_1000), bits_to_mb(bw_1000))
    }

    return (storage, storage_kb, storage_mb, storage_gb, bw_results)

if __name__ == '__main__':
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    exact_total_chunks = calculate_exact_dataset_metrics(data_dir)

    size = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

    gvol = [18831, 18831, 18831, 18831, 18831, 18831, 18831, 18831, 18831, 18831]
    gval = [65536, 131054, 262092, 524203, 1048491, 2096664, 4192389, 8385572, 16774950, 24436793]
    gstash = [0, 0, 0, 0, 0, 0, 0, 0, 3, 4]
    gpairave = [j / i for i, j in zip(gvol, gval)]

    print("\nResults for Enron dataset:")
    print("-" * 100)
    for i in range(len(gval)):
        storage, storage_kb, storage_mb, storage_gb, bw_results = compute_storage(gvol[i], gval[i], gstash[i], gpairave[i])
        
        print(f"\nSize 2^{size[i]}:")
        print(f"Storage: {storage:,} bits")
        print(f"        = {storage_kb:.2f} KB")
        print(f"        = {storage_mb:.2f} MB")
        print(f"        = {storage_gb:.2f} GB")

        print("\nBandwidth:")
        for batch_size, (bw, bw_kb, bw_mb) in bw_results.items():
            label = "Base" if batch_size == 'base' else f"Batch {batch_size}"
            print(f"{label:>10}: {bw:,} bits")
            print(f"{' ':10}= {bw_kb:.2f} KB")
            print(f"{' ':10}= {bw_mb:.2f} MB")

        if i == len(gval) - 1 and exact_total_chunks > 0:
            docomega = math.ceil(doc_chunk_size * 8 / rho)
            docmap_storage = (docomega * exact_total_chunks) * q
            total_with_docmap = storage + docmap_storage
            print(f"\nDocMap Included: {total_with_docmap:,} bits ({bits_to_gb(total_with_docmap):.2f} GB)")
