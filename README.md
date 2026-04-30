# MARS - Multi Adversarial Owner and Reader Replication-free Searchable Encryption

MARS is a system for multi-key searchable encryption that allows multiple data owners to store and share encrypted multimaps with clients while maintaining security against arbitrary collusion. It uses unkeyed PIR augmented with flexible-sharing mechanisms for efficient searching without data replication.

## Warning

⚠️ **Research Use Only**

This implementation was built specifically to produce data for our research paper. While it's a decent implementation for other researchers to use in their academic work, it is **not** recommended for production use. The code has not been audited for security vulnerabilities and may not implement all necessary safeguards for real-world applications.

If you plan to do searchable encryption in a production environment, you should seek or create a properly audited implementation. If you create a production-ready version of this code and can verify its security, please let me know and I'll be happy to link to your repository.

## Features

- Unkeyed PIR with flexible-sharing mechanisms.
- Support for large-scale multimaps (tested up to 24M entries).
- Query time of ~1s for keywords matching 100 documents. Scales linearly with the number of shards.
- Storage overhead factor of ~6x.

## Requirements

### Rust
1. Install the latest stable version of Rust and Cargo:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # On Unix/macOS
   # OR visit https://rustup.rs/ for Windows installation
   ```

### Python Dependencies
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Unix/macOS
   env\Scripts\activate     # On Windows
   ```

2. Install required packages:
   ```bash
   pip install nltk numpy tqdm
   ```

3. Download required NLTK data:
   ```python
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('words')"
   ```

## Getting Started

1. Generate test data:
   ```bash
   cd data
   python generate_power.py  # For power-law distributed synthetic data
   # OR
   python parse_enron.py    # For processing Enron dataset
   ```

2. Build and run:
   ```bash
   cargo build --release
   ./target/release/mkse --data-dir data
   ```

## Running Tests

The project includes several test scenarios:

```bash
# Run all tests
DATA_DIR=<data> USE_EXISTING=false make all

# Run sharding tests
DATA_DIR=<data> USE_EXISTING=true make sharding

# Run subset tests
DATA_DIR=<data> USE_EXISTING=true make subsets

# Run sharding tests with keywords linked to $|$Docs_w$|$ documents
DATA_DIR=<data> USE_EXISTING=true make sharding-with-keywords
```

## Configuration

Command-line arguments:
- `--data-dir`: Specify the data directory
- `--read-keywords`: Provide a file containing keywords to query
- `--log-trace`: Enable detailed logging

## License

GPLv3.
