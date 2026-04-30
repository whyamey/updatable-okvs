.PHONY: all clean sharding subsets test-subset-100 test-subset-500 test-subset-1000 generate_keyword_files sharding-with-keywords test-2-shards-keywords test-4-shards-keywords test-8-shards-keywords test-16-shards-keywords test-no-shard-keywords

RUSTFLAGS = '-C target-cpu=native'
DATA_DIR ?= data
USE_EXISTING ?= false
SCRIPTS_DIR ?= data
OUTPUT_DIR = output
LOGS_DIR = $(OUTPUT_DIR)/logs
SHARDS_DIR = $(OUTPUT_DIR)/shards

all: sharding sharding-with-keywords subsets
sharding: setup-dirs target/release/mkse test-2-shards test-4-shards test-8-shards test-16-shards test-no-shard
subsets: setup-dirs target/release/mkse test-subset-100 test-subset-500 test-subset-1000
sharding-with-keywords: setup-dirs target/release/mkse generate_shard_keywords test-16-shards-keywords test-8-shards-keywords test-4-shards-keywords test-2-shards-keywords test-no-shard-keywords

setup-dirs:
	mkdir -p $(OUTPUT_DIR)
	mkdir -p $(LOGS_DIR)
	mkdir -p $(SHARDS_DIR)

target/release/mkse:
	RUSTFLAGS=$(RUSTFLAGS) cargo build --release

generate_data: setup-dirs
	@if [ "$(USE_EXISTING)" = "false" ]; then \
		echo "Generating new data..."; \
		python $(SCRIPTS_DIR)/generate_power.py; \
	else \
		echo "Using existing data..."; \
	fi

generate_initial_shard: target/release/mkse generate_data
	mkdir -p $(SHARDS_DIR)/shard_2
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_2 2
	cp $(SHARDS_DIR)/shard_2/keywords_to_query $(OUTPUT_DIR)/keywords_to_query
	# Verify the file exists
	@test -f $(OUTPUT_DIR)/keywords_to_query || (echo "Failed to create keywords_to_query file" && exit 1)

test-2-shards: generate_initial_shard
	@test -f $(OUTPUT_DIR)/keywords_to_query || (echo "keywords_to_query file missing" && exit 1)
	for i in 0 1; do \
		mkdir -p $(SHARDS_DIR)/shard_2/data_$$i; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_2/data_$$i/; \
		cp $(SHARDS_DIR)/shard_2/files_$$i.txt $(SHARDS_DIR)/shard_2/data_$$i/files.txt; \
		cp $(SHARDS_DIR)/shard_2/keywords_$$i.txt $(SHARDS_DIR)/shard_2/data_$$i/keywords.txt; \
		cp $(SHARDS_DIR)/shard_2/shard_$$i.json $(SHARDS_DIR)/shard_2/data_$$i/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query $(SHARDS_DIR)/shard_2/data_$$i/; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/shard_2/data_$$i --read-keywords $(SHARDS_DIR)/shard_2/data_$$i/keywords_to_query > $(LOGS_DIR)/output_2_$$i.txt 2>&1; \
	done
	rm -rf $(SHARDS_DIR)/shard_2

test-4-shards: generate_initial_shard
	mkdir -p $(SHARDS_DIR)/shard_4
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_4 4
	for i in 0 1 2 3; do \
		mkdir -p $(SHARDS_DIR)/shard_4/data_$$i; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_4/data_$$i/; \
		cp $(SHARDS_DIR)/shard_4/files_$$i.txt $(SHARDS_DIR)/shard_4/data_$$i/files.txt; \
		cp $(SHARDS_DIR)/shard_4/keywords_$$i.txt $(SHARDS_DIR)/shard_4/data_$$i/keywords.txt; \
		cp $(SHARDS_DIR)/shard_4/shard_$$i.json $(SHARDS_DIR)/shard_4/data_$$i/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query $(SHARDS_DIR)/shard_4/data_$$i/; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/shard_4/data_$$i --read-keywords $(SHARDS_DIR)/shard_4/data_$$i/keywords_to_query > $(LOGS_DIR)/output_4_$$i.txt 2>&1; \
	done
	rm -rf $(SHARDS_DIR)/shard_4

test-8-shards: generate_initial_shard
	mkdir -p $(SHARDS_DIR)/shard_8
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_8 8
	for i in 0 1 2 3 4 5 6 7; do \
		mkdir -p $(SHARDS_DIR)/shard_8/data_$$i; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_8/data_$$i/; \
		cp $(SHARDS_DIR)/shard_8/files_$$i.txt $(SHARDS_DIR)/shard_8/data_$$i/files.txt; \
		cp $(SHARDS_DIR)/shard_8/keywords_$$i.txt $(SHARDS_DIR)/shard_8/data_$$i/keywords.txt; \
		cp $(SHARDS_DIR)/shard_8/shard_$$i.json $(SHARDS_DIR)/shard_8/data_$$i/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query $(SHARDS_DIR)/shard_8/data_$$i/; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/shard_8/data_$$i --read-keywords $(SHARDS_DIR)/shard_8/data_$$i/keywords_to_query > $(LOGS_DIR)/output_8_$$i.txt 2>&1; \
	done
	rm -rf $(SHARDS_DIR)/shard_8

test-16-shards: generate_initial_shard
	mkdir -p $(SHARDS_DIR)/shard_16
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_16 16
	for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		mkdir -p $(SHARDS_DIR)/shard_16/data_$$i; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_16/data_$$i/; \
		cp $(SHARDS_DIR)/shard_16/files_$$i.txt $(SHARDS_DIR)/shard_16/data_$$i/files.txt; \
		cp $(SHARDS_DIR)/shard_16/keywords_$$i.txt $(SHARDS_DIR)/shard_16/data_$$i/keywords.txt; \
		cp $(SHARDS_DIR)/shard_16/shard_$$i.json $(SHARDS_DIR)/shard_16/data_$$i/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query $(SHARDS_DIR)/shard_16/data_$$i/; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/shard_16/data_$$i --read-keywords $(SHARDS_DIR)/shard_16/data_$$i/keywords_to_query > $(LOGS_DIR)/output_16_$$i.txt 2>&1; \
	done
	rm -rf $(SHARDS_DIR)/shard_16

test-no-shard: generate_initial_shard
	./target/release/mkse --data-dir $(DATA_DIR) --read-keywords $(OUTPUT_DIR)/keywords_to_query > $(LOGS_DIR)/output_no_shard.txt 2>&1

generate_shard_keywords: setup-dirs generate_data
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 100 1 --keywords-output $(OUTPUT_DIR)/shard_keywords_100
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 500 1 --keywords-output $(OUTPUT_DIR)/shard_keywords_500
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 1000 1 --keywords-output $(OUTPUT_DIR)/shard_keywords_1000

test-2-shards-keywords:
	mkdir -p $(SHARDS_DIR)/shard_2
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_2 2 --sample_size 0
	for i in 0 1; do \
		mkdir -p $(SHARDS_DIR)/shard_2/data_$${i}; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_2/data_$${i}/; \
		cp $(SHARDS_DIR)/shard_2/files_$${i}.txt $(SHARDS_DIR)/shard_2/data_$${i}/files.txt; \
		cp $(SHARDS_DIR)/shard_2/keywords_$${i}.txt $(SHARDS_DIR)/shard_2/data_$${i}/keywords.txt; \
		cp $(SHARDS_DIR)/shard_2/shard_$${i}.json $(SHARDS_DIR)/shard_2/data_$${i}/keyword_file_map.json; \
		for freq in 100 500 1000; do \
			cp $(OUTPUT_DIR)/shard_keywords_$${freq} $(SHARDS_DIR)/shard_2/data_$${i}/keywords_to_query; \
			./target/release/mkse --data-dir $(SHARDS_DIR)/shard_2/data_$${i} \
				--read-keywords $(SHARDS_DIR)/shard_2/data_$${i}/keywords_to_query \
				> $(LOGS_DIR)/output_2_$${i}_freq_$${freq}.txt 2>&1; \
		done; \
	done
	rm -rf $(SHARDS_DIR)/shard_2

test-4-shards-keywords:
	mkdir -p $(SHARDS_DIR)/shard_4
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_4 4 --sample_size 0
	for i in 0 1 2 3; do \
		mkdir -p $(SHARDS_DIR)/shard_4/data_$${i}; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_4/data_$${i}/; \
		cp $(SHARDS_DIR)/shard_4/files_$${i}.txt $(SHARDS_DIR)/shard_4/data_$${i}/files.txt; \
		cp $(SHARDS_DIR)/shard_4/keywords_$${i}.txt $(SHARDS_DIR)/shard_4/data_$${i}/keywords.txt; \
		cp $(SHARDS_DIR)/shard_4/shard_$${i}.json $(SHARDS_DIR)/shard_4/data_$${i}/keyword_file_map.json; \
		for freq in 100 500 1000; do \
			cp $(OUTPUT_DIR)/shard_keywords_$${freq} $(SHARDS_DIR)/shard_4/data_$${i}/keywords_to_query; \
			./target/release/mkse --data-dir $(SHARDS_DIR)/shard_4/data_$${i} \
				--read-keywords $(SHARDS_DIR)/shard_4/data_$${i}/keywords_to_query \
				> $(LOGS_DIR)/output_4_$${i}_freq_$${freq}.txt 2>&1; \
		done; \
	done
	rm -rf $(SHARDS_DIR)/shard_4

test-8-shards-keywords:
	mkdir -p $(SHARDS_DIR)/shard_8
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_8 8 --sample_size 0
	for i in 0 1 2 3 4 5 6 7; do \
		mkdir -p $(SHARDS_DIR)/shard_8/data_$${i}; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_8/data_$${i}/; \
		cp $(SHARDS_DIR)/shard_8/files_$${i}.txt $(SHARDS_DIR)/shard_8/data_$${i}/files.txt; \
		cp $(SHARDS_DIR)/shard_8/keywords_$${i}.txt $(SHARDS_DIR)/shard_8/data_$${i}/keywords.txt; \
		cp $(SHARDS_DIR)/shard_8/shard_$${i}.json $(SHARDS_DIR)/shard_8/data_$${i}/keyword_file_map.json; \
		for freq in 100 500 1000; do \
			cp $(OUTPUT_DIR)/shard_keywords_$${freq} $(SHARDS_DIR)/shard_8/data_$${i}/keywords_to_query; \
			./target/release/mkse --data-dir $(SHARDS_DIR)/shard_8/data_$${i} \
				--read-keywords $(SHARDS_DIR)/shard_8/data_$${i}/keywords_to_query \
				> $(LOGS_DIR)/output_8_$${i}_freq_$${freq}.txt 2>&1; \
		done; \
	done
	rm -rf $(SHARDS_DIR)/shard_8

test-16-shards-keywords:
	mkdir -p $(SHARDS_DIR)/shard_16
	python $(SCRIPTS_DIR)/create_shards.py $(DATA_DIR)/keyword_file_map.json $(SHARDS_DIR)/shard_16 16 --sample_size 0
	for i in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do \
		mkdir -p $(SHARDS_DIR)/shard_16/data_$${i}; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/shard_16/data_$${i}/; \
		cp $(SHARDS_DIR)/shard_16/files_$${i}.txt $(SHARDS_DIR)/shard_16/data_$${i}/files.txt; \
		cp $(SHARDS_DIR)/shard_16/keywords_$${i}.txt $(SHARDS_DIR)/shard_16/data_$${i}/keywords.txt; \
		cp $(SHARDS_DIR)/shard_16/shard_$${i}.json $(SHARDS_DIR)/shard_16/data_$${i}/keyword_file_map.json; \
		for freq in 100 500 1000; do \
			cp $(OUTPUT_DIR)/shard_keywords_$${freq} $(SHARDS_DIR)/shard_16/data_$${i}/keywords_to_query; \
			./target/release/mkse --data-dir $(SHARDS_DIR)/shard_16/data_$${i} \
				--read-keywords $(SHARDS_DIR)/shard_16/data_$${i}/keywords_to_query \
				> $(LOGS_DIR)/output_16_$${i}_freq_$${freq}.txt 2>&1; \
		done; \
	done
	rm -rf $(SHARDS_DIR)/shard_16

test-no-shard-keywords: generate_shard_keywords
	for freq in 100 500 1000; do \
		cp $(OUTPUT_DIR)/shard_keywords_$${freq} $(DATA_DIR)/keywords_to_query; \
		./target/release/mkse --data-dir $(DATA_DIR) \
			--read-keywords $(DATA_DIR)/keywords_to_query \
			> $(LOGS_DIR)/output_no_shard_freq_$${freq}.txt 2>&1; \
	done

generate_keyword_files: setup-dirs generate_data
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 100 1 --keywords-output $(OUTPUT_DIR)/keywords_to_query_100
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 500 1 --keywords-output $(OUTPUT_DIR)/keywords_to_query_500
	python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json --generate-keywords 1000 1 --keywords-output $(OUTPUT_DIR)/keywords_to_query_1000

test-subset-100: target/release/mkse generate_keyword_files
	for size in 16 17 18 19 20 21 22 23 24; do \
		mkdir -p $(SHARDS_DIR)/subset_100_2\^$$size; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_100_2\^$$size/; \
		python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json \
			--load-keywords $(OUTPUT_DIR)/keywords_to_query_100 \
			--target-size 2\^$$size \
			--output-file $(SHARDS_DIR)/subset_100_2\^$$size/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query_100 $(SHARDS_DIR)/subset_100_2\^$$size/keywords_to_query; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/subset_100_2\^$$size \
			--read-keywords $(SHARDS_DIR)/subset_100_2\^$$size/keywords_to_query > $(LOGS_DIR)/output_subset_100_2\^$$size.txt 2>&1; \
		rm -rf $(SHARDS_DIR)/subset_100_2\^$$size; \
	done; \
	mkdir -p $(SHARDS_DIR)/subset_100_full; \
	rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_100_full/; \
	cp $(OUTPUT_DIR)/keywords_to_query_100 $(SHARDS_DIR)/subset_100_full/keywords_to_query; \
	./target/release/mkse --data-dir $(SHARDS_DIR)/subset_100_full \
		--read-keywords $(SHARDS_DIR)/subset_100_full/keywords_to_query > $(LOGS_DIR)/output_subset_100_full.txt 2>&1; \
	rm -rf $(SHARDS_DIR)/subset_100_full

test-subset-500: target/release/mkse generate_keyword_files
	for size in 16 17 18 19 20 21 22 23 24; do \
		mkdir -p $(SHARDS_DIR)/subset_500_2\^$$size; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_500_2\^$$size/; \
		python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json \
			--load-keywords $(OUTPUT_DIR)/keywords_to_query_500 \
			--target-size 2\^$$size \
			--output-file $(SHARDS_DIR)/subset_500_2\^$$size/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query_500 $(SHARDS_DIR)/subset_500_2\^$$size/keywords_to_query; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/subset_500_2\^$$size \
			--read-keywords $(SHARDS_DIR)/subset_500_2\^$$size/keywords_to_query > $(LOGS_DIR)/output_subset_500_2\^$$size.txt 2>&1; \
		rm -rf $(SHARDS_DIR)/subset_500_2\^$$size; \
	done; \
	mkdir -p $(SHARDS_DIR)/subset_500_full; \
	rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_500_full/; \
	cp $(OUTPUT_DIR)/keywords_to_query_500 $(SHARDS_DIR)/subset_500_full/keywords_to_query; \
	./target/release/mkse --data-dir $(SHARDS_DIR)/subset_500_full \
		--read-keywords $(SHARDS_DIR)/subset_500_full/keywords_to_query > $(LOGS_DIR)/output_subset_500_full.txt 2>&1; \
	rm -rf $(SHARDS_DIR)/subset_500_full

test-subset-1000: target/release/mkse generate_keyword_files
	for size in 16 17 18 19 20 21 22 23 24; do \
		mkdir -p $(SHARDS_DIR)/subset_1000_2\^$$size; \
		rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_1000_2\^$$size/; \
		python $(SCRIPTS_DIR)/parse_subset.py $(DATA_DIR)/keyword_file_map.json \
			--load-keywords $(OUTPUT_DIR)/keywords_to_query_1000 \
			--target-size 2\^$$size \
			--output-file $(SHARDS_DIR)/subset_1000_2\^$$size/keyword_file_map.json; \
		cp $(OUTPUT_DIR)/keywords_to_query_1000 $(SHARDS_DIR)/subset_1000_2\^$$size/keywords_to_query; \
		./target/release/mkse --data-dir $(SHARDS_DIR)/subset_1000_2\^$$size \
			--read-keywords $(SHARDS_DIR)/subset_1000_2\^$$size/keywords_to_query > $(LOGS_DIR)/output_subset_1000_2\^$$size.txt 2>&1; \
		rm -rf $(SHARDS_DIR)/subset_1000_2\^$$size; \
	done; \
	mkdir -p $(SHARDS_DIR)/subset_1000_full; \
	rsync -a $(DATA_DIR)/ $(SHARDS_DIR)/subset_1000_full/; \
	cp $(OUTPUT_DIR)/keywords_to_query_1000 $(SHARDS_DIR)/subset_1000_full/keywords_to_query; \
	./target/release/mkse --data-dir $(SHARDS_DIR)/subset_1000_full \
		--read-keywords $(SHARDS_DIR)/subset_1000_full/keywords_to_query > $(LOGS_DIR)/output_subset_1000_full.txt 2>&1; \
	rm -rf $(SHARDS_DIR)/subset_1000_full

clean:
	rm -rf $(OUTPUT_DIR)
	cargo clean
