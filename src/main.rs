use chrono::Local;
use log::{error, LevelFilter};
use log4rs::append::console::ConsoleAppender;
use log4rs::append::file::FileAppender;
use log4rs::config::{Appender, Config, Root};
use log4rs::encode::pattern::PatternEncoder;
use log4rs::filter::threshold::ThresholdFilter;
use rand::{thread_rng, Rng};
use std::env;
use std::fs;

mod bench;
mod dip;
mod generate_mmap;

use bench::Benchmarker;

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut keywords_file: Option<String> = None;
    let mut data_dir: String = "data".to_string();
    let mut log_level = LevelFilter::Info;
    let mut use_bfs = false;
    let mut run_plaintext_retrieval = false;
    let mut run_docmap_test = false;
    let mut run_ddr_test = false;
    let mut run_insert_test = false;
    let mut insert_split = 0.8;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--read-keywords" => {
                if i + 1 < args.len() {
                    keywords_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    error!("Missing filename after --read-keywords");
                    return Err("Missing filename after --read-keywords".into());
                }
            }
            "--data-dir" => {
                if i + 1 < args.len() {
                    data_dir = args[i + 1].clone();
                    i += 2;
                } else {
                    error!("Missing directory after --data-dir");
                    return Err("Missing directory after --data-dir".into());
                }
            }
            "--log-trace" => {
                log_level = LevelFilter::Trace;
                i += 1;
            }
            "--use-bfs" => {
                use_bfs = true;
                i += 1;
            }
            "--run-plaintext-retrieval" => {
                run_plaintext_retrieval = true;
                i += 1;
            }
            "--run-docmap-test" => {
                run_docmap_test = true;
                i += 1;
            }
            "--run-ddr-test" => {
                run_ddr_test = true;
                i += 1;
            }
            "--run-insert-test" => {
                run_insert_test = true;
                i += 1;
            }
            "--insert-split" => {
                if i + 1 < args.len() {
                    insert_split = args[i + 1].parse().unwrap_or(0.8);
                    i += 2;
                }
            }
            _ => i += 1,
        }
    }

    let stdout = ConsoleAppender::builder()
        .encoder(Box::new(PatternEncoder::new(
            "{d(%Y-%m-%d %H:%M:%S)} {h({l})} - {m}\n",
        )))
        .build();

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let random_suffix: u32 = thread_rng().gen_range(1000..9999);
    let log_file_name = format!("log/output_{timestamp}_{random_suffix}.log");

    fs::create_dir_all("log")?;
    let file = FileAppender::builder()
        .encoder(Box::new(PatternEncoder::new("{d} - {l} - {m}\n")))
        .build(&log_file_name)?;

    let config = Config::builder()
        .appender(
            Appender::builder()
                .filter(Box::new(ThresholdFilter::new(log_level)))
                .build("stdout", Box::new(stdout)),
        )
        .appender(Appender::builder().build("file", Box::new(file)))
        .build(
            Root::builder()
                .appender("stdout")
                .appender("file")
                .build(log_level),
        )?;

    log4rs::init_config(config)?;

    let p = 256;
    let epsilon = 0.2;
    let run_decryption_test = false;

    let mut benchmarker = Benchmarker::new(
        p,
        epsilon,
        run_decryption_test,
        keywords_file,
        data_dir,
        use_bfs,
        run_plaintext_retrieval,
        run_docmap_test,
        run_ddr_test,
        run_insert_test,
        insert_split,
    )?;

    if let Err(e) = benchmarker.run_benchmarks() {
        error!("Error running benchmarks: {}", e);
    }

    log::logger().flush();

    Ok(())
}
