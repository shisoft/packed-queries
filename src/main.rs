use kmeans::{KMeans, KMeansConfig, KMeansState};
use memchr::{memchr_iter, Memchr};
use memmap2::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Result;
use std::env;
use std::fs::File;
use std::io::{self, BufRead};

#[derive(Serialize, Deserialize, Debug)]
enum QueryObjective {
    Maximize(String),
    Minimize(String),
}

#[derive(Serialize, Deserialize, Debug)]
enum QueryConstrainComp {
    Less(f32),
    Greater(f32),
    LessEq(f32),
    GreatEq(f32),
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryConstrainExpr {
    attr: String,
    comp: QueryConstrainComp,
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryConstrainCount {
    comp: QueryConstrainComp,
    val: f32,
}

#[derive(Serialize, Deserialize, Debug)]
enum QueryConstrain {
    Sum(QueryConstrainExpr),
    Count(QueryConstrainCount),
}

#[derive(Serialize, Deserialize, Debug)]
struct Query {
    obj: QueryObjective,
    cons: Vec<QueryConstrain>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DataSet {
    rows: usize,
    num_cols: usize,
    buffer: Vec<f32>,
}

impl DataSet {
    fn new(buffer: Vec<f32>, num_cols: usize) -> Self {
        Self {
            rows: buffer.len() / num_cols,
            num_cols,
            buffer,
        }
    }
    fn row(&self, i: usize) -> &[f32] {
        let cols = self.num_cols;
        let start = i * cols;
        &self.buffer.as_slice()[start..start + cols]
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("Packet queries. Parameters {:?}", args);
    if args.len() < 2 {
        panic!("Must supply dataset file");
    }
    let dataset_file = &args[1];
    print!("Mapping dataset {} to memory", dataset_file);
    let mapped = unsafe {
        MmapOptions::new()
            .map(&File::open(dataset_file).unwrap())
            .unwrap()
    };
    println!("Searching for header");
    let mut lines = memchr_iter(b'\n', &mapped);
    let first_line = lines.next().unwrap();
    println!("Found first line at {}", first_line);
    let headers = read_line(&mapped, 0, first_line);
    println!("Found header {:?}", headers);
    println!("Read all data into memory for clustering");
    let num_cols = headers.len();
    let data = read_all_data(&mapped, lines, first_line, num_cols);
    println!(
        "Read total of {} row of data, preparing clustering data",
        data.rows
    );
    let k = 64;
    let clusters = clustering(&data, k);
    let centroids = clusters
        .centroids
        .chunks_exact(num_cols)
        .collect::<Vec<_>>();
    assert_eq!(
        centroids.len(),
        k,
        "Have {} centroids, expect {}",
        centroids.len(),
        k
    );
    println!("Query >");
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let str_line = line.unwrap();
        let query_res = serde_json::from_str::<Query>(&str_line);
        if let Ok(query) = query_res {
            println!("Accepting query {:?}", query);
        } else {
            println!("Cannot parse json query \"{}\"", str_line);
        }
        println!("Query >");
    }
}

fn clustering(data: &DataSet, k: usize) -> KMeansState<f32> {
    let iter = 100;
    let (sample_cnt, sample_dims, k, max_iter) = (data.rows, data.num_cols, k, iter);
    let data_clone = data.buffer.clone();
    // Calculate kmeans, using kmean++ as initialization-method
    println!(
        "Clustering with k: {}, iter: {}, dim {}",
        k, iter, data.num_cols
    );
    let kmean = KMeans::new(data_clone, sample_cnt, sample_dims);
    let result = kmean.kmeans_minibatch(
        1024,
        k,
        max_iter,
        KMeans::init_kmeanplusplus,
        &KMeansConfig::default(),
    );
    println!("Clustered...");
    result
}

fn read_line(mem: &Mmap, start: usize, ends: usize) -> Vec<&str> {
    let raw_bytes = &mem[start..ends];
    let line = std::str::from_utf8(raw_bytes).unwrap();
    // print!("Read line {}", line);
    line.split(",").collect()
}

fn str_line_to_num_line(line_str: Vec<&str>) -> Vec<f32> {
    line_str
        .iter()
        .map(|str| {
            str.trim()
                .parse()
                .expect(&format!("Error on parsing {}, line {:?}", str, line_str))
        })
        .collect()
}

fn read_all_data(mem: &Mmap, lines: Memchr, start: usize, num_cols: usize) -> DataSet {
    let mut rows = vec![];
    let mut start = start;
    // Linear scan to obtain positions
    for end in lines {
        rows.push((start, end));
        start = end;
    }
    // Parallel read from mapped file
    let buffer = rows
        .into_par_iter()
        .filter_map(|(start, end)| {
            let line_str = read_line(mem, start, end);
            if line_str.len() >= num_cols {
                // Skip invalid lines
                Some(str_line_to_num_line(line_str))
            } else {
                None
            }
        })
        .flatten()
        .collect();
    DataSet::new(buffer, num_cols)
}
