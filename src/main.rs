use memmap2::*;
use std::env;
use std::fs::File;
use memchr::{Memchr, memchr_iter};
use rayon::prelude::*;

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
    print!("Found header {:?}", headers);
    print!("Read all data into memory for clustering");
    let data = read_all_data(&mapped, lines, first_line, headers.len());
    print!("Read total of {} row of data", data.len());
}

fn read_line(mem: &Mmap, start: usize, ends: usize) -> Vec<&str> {
    let raw_bytes = &mem[start..ends];
    let line = std::str::from_utf8(raw_bytes).unwrap();
    // print!("Read line {}", line);
    line.split(",").collect()
}

fn str_line_to_num_line(line_str: Vec<&str>) -> Vec<f32> {
    line_str.iter().map(|str| str.trim().parse().expect(&format!("Error on parsing {}, line {:?}", str, line_str))).collect()
} 

fn read_all_data(mem: &Mmap, lines: Memchr, start: usize, num_cols: usize) -> Vec<Vec<f32>> {
    let mut rows = vec![];
    let mut start = start;
    // Linear scan to obtain positions
    for end in lines {
        rows.push((start, end));
        start = end;
    }
    // Parallel read from mapped file
    rows
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
    .collect()
}