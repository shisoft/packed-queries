use itertools::Itertools;
use json5;
use kmeans::{KMeans, KMeansConfig, KMeansState};
use memchr::{memchr_iter, Memchr};
use memmap2::*;
use minilp::{ComparisonOp, LinearExpr, OptimizationDirection, Problem, Solution, Variable};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::io::BufRead;

#[derive(Serialize, Deserialize, Debug)]
enum QueryObjective {
    Maximize(String),
    Minimize(String),
}

#[derive(Serialize, Deserialize, Debug)]
enum QueryConstrainComp {
    // Does not support lesser and greater
    Eq(f32),
    LessEq(f32),
    GreatEq(f32),
}

#[derive(Serialize, Deserialize, Debug)]
struct QueryConstrainExpr {
    attr: String,
    comp: QueryConstrainComp,
}

#[derive(Serialize, Deserialize, Debug)]
enum QueryConstrain {
    Sum(QueryConstrainExpr),
    Count(QueryConstrainComp),
}

#[derive(Serialize, Deserialize, Debug)]
struct Query {
    obj: QueryObjective,
    cons: Vec<QueryConstrain>,
    repeat_0: bool,
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
    let header_index = headers
        .iter()
        .enumerate()
        .map(|(i, h)| (h.to_string(), i))
        .collect::<HashMap<_, _>>();
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
    assert_eq!(clusters.assignments.len(), data.rows);
    let cluster_rows = clusters
        .assignments
        .iter()
        .enumerate()
        .map(|(i, c)| (c, i))
        .group_by(|(c, _)| **c)
        .into_iter()
        .map(|(c, i)| (c, i.map(|(_, i)| i).collect::<Vec<_>>()))
        .collect::<HashMap<_, _>>();
    let mut query_id = 1;
    println!("Query [{}] >", query_id);
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let str_line = line.unwrap();
        let query_res = json5::from_str::<Query>(&str_line);
        match query_res {
            Ok(query) => {
                println!("Accepting query {:?}", query);
                run_query(&data, &query, &centroids, &header_index, &cluster_rows);
            }
            Err(e) => {
                println!("Cannot parse json query \"{}\", reason: {:?}", str_line, e);
            }
        }
        query_id += 1;
        println!("Query [{}] >", query_id);
    }
}

fn run_query(
    data: &DataSet,
    query: &Query,
    centroids: &Vec<&[f32]>,
    headers: &HashMap<String, usize>,
    cluster_rows: &HashMap<usize, Vec<usize>>,
) {
    if let Some((solution, _vars)) = sketch(query, centroids, headers) {
        let mut rand = thread_rng();
        let selected_variable = solution
            .iter()
            .enumerate()
            .filter(|(_i, (_var, val))| **val >= 1.0)
            .collect::<Vec<_>>();
        let mut candidates = selected_variable
            .iter()
            .map(|(cluster, (var, _))| {
                (
                    cluster,
                    var,
                    cluster_rows[cluster]
                        .iter()
                        .map(|row_id| data.row(*row_id))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        candidates.shuffle(&mut rand);
        let mut result_set = vec![];
        while let Some((cluster_id, _var, picked_rows)) = candidates.pop() {
            if let Some((solution, vars)) = refine(
                query,
                centroids,
                &candidates,
                *cluster_id,
                &picked_rows,
                &result_set,
                headers,
            ) {
                // All the variables == 0.0, backtrack, do cluster_id first
                if solution.iter().all(|(_, val)| *val == 0.0) {
                    panic!("BACKTRACK");
                }
                if let Some((var, _val)) = solution
                    .iter()
                    .filter(|(_var, val)| **val >= 1.0)
                    .collect::<Vec<_>>()
                    .get(0)
                {
                    let picked_row_id = vars[var];
                    let picked_row = picked_rows[picked_row_id];
                    result_set.push((picked_row_id, picked_row));
                } else {
                    break;
                }
            } else {
                return;
            }
        }
        print!("Result: {} rows", result_set.len());
        headers.iter().for_each(|(s, _)| print!("|{}\t", s));
        headers.iter().for_each(|_| println!("|"));
        headers.iter().for_each(|_| print!("+\t"));
        headers.iter().for_each(|_| println!("+"));
        result_set.iter().for_each(|(_row_id, row)| {
            row.iter().for_each(|val| {
                print!("|{}\t", val);
            });
            println!("|");
        });
    }
}

fn refine(
    query: &Query,
    centroids: &Vec<&[f32]>,
    candidates: &Vec<(&usize, &Variable, Vec<&[f32]>)>,
    _picked_cluster: usize,
    picked_rows: &Vec<&[f32]>,
    result_set: &Vec<(usize, &[f32])>,
    headers: &HashMap<String, usize>,
) -> Option<(Solution, HashMap<Variable, usize>)> {
    let obj_field;
    let mut problem = match &query.obj {
        QueryObjective::Maximize(v) => {
            obj_field = v;
            Problem::new(OptimizationDirection::Maximize)
        }
        QueryObjective::Minimize(v) => {
            obj_field = v;
            Problem::new(OptimizationDirection::Minimize)
        }
    };
    let obj_field_idx = if let Some(idx) = headers.get(obj_field) {
        *idx
    } else {
        println!("Cannot find objective field '{}'", obj_field);
        return None;
    };
    let vars = picked_rows
        .iter()
        .enumerate()
        .filter_map(|(i, row)| {
            let coefficient = row[obj_field_idx] as f64;
            let boundaries = (f64::NEG_INFINITY, f64::INFINITY);
            Some((problem.add_var(coefficient, boundaries), i))
        })
        .collect::<HashMap<_, _>>();
    let mut cons_vals = vec![];

    query.cons.iter().for_each(|c| match c {
        QueryConstrain::Sum(expr) => {
            if let Some(index) = headers.get(&expr.attr) {
                let mut lhs = LinearExpr::empty();
                let mut sum_cof = 0.0;
                vars.iter().for_each(|(v, row)| {
                    let cof = picked_rows[*row][*index];
                    sum_cof += cof;
                    lhs.add(*v, cof as f64);
                });
                let (comp_op, mut rhs) = expr.comp.to_solver_op();
                candidates.iter().for_each(|(cluster_id, _, _)| {
                    let row = centroids[**cluster_id];
                    rhs -= row[*index];
                });
                result_set.iter().for_each(|(_, row)| {
                    rhs -= row[*index];
                });
                problem.add_constraint(lhs, comp_op, rhs as f64);
                cons_vals.push(sum_cof);
            } else {
                println!("Cannot find field \"{}\"", expr.attr);
            }
        }
        QueryConstrain::Count(count) => {
            let mut lhs = LinearExpr::empty();
            vars.iter().for_each(|(v, _row)| {
                lhs.add(*v, 1.0f64);
            });
            let (comp_op, mut rhs) = count.to_solver_op();
            candidates.iter().for_each(|_| {
                rhs -= 1.0;
            });
            result_set.iter().for_each(|_| {
                rhs -= 1.0;
            });
            problem.add_constraint(lhs, comp_op, rhs as f64);
        }
    });
    if query.repeat_0 {
        vars.iter().for_each(|(v, _row)| {
            let mut lhs = LinearExpr::empty();
            lhs.add(*v, 1.0f64);
            problem.add_constraint(lhs, ComparisonOp::Le, 1.0);
        });
    }
    match problem.solve() {
        Ok(solution) => Some((solution, vars)),
        Err(err) => {
            println!(
                "Cannot solve the linear programming problem in the refine phase: {:?}",
                err
            );
            None
        }
    }
}

fn sketch(
    query: &Query,
    tuples: &Vec<&[f32]>,
    headers: &HashMap<String, usize>,
) -> Option<(Solution, Vec<(usize, Variable)>)> {
    let obj_field;
    let mut problem = match &query.obj {
        QueryObjective::Maximize(v) => {
            obj_field = v;
            Problem::new(OptimizationDirection::Maximize)
        }
        QueryObjective::Minimize(v) => {
            obj_field = v;
            Problem::new(OptimizationDirection::Minimize)
        }
    };
    let obj_field_idx = if let Some(idx) = headers.get(obj_field) {
        *idx
    } else {
        println!("Cannot find objective field '{}'", obj_field);
        return None;
    };
    let vars = tuples
        .iter()
        .map(|row| {
            let coefficient = row[obj_field_idx] as f64;
            let boundaries = (f64::NEG_INFINITY, f64::INFINITY);
            problem.add_var(coefficient, boundaries) // TODO: Refine this
        })
        .enumerate()
        .collect::<Vec<_>>();
    let mut cons_vals = vec![];
    query.cons.iter().for_each(|c| match c {
        QueryConstrain::Sum(expr) => {
            if let Some(index) = headers.get(&expr.attr) {
                let mut lhs = LinearExpr::empty();
                let mut sum_cof = 0.0;
                vars.iter().for_each(|(row, v)| {
                    let cof = tuples[*row][*index];
                    sum_cof += cof;
                    lhs.add(*v, cof as f64);
                });
                let (comp_op, rhs) = expr.comp.to_solver_op();
                problem.add_constraint(lhs, comp_op, rhs as f64);
                cons_vals.push(sum_cof);
            } else {
                println!("Cannot find field \"{}\"", expr.attr);
            }
        }
        QueryConstrain::Count(count) => {
            let mut lhs = LinearExpr::empty();
            vars.iter().for_each(|(_row, v)| {
                lhs.add(*v, 1.0f64);
            });
            let (comp_op, rhs) = count.to_solver_op();
            problem.add_constraint(lhs, comp_op, rhs as f64);
        }
    });
    if query.repeat_0 {
        vars.iter().for_each(|(_row, v)| {
            let mut lhs = LinearExpr::empty();
            lhs.add(*v, 1.0f64);
            problem.add_constraint(lhs, ComparisonOp::Le, 1.0);
        });
    }
    match problem.solve() {
        Ok(solution) => Some((solution, vars)),
        Err(err) => {
            println!("Cannot solve the linear programming problem: {:?}", err);
            None
        }
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

impl QueryConstrainComp {
    fn to_solver_op(&self) -> (ComparisonOp, f32) {
        let rhs;
        let comp_op = match self {
            QueryConstrainComp::LessEq(r) => {
                rhs = *r;
                ComparisonOp::Le
            }
            QueryConstrainComp::GreatEq(r) => {
                rhs = *r;
                ComparisonOp::Ge
            }
            QueryConstrainComp::Eq(r) => {
                rhs = *r;
                ComparisonOp::Eq
            }
        };
        (comp_op, rhs)
    }
}
