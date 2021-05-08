use humantime::format_duration;
use json5;
use linfa::{traits::*, DatasetBase};
use linfa_clustering::*;
use lp_solvers::problem::{Problem, StrExpression, Variable};
use lp_solvers::solvers::{Cplex, SolverTrait};
use lp_solvers::{
    lp_format::{Constraint, LpObjective},
    solvers::{Solution, Status},
};
use memchr::{memchr_iter, Memchr};
use memmap2::*;
use ndarray::*;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::BufRead;
use std::{cmp::Ordering, time::Instant};

const STRING_BUFFER_SIZE: usize = 10240;

#[derive(Serialize, Deserialize, Debug)]
enum QueryObjective {
    Maximize(String),
    Minimize(String),
}

#[derive(Serialize, Deserialize, Debug)]
enum QueryConstrainComp {
    // Does not support lesser and greater
    Eq(f32),
    LE(f32),
    GE(f32),
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
    direct: Option<bool>,
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
    fn all_rows(&self) -> Vec<&[f32]> {
        let cols = self.num_cols;
        (0..self.buffer.len())
            .step_by(cols)
            .map(|i| &self.buffer.as_slice()[i..i + cols])
            .collect()
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
    let data = {
        let _w = Watch::start("Read data");
        read_all_data(&mapped, lines, first_line, num_cols)
    };
    println!(
        "Read total of {} row of data, preparing clustering data",
        data.rows
    );
    let k = 64;
    let clusters = {
        let _w = Watch::start("Clustering");
        clustering(&data, k)
    };
    let centroids = clusters.representatives_vec();
    assert_eq!(
        centroids.len(),
        k,
        "Have {} centroids, expect {}",
        centroids.len(),
        k
    );
    assert_eq!(clusters.assignments.len(), data.rows);
    let mut cluster_rows = HashMap::new();
    clusters.assignments.iter().enumerate().for_each(|(i, c)| {
        cluster_rows.entry(*c).or_insert_with(|| vec![]).push(i);
    });
    // println!(
    //     "Clusters: {:?}",
    //     cluster_rows
    //         .iter()
    //         .map(|(c, r)| format!("{}: {}", c, r.len()))
    //         .collect::<Vec<_>>()
    // );
    let mut query_id = 4;
    println!("Query [{}] >", query_id);
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let str_line = line.unwrap();
        let query_res = json5::from_str::<Query>(&str_line);
        match query_res {
            Ok(query) => {
                println!("Accepting query {:?}", query);
                if query.direct != Some(true) {
                    // Use sketch-refine approach
                    {
                        let _w = Watch::start("Sketch-refine query");
                        run_query(
                            query_id,
                            &data,
                            &query,
                            &centroids,
                            &headers,
                            &header_index,
                            &cluster_rows,
                        );
                    }
                } else {
                    // Use direct approach
                    {
                        let _w = Watch::start("Direct query");
                        run_direct_query(query_id, &data, &query, &headers, &header_index);
                    }
                }
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
    query_id: usize,
    data: &DataSet,
    query: &Query,
    centroids: &Vec<&[f32]>,
    headers: &Vec<&str>,
    header_index: &HashMap<String, usize>,
    cluster_rows: &HashMap<usize, Vec<usize>>,
) {
    println!("Using sketch-refine approach");
    if let Some((solution, _vars)) = sketch(query_id, query, centroids, headers, header_index) {
        println!("Sketch solution: {:?}", solution);
        let mut rand = thread_rng();
        if solution.status != Status::Optimal {
            println!("Sketch result not optimal: {:?}", solution.status);
        }
        let selected_variable = solution
            .results
            .iter()
            // remove start with `x`
            .filter(|(var, _)| var.starts_with("v"))
            .map(|tuple| {
                // println!("Checking selected tuple {:?}", tuple);
                let (var, val) = tuple;
                let var_id = var[1..].parse::<usize>().unwrap();
                // println!("{} => {}", var, var_id);
                // assert!(*val >= 0.0, "Variable is negative {:?} = {}", var, val);
                (var_id, tuple)
            })
            .filter(|(_i, (_var, val))| **val >= 1.0)
            .collect::<Vec<_>>();
        let mut candidates = selected_variable
            .iter()
            .map(|(cluster, (var, _))| {
                // println!("Extract cluster {}", cluster);
                (
                    cluster,
                    var,
                    cluster_rows[cluster]
                        .iter()
                        .map(|row_id| (*cluster, data.row(*row_id)))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        println!("Have {} candidates", candidates.len());
        candidates.shuffle(&mut rand);
        let mut candidates_copy = candidates.clone();
        let mut result_set = vec![];
        let mut iter_num = 0;
        while let Some((cluster_id, _var, picked_rows)) = candidates.pop() {
            if let Some((solution, vars)) = refine(
                query_id,
                iter_num,
                query,
                centroids,
                &candidates,
                *cluster_id,
                &picked_rows,
                &result_set,
                headers,
                header_index,
            ) {
                // println!("Iter {}, solution {:?}", iter_num, solution);
                iter_num += 1;
                // All the variables == 0.0, backtrack, do cluster_id first
                if solution.status == Status::Infeasible
                    || solution.status == Status::NotSolved
                    || solution.status == Status::Unbounded
                    || solution.results.iter().all(|(_, val)| *val == 0.0)
                {
                    println!(
                        "Cannot have a viable solution from the solver: {:?}. Backtracking",
                        solution.status
                    );
                    reorder_candidate(&mut candidates_copy, cluster_id);
                    candidates = candidates_copy.clone();
                    // Reset variables
                    result_set.clear();
                    iter_num = 0;
                    continue;
                }
                if let Some((var, _val)) = solution
                    .results
                    .iter()
                    .filter(|(_var, val)| **val >= 1.0)
                    .collect::<Vec<_>>()
                    .get(0)
                {
                    let var_id = var[1..].parse::<usize>().unwrap();
                    //println!("{} => {}", var, var_id);
                    let picked_row_id = vars[var_id].0;
                    let picked_row = picked_rows[picked_row_id];
                    result_set.push(picked_row);
                } else {
                    break;
                }
            } else {
                return;
            }
        }
        println!("Result: {} rows", result_set.len());
        headers.iter().for_each(|s| print!("|{}\t", s));
        println!("|");
        result_set.iter().for_each(|(_row_id, row)| {
            row.iter().for_each(|val| {
                print!("|{}\t", val);
            });
            println!("|");
        });
    }
}

fn reorder_candidate(
    candidates: &mut Vec<(&usize, &&String, Vec<(usize, &[f32])>)>,
    cluster_id: &usize,
) {
    let index = candidates
        .iter()
        .enumerate()
        .find(|(_, (cid, _, _))| **cid == *cluster_id)
        .map(|(i, _)| i)
        .expect("Unexpected, should found the candidate");
    let cluster = candidates.remove(index);
    // Put it back to the head of the candidates
    candidates.insert(0, cluster);
}

fn refine(
    query_id: usize,
    iter: usize,
    query: &Query,
    centroids: &Vec<&[f32]>,
    candidates: &Vec<(&usize, &&String, Vec<(usize, &[f32])>)>,
    _picked_cluster: usize,
    picked_rows: &Vec<(usize, &[f32])>,
    result_set: &Vec<(usize, &[f32])>,
    _headers: &Vec<&str>,
    header_index: &HashMap<String, usize>,
) -> Option<(Solution, Vec<(usize, String)>)> {
    let obj_field;
    let objective = match &query.obj {
        QueryObjective::Maximize(v) => {
            obj_field = v;
            LpObjective::Maximize
        }
        QueryObjective::Minimize(v) => {
            obj_field = v;
            LpObjective::Minimize
        }
    };
    let mut problem = Problem {
        name: format!("pq-{}-refine-{}", query_id, iter),
        sense: objective,
        objective: StrExpression(obj_field.to_owned()),
        variables: vec![],
        constraints: vec![],
    };
    let obj_field_idx = if let Some(idx) = header_index.get(obj_field) {
        *idx
    } else if obj_field == "*" {
        usize::MAX
    } else {
        println!("Cannot find objective field '{}'", obj_field);
        return None;
    };
    println!("Iter {} Picked {} rows", iter, picked_rows.len());
    let mut objective_expr = "".to_string();
    let vars = picked_rows
        .iter()
        .enumerate()
        .map(|(i, (_, row))| {
            let coefficient = *row.get(obj_field_idx).unwrap_or(&1.0) as f64;
            let var_name = format!("v{}", i);
            let variable = Variable {
                name: var_name.clone(),
                is_integer: true, // We want ILP solver
                lower_bound: 0.0,
                upper_bound: f64::MAX,
            };
            problem.variables.push(variable);
            objective_expr.push_str(&format!("+{}{}", coefficient, var_name));
            (i, var_name)
        })
        .collect::<Vec<_>>();
    objective_expr = objective_expr[1..].to_string();
    let mut constrains = vec![];
    query.cons.iter().for_each(|c| match c {
        QueryConstrain::Sum(expr) => {
            if let Some(index) = header_index.get(&expr.attr) {
                let mut lhs = String::with_capacity(STRING_BUFFER_SIZE);
                vars.iter().for_each(|(row, v)| {
                    let cof = picked_rows[*row].1[*index];
                    lhs.push_str(&format!("+{}{}", cof, *v));
                });
                let (comp_op, mut rhs) = expr.comp.to_solver_op();
                candidates.iter().for_each(|(cluster_id, _, _)| {
                    let row = centroids[**cluster_id];
                    rhs -= row[*index];
                });
                result_set.iter().for_each(|(_, row)| {
                    rhs -= row[*index];
                });
                lhs = lhs[1..].to_string();
                constrains.push(Constraint {
                    lhs: StrExpression(lhs),
                    operator: comp_op,
                    rhs: rhs as f64,
                });
            } else {
                println!("Cannot find field \"{}\"", expr.attr);
            }
        }
        QueryConstrain::Count(count) => {
            let mut lhs = String::with_capacity(STRING_BUFFER_SIZE);
            vars.iter().for_each(|(_row, v)| {
                lhs.push_str(&format!("+{}", *v));
            });
            let (comp_op, rhs) = count.to_solver_op();
            lhs = lhs[1..].to_string();
            constrains.push(Constraint {
                lhs: StrExpression(lhs),
                operator: comp_op,
                rhs: rhs as f64,
            });
        }
    });
    if query.repeat_0 {
        vars.iter().for_each(|(_row, v)| {
            constrains.push(Constraint {
                lhs: StrExpression(v.clone()),
                operator: Ordering::Less,
                rhs: 1.0,
            })
        });
    }
    problem.constraints = constrains;
    problem.objective = StrExpression(objective_expr);
    match Cplex::default().run(&problem) {
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
    query_id: usize,
    query: &Query,
    tuples: &Vec<&[f32]>,
    _headers: &Vec<&str>,
    header_index: &HashMap<String, usize>,
) -> Option<(Solution, Vec<(usize, String)>)> {
    let obj_field;
    let objective = match &query.obj {
        QueryObjective::Maximize(v) => {
            obj_field = v;
            LpObjective::Maximize
        }
        QueryObjective::Minimize(v) => {
            obj_field = v;
            LpObjective::Minimize
        }
    };
    let mut problem = Problem {
        name: format!("pq-{}-sketch", query_id),
        sense: objective,
        objective: StrExpression(obj_field.to_owned()),
        variables: vec![],
        constraints: vec![],
    };
    let obj_field_idx = if let Some(idx) = header_index.get(obj_field) {
        *idx
    } else if obj_field == "*" {
        usize::MAX
    } else {
        println!("Cannot find objective field '{}'", obj_field);
        return None;
    };
    let mut objective_expr = "".to_string();
    let vars;
    {
        let _w = Watch::start("Constructed problem for Cplex");
        vars = tuples
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let coefficient = *row.get(obj_field_idx).unwrap_or(&1.0) as f64;
                let var_name = format!("v{}", i);
                let variable = Variable {
                    name: var_name.clone(),
                    is_integer: true, // We want ILP solver
                    lower_bound: 0.0,
                    upper_bound: f64::MAX,
                };
                problem.variables.push(variable);
                objective_expr.push_str(&format!("+{}{}", coefficient, var_name));
                (i, var_name)
            })
            .collect::<Vec<_>>();
        objective_expr = objective_expr[1..].to_string();
        let mut constrains = vec![];
        query.cons.iter().for_each(|c| match c {
            QueryConstrain::Sum(expr) => {
                if let Some(index) = header_index.get(&expr.attr) {
                    //println!("Attr: {}", expr.attr);
                    let mut lhs = String::with_capacity(STRING_BUFFER_SIZE);
                    let mut lhs_sum = 0.0;
                    vars.iter().for_each(|(row, v)| {
                        let cof = tuples[*row][*index];
                        lhs_sum += cof;
                        lhs.push_str(&format!("+{}{}", cof, *v));
                    });
                    let (comp_op, rhs) = expr.comp.to_solver_op();
                    //println!("Sketch constrain rhs {}, lhs sum {}, op {:?}, exp: {}", rhs, lhs_sum, comp_op, lhs);
                    // lhs, comp_op, rhs as f64
                    lhs = lhs[1..].to_string();
                    constrains.push(Constraint {
                        lhs: StrExpression(lhs),
                        operator: comp_op,
                        rhs: rhs as f64,
                    });
                } else {
                    println!("Cannot find field \"{}\"", expr.attr);
                }
            }
            QueryConstrain::Count(count) => {
                let mut lhs = String::with_capacity(STRING_BUFFER_SIZE);
                vars.iter().for_each(|(_row, v)| {
                    lhs.push_str(&format!("+{}", *v));
                });
                let (comp_op, rhs) = count.to_solver_op();
                lhs = lhs[1..].to_string();
                constrains.push(Constraint {
                    lhs: StrExpression(lhs),
                    operator: comp_op,
                    rhs: rhs as f64,
                });
            }
        });
        if query.repeat_0 {
            vars.iter().for_each(|(_row, v)| {
                constrains.push(Constraint {
                    lhs: StrExpression(v.clone()),
                    operator: Ordering::Less,
                    rhs: 1.0,
                })
            });
        }
        // vars.iter().for_each(|(_row, v)| {
        //     let mut lhs = LinearExpr::empty();
        //     lhs.add(*v, 1.0f64);
        //     problem.add_constraint(lhs, ComparisonOp::Ge, 0.0);
        // });
        //println!("Sketch constrains {:?}", constrains.);
        //println!("Sketch objectives {:?} {:?}", objective, objective_expr);
        problem.constraints = constrains;
        problem.objective = StrExpression(objective_expr);
    }
    {
        let _w = Watch::start("Solved the problem with Cplex");
        match Cplex::default().run(&problem) {
            Ok(solution) => Some((solution, vars)),
            Err(err) => {
                println!(
                    "Cannot solve the linear programming problem phase: {:?}",
                    err
                );
                None
            }
        }
    }
}

fn to_ndarray(data: &DataSet) -> Array2<f32> {
    let shape = (data.rows, data.num_cols);
    let vec = data.buffer.clone();
    Array2::from_shape_vec(shape, vec).expect("Cannot convert array")
}

fn clustering(data: &DataSet, k: usize) -> Clusters {
    println!("Clustering with k {}", k);
    let ndarray = to_ndarray(data);
    let obversations = DatasetBase::from(ndarray);
    let rand = StdRng::from_rng(thread_rng()).unwrap();
    // let clusters = Dbscan::params(k).tolerance(1e-2).transform(&ndarray);
    let model = {
        let _w = Watch::start("Cluster Fitting...");
        KMeans::params_with_rng(k, rand)
            .max_n_iterations(100)
            .fit(&obversations)
            .expect("KMeans fitted")
    };
    let assignments = {
        let _w = Watch::start("Cluster Predicting...");
        model.predict(obversations)
    };
    Clusters {
        ncols: data.num_cols,
        representatives: assignments.records,
        assignments: assignments.targets,
    }
}

struct Clusters {
    ncols: usize,
    assignments: Array1<usize>,
    representatives: Array2<f32>,
}

impl Clusters {
    fn representatives_vec(&self) -> Vec<&[f32]> {
        let slice = self.representatives.as_slice().unwrap();
        let ncols = self.ncols;
        let ndata = slice.len();
        (0..ndata)
            .step_by(ncols)
            .map(|i| {
                let ends = i + ncols;
                &slice[i..ends]
            })
            .collect()
    }
}

// fn clustering(data: &DataSet, k: usize) -> KMeansState<f32> {
//     let iter = 32;
//     let (sample_cnt, sample_dims, k, max_iter) = (data.rows, data.num_cols, k, iter);
//     let data_clone = data.buffer.clone();
//     // Calculate kmeans, using kmean++ as initialization-method
//     println!(
//         "Clustering with k: {}, iter: {}, dim {}, size {}",
//         k,
//         iter,
//         data.num_cols,
//         data_clone.len()
//     );
//     let kmean = KMeans::new(data_clone, sample_cnt, sample_dims);
//     let result = kmean.kmeans_minibatch(
//         1024,
//         k,
//         max_iter,
//         KMeans::init_kmeanplusplus,
//         &KMeansConfig::default(),
//     );
//     println!("Clustered...");
//     result
// }

fn read_line(mem: &Mmap, start: usize, ends: usize) -> Vec<&str> {
    let raw_bytes = &mem[start..ends];
    let line = std::str::from_utf8(raw_bytes).unwrap();
    // print!("Read line {}", line);
    line.split(",").collect()
}

fn str_line_to_num_line(line_str: Vec<&str>) -> Vec<f32> {
    line_str
        .iter()
        .map(|str| match str.trim().parse() {
            Ok(num) => num,
            Err(e) => panic!("Error '{:?}' on parsing {}, line {:?}", e, str, line_str),
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
    fn to_solver_op(&self) -> (Ordering, f32) {
        let rhs;
        let comp_op = match self {
            QueryConstrainComp::LE(r) => {
                rhs = *r;
                Ordering::Less
            }
            QueryConstrainComp::GE(r) => {
                rhs = *r;
                Ordering::Greater
            }
            QueryConstrainComp::Eq(r) => {
                rhs = *r;
                Ordering::Equal
            }
        };
        (comp_op, rhs)
    }
}

fn run_direct_query(
    query_id: usize,
    data: &DataSet,
    query: &Query,
    headers: &Vec<&str>,
    header_index: &HashMap<String, usize>,
) {
    // Reuse sketch for runnning solver for direct query
    println!("Using direct approach");
    let all_rows = data.all_rows();
    if let Some((solution, _vars)) = sketch(query_id, query, &all_rows, headers, header_index) {
        if solution.status != Status::Optimal {
            println!("Driect result not optimal: {:?}", solution.status);
        }
        let result_set = solution
            .results
            .iter()
            // remove start with `x`
            .filter(|(var, _)| var.starts_with("v"))
            .map(|tuple| {
                // println!("Checking selected tuple {:?}", tuple);
                let (var, _val) = tuple;
                let var_id = var[1..].parse::<usize>().unwrap();
                // println!("{} => {}", var, var_id);
                // assert!(*val >= 0.0, "Variable is negative {:?} = {}", var, val);
                (var_id, tuple)
            })
            .filter(|(_i, (_var, val))| **val >= 1.0)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let max_rows = 50;
        println!("Result: {} rows", result_set.len());
        headers.iter().for_each(|s| print!("|{}\t", s));
        println!("|");
        result_set
            .iter()
            .take(max_rows)
            .map(|row_id| &all_rows[*row_id])
            .for_each(|row| {
                row.iter().for_each(|val| {
                    print!("|{}\t", val);
                });
                println!("|");
            });
        if result_set.len() > max_rows {
            println!("And more...")
        }
    } else {
        println!("Cannot find the solution for direct approach")
    }
}

struct Watch {
    name: &'static str,
    start: Instant,
}

impl Watch {
    pub fn start(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
        }
    }
}

impl Drop for Watch {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        println!(
            "Time for {} elapsed {}, total {} ms",
            self.name,
            format_duration(elapsed),
            elapsed.as_millis()
        )
    }
}
