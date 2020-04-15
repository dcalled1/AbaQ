use crate::single_var::*;
use std::f64;

mod single_var;

fn main() {
    let f = |x: f64| x.sin().powi(2).ln_1p() - 0.5f64;
    let df = |x: f64| x.sin()*x.cos()*(-2f64)/(x.sin().powi(2) + 1f64);
    let f1 = |x: f64| x.sin().powi(2).ln_1p() - 0.5f64 - x;
    let g = |x: f64| x.sin().powi(2).ln_1p() - 0.5f64;
    let h = |x :f64| x.exp() - x - 1f64;
    let dh = |x: f64| x.exp() - 1f64;
    let ddh = |x: f64| x.exp();
    let tol = 1e-7f64;
    let n = 100;
    let et = Error::Absolute;
    let is = incremental_search(f, -3f64, 0.5, n);
    println!("{:?}",&is);
    let bs = bisection(f,1f64, 0f64, tol, n, et);
    println!("{:?}",&bs);
    let fp = false_position(f,1f64, 0f64, tol, n, et);
    println!("{:?}",&fp);
    let pf = fixed_point(f1, g, -0.5, tol, n, et);
    println!("{:?}",&pf);
}
