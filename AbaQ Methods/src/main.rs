mod root_finding;
use root_finding::methods::*;
use crate::root_finding::utilities::Error;

mod linear_equations;
use linear_equations::methods::*;

use std::f64;
use nalgebra::{DMatrix, DVector};


fn main() {
    /*{
        let f = |x: f64| Ok(x.sin().powi(2).ln_1p() - 0.5);
        let df = |x: f64| Ok(x.sin() * x.cos() * 2f64 * (x.sin().powi(2) + 1f64).powi(-1));
        let f1 = |x: f64| Ok(x.sin().powi(2).ln_1p() - 0.5 - x);
        let g = |x: f64| Ok(x.sin().powi(2).ln_1p() - 0.5);
        let h = |x: f64| Ok(x.exp() - x - 1f64);
        let dh = |x: f64| Ok(x.exp() - 1f64);
        let d2h = |x: f64| Ok(x.exp());
        let tol = 1e-7f64;
        let n = 100;
        let et = Error::Absolute;
        let is = incremental_search(f, -3f64, 0.5, n);
        println!("Incremental Search : {:?}", &is);
        let (bs, bslog) = bisection(f, 1f64, 0f64, tol, n, et);
        println!("Bisection: {:?}\n{}\n", &bs, bslog);
        let (fp, fplog) = false_position(f, 1f64, 0f64, tol, n, et);
        println!("False position: {:?}\n{}\n", &fp, fplog);
        let (pf, pflog) = fixed_point(f1, g, -0.5, tol, n, et);
        println!("Fixed point: {:?}\n{}\n", &pf, pflog);
        let (nt, ntlog) = newton(f, df, 0.5, tol, n, et);
        println!("Newton: {:?}\n{}\n", &nt, ntlog);
        let (sc, sclog) = secant(f, 0.5, 1f64, tol, n, et);
        println!("Secant: {:?}\n{}\n", &sc, sclog);
        let (mr, mrlog) = multiple_root(h, dh, d2h, 1f64, tol, n, et);
        println!("Multiple roots:{:?}\n{}\n", &mr, mrlog);
        let (st, stlog) = steffensen(f, 0.5, tol, n, et);
        println!("Steffensen: {:?}\n{}\n", &st, stlog);
        let (ml, mllog) = muller(f, 1., 2., 3., tol, n, et);
        println!("Muller: {:?}\n{}\n", &ml, mllog);
        let (apf, apflog) = accelerated_fixed_point(f1, g, -0.5, tol, n, et);
        println!("Aitken: {:?}\n{}\n", &apf, apflog);
    }*/
    {
        /*let matr = vec![
            14f64, 0., 0., 0.,
            6., 13.71, 0., 0.,
            -2., 2.42, -25.23, 0.,
            3., -5.64, 6.38, 14.05,
        ];*/
        let matr = vec![
            14.05, 6.38, -5.64, 3.,
            0., -25.23, 2.42, -2.,
            0., 0., 13.71, 6.,
            0., 0., 0., 14.,
        ];
        let vect = vec![22.13, -33.02, 29.42, 12f64, ];
    }

}
