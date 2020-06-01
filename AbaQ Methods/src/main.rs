mod root_finding;
use root_finding::methods::*;
use root_finding::utilities::Error;

mod linear_equations;
use linear_equations::methods::*;

mod interpolation;


use std::f64;
use ndarray::prelude::*;
use nalgebra::Complex;
use rand::Rng;
use ndarray_linalg::generate::{random};
use ndarray_linalg::lapack::eigh::*;
use ndarray_linalg::solve::*;
use ndarray::OwnedRepr;
use crate::linear_equations::utilities::{spectral_radius, IterationType};
use crate::interpolation::methods::{vandermonde, divided_differences, lagrange_pol, linear_splines, quadratic_splines, cubic_splines};


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
    {/*
        let mat = array![[10f64, -3., 0., -1.,],
                         [-3., 15., 2., 7.,],
                         [0., 2., 8., 2.,],
                         [-1., 7., 2., 9.,],];
        let vc = array![1., 1., 1., 1.,];
        let vc0 = array![0., 0., 0., 0.,];*/


        //let a = array![[1., 2., 3., 4.,]];
        //let a1 = array![1.5, 2.5, 3.5,];
        /*let mat = array![[8.1472, 0.9754, 1.5761, 1.4189, 6.5574,],
         [9.0579, 2.7850, 9.7059, 4.2176, 0.3571,] ,
            [1.2699, 5.4688, 9.5717, 9.1574, 8.4913,],
            [9.1338, 9.5751, 4.8538, 7.9221, 9.3399,] ,
            [6.3236, 9.6489,  8.0028, 9.5949, 6.7874], ];
        let ans = match spectral_radius(&mat) {
            Ok(e) => e,
            Err(_) => 0.
        };
        println!("{}", ans);
        println!("{:?}", mat);*/
        /*let m = array![[4., -1., 0., 3.,],
                       [1., 15.5, 3., 8.,],
                       [8., -1.3, -4., 1.1,],
                       [14., 5., -2., 30.,],];

        let v = Array1::<f64>::zeros(4);
        let x0 = Array1::<f64>::ones(4);
        iterate(&m, &v, &x0, IterationType::Jacobi, 1e-7, 100);*/
        /*let a = array![[3., 2., -3.4, 1.],
                       [2., 6., 2., -5.],
                       [-3.4, -2., 3., 2.],
                       [1., -5., 2., 1.]];
        let b = Array1::<f64>::ones(4);
        let x0 = Array1::<f64>::ones(4);
        //let (ans, stag) = gaussian_elimination_total_pivoting(&a, &b);
        //println!("{:?}\n--------\n{:?}\n\n", ans, stag);
        let (ans, stag) = cholesky(&a, &b);

        println!("{:?}\n--------\n{:?}\n\n", ans, stag);
        let nmax = 100usize;
        let tol = 1e-7;

        let x = array![-1., 1., 2.338688085676038, 6.,];
        let y = array![1., 3., -0.494, -2.,];
        let nmax = 100usize;
        let tol = 1e-7;*/


        //iterate_parcial(&a, &b, &x0, tol, 100);
        //iterate(&a, &b, &x0, IterationType::SOR(0.000000005), tol, 100);
        //doolittle(&a, &b);
        //println!("{}", a.solve_into(b).unwrap());

        /*let x0 = Array1::<f64>::zeros(4);
        let x = array![-1., 0., 3., 4.,];
        let y = array![15.5, 3., 8., 1.,];
        let nmax = 100usize;
        let tol = 1e-7;

        println!("\n=========================================================\nsimple elimination");
        simple_elimination_lu(&a);
        println!("\n=========================================================\npivoting elimination");
        pivoting_elimination_lu(&a);
        println!("\n=========================================================\ncrout");
        crout(&a, &b);
        println!("\n=========================================================\ncholesky");
        cholesky(&a, &b);
        println!("\n=========================================================\ndoolittle");
        doolittle(&a, &b);
        println!("\n=========================================================\njacobi");
        iterate(&a, &b, &x0, IterationType::Jacobi, tol, nmax);
        println!("\n=========================================================\ngauss seidel");
        iterate(&a, &b, &x0, IterationType::GaussSeidel, tol, nmax);
        println!("\n=========================================================\nsor");
        iterate(&a, &b, &x0, IterationType::SOR(1.5), tol, nmax);
        println!("\n=========================================================\nvandermonde");
        vandermonde(&x, &y);
        println!("\n=========================================================\ndivided differences");
        divided_differences(&x, &y);
        println!("\n=========================================================\nlagrange");
        lagrange_pol(&x, &y);
        println!("\n=========================================================\n");
*/

        let x = array![-2., -1., 2., 3.,];
        let y = array![12.1353, 6.3679, -4.6109, 2.0855,];

        let (ans, a) = cubic_splines(&x, &y).unwrap();
        println!("{:?}\n\n{}", ans, a);

    }

}
