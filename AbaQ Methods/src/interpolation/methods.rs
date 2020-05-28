use ndarray::{Array1, Array2, Zip};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use crate::linear_equations::utilities::Error;
use crate::linear_equations::methods::{crout, cholesky, gaussian_elimination, elimination_with_total_pivoting};
use ndarray_linalg::{Solve, Determinant};

pub fn vandermonde(x: &Array1<f64>, y: &Array1<f64>) -> Result<String, Error>{
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let m = Array2::<f64>::from_shape_fn((n, n),
        |(i, j)| {
            x[i].powi((n - j - 1) as i32)
        }
    );
    let de = m.det().unwrap();
    println!("det: {}\n{}", de, m);
    let a = m.solve_into(y.clone()).unwrap();
    let mut pol = String::new();
    for i in (1..n).rev() {
        let e = a[n - i - 1];
        if e != 0. {
            let xi = match i {
                0 => String::new(),
                1 => String::from("x"),
                _ => format!("x^{}", i),
            };
            if e == 1. {
                pol.push_str(format!("{}", xi).as_str())
            } else if e == -1. {
                pol.push_str(format!("-{}", xi).as_str())
            } else {
            pol.push_str(format!("{:+}*{}", e, xi).as_str())
        }
        }
    }
    if a[n - 1] != 0. {
        pol.push_str(format!("{:+}", a[n - 1]).as_str());
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("{}\n{}", a, pol);
    Ok(pol)
}

pub fn divided_differences(x: &Array1<f64>, fx: &Array1<f64>) -> Result<String, Error> {
    let n = x.len();
    if n != fx.len() {
        return Err(Error::BadIn);
    }
    let mut m = Array2::<f64>::zeros((n, n + 1));
    let (mut c0, mut c1) = m.multi_slice_mut((s![.., 0], s![.., 1]));
    Zip::from(&mut c0).and(&mut c1).and(x).and(fx).par_apply(|m0, m1, xi, yi| {
        *m0 = *xi;
        *m1 = *yi;
    });
    //println!("{}", m);

    for k in 2..=n {
        for i in k - 1..n {
            m[(i, k)] = (m[(i, k-1)] - m[(i - 1, k - 1)]) / (x[i] - x[i + 1 - k]);
        }
    }
    println!("{}", m);
    let mut pol = format!("{:+}", m[(0, 1)]);
    for i in 1..n {
        if m[(i, i + 1)] != 0. {
            pol.push_str(format!("{:+}", m[(i, i + 1)]).as_str());
            for j in 0..i {

                pol.push_str(format!("*(x{:+})", -x[j]).as_str());
            }
        }
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("{}", pol);
    Ok(pol)
}

pub fn lagrange_pol(x: &Array1<f64>, y: &Array1<f64>) -> Result<String, Error> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::BadIn);
    }
    let mut pol = String::new();
    let ls = Zip::indexed(x).par_apply_collect(|i, xi| {
        x.indexed_iter().filter(|(j, _)| {
            *j != i
        }).fold(1., |acc, (_, xj)| {
            acc * (xi - xj)
        })
    });
    for i in 0..n {
        pol.push_str(format!("{:+}", y[i] / ls[i]).as_str());
        for j in 0..i {
            pol.push_str(format!("(x{:+})", -x[j]).as_str());
        }
        for j in i+1..n {
            pol.push_str(format!("(x{:+})", -x[j]).as_str());
        }
    }
    if pol.starts_with("+") {
        pol.remove(0);
    }
    println!("\n\n{}", pol);
    Ok(pol)
}
