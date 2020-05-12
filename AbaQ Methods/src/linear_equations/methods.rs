use num_traits::{Zero, One};
use nalgebra::{ComplexField, Complex};
use crate::linear_equations::utilities::{Error, swap_rows, swap_entire_cols, FactorizationType, IterationType, spectral_radius};
use ndarray::{Array2, Array1, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::mem;
use num_traits::float::FloatCore;
use ndarray_linalg::norm::*;
use ndarray_linalg::Inverse;

fn back_substitution<T: ComplexField>(A: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, Error> {
    if !A.is_square() {
        return Err(Error::BadIn);
    }
    let n = b.len();
    let mut x = Array1::<T>::zeros(n);
    if A[(n - 1, n - 1)].is_zero() {
        return Err(Error::DivBy0);
    }
    x[n-1] = b[n-1]/A[(n - 1, n - 1)];
    let mut sum: T;
    for i in (0..=n-2).rev() {
        sum = T::zero();
        for p in (i + 1) ..= (n - 1) {
            sum += A[(i, p)] * x[p];
        }
        if A[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        x[i] = (b[i] - sum)/A[(i, i)];
    }
    Ok(x)

}

pub(crate) fn forward_substitution<T: ComplexField>(A: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, Error> {
    if !A.is_square() {
        return Err(Error::BadIn);
    }
    let n = b.len();
    let mut x = Array1::<T>::zeros(n);
    if A[(0, 0)].is_zero() {
        return Err(Error::DivBy0);
    }
    x[0] = b[0]/A[(0, 0)];
    println!("{}", x);
    let mut sum: T;
    for i in 1..=n-1 {
        sum = T::zero();
        for p in 0..= (i - 1) {
            sum += A[(i, p)] * x[p];
        }
        if A[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        x[i] = (b[i] - sum)/A[(i, i)];

        println!("{}", x);
    }
    Ok(x)
}

fn eliminate(m: &mut Array2<f64>, k: usize) -> Array1<f64>{
    let row_k = Zip::from(m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
    Zip::from(m.slice_mut(s![k+1.., k..]).genrows_mut())
        .par_apply_collect(|mut row_i| {
            let mul = row_i[0]/row_k[0];
            Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
            mul
        })
}

pub fn simple_elimination(m: &Array2<f64>) -> Result<Array2<f64>, Error> {
    let n = m.nrows();
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    println!("{}", new_m);
    for k in 0..n-1 {
        if new_m[[k, k]] == 0. {
            for i in k+1..n {
                if new_m[[i, k]] != 0. {
                    swap_rows(&mut new_m, i, k, k);
                    break;
                }
            }
        }
        let mults = eliminate(&mut new_m, k);
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_m, mults);
    }
    Ok(new_m)
}

pub fn elimination_with_partial_pivoting(m: &Array2<f64>) -> Result<Array2<f64>, Error> {
    let n = m.nrows();
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    println!("{}", new_m);
    for k in 0..n-1 {
        let (mut max, mut max_row) = (new_m[(k, k)].abs(), k);
        for s in k+1..n {
            let tmp = new_m[(s, k)].abs();
            if tmp > max {
                max = tmp;
                max_row = s;
            }
        }
        if max == 0. {
            return Err(Error::MultipleSolution);
        }
        if max_row != k {
            swap_rows(&mut new_m, max_row, k, k);
        }
        let mults = eliminate(&mut new_m, k);
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_m, mults);
    }
    Ok(new_m)
}

pub fn elimination_with_total_pivoting(m: &Array2<f64>) -> Result<(Array2<f64>, Array1<usize>), Error> {
    let n = m.nrows();
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    let mut marks = Array1::<usize>::from((0..n).collect::<Vec<usize>>());
    println!("{} \nmarks:\n{}", new_m, marks);
    for k in 0..n-1 {
        let (mut max, mut max_row, mut max_col) = (0., k, k);
        for r in k..n {
            for s in k..n {
                let tmp = new_m[(r, s)].abs();
                if tmp > max {
                    max = tmp;
                    max_row = r;
                    max_col = s;
                }
            }
        }
        if max == 0. {
            return Err(Error::MultipleSolution);
        }
        if max_row != k {
            swap_rows(&mut new_m, max_row, k, k);
        }
        if max_col != k {
            swap_entire_cols(&mut new_m, max_col, k);
            marks.swap(max_col, k);
        }
        let mults = eliminate(&mut new_m, k);
        println!("k: {}\nU: \n{}\nmults:\n{}\nmarks:\n{}\n----------", k, new_m, mults, marks);
    }
    Ok((new_m, marks))
}

pub fn simple_elimination_lu(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), Error> {
    let n = m.nrows();
    let mut mults = Array2::<f64>::eye(n);
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    println!("{}", new_m);
    for k in 0..n-1 {
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .and(mults.slice_mut(s![k+1.., k]))
            .par_apply(|mut row_i, mut mul_slot| {
            let mul = row_i[0]/row_k[0];
            *mul_slot = mul;
            Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
        });

        /*let mut mul: f64;
        for i in k+1..n {
            mul = new_m[(i, k)]/ new_m[(k, k)];
            mults[(i, k)] = mul;
            for j in k..n {
                new_m[(i, j)] -= mul * new_m[(k, j)]
            }
        }*/
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_m, mults);
    }
    Ok((new_m, mults))
}

pub fn pivoting_elimination_lu(m: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Array1<usize>), Error> {
    let n = m.nrows();
    let mut mults = Array2::<f64>::eye(n);
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    let mut marks = Array1::<usize>::from((0..n).collect::<Vec<usize>>());
    println!("{}", new_m);
    for k in 0..n-1 {
        let (mut max, mut max_row) = (new_m[(k, k)].abs(), k);
        for s in k+1..n {
            let tmp = new_m[(s, k)].abs();
            if tmp > max {
                max = tmp;
                max_row = s;
            }
        }
        if max == 0. {
            return Err(Error::MultipleSolution);
        }
        if max_row != k {
            swap_rows(&mut new_m, max_row, k, k);
            marks.swap(max_row, k);
        }
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .and(mults.slice_mut(s![k+1.., k]))
            .par_apply(|mut row_i, mut mul_slot| {
                let mul = row_i[0]/row_k[0];
                *mul_slot = mul;
                Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
            });
        println!("k: {}\nU: \n{}\nmults:\n{}\nmarks:\n{}\n----------", k, new_m, mults, marks);
    }
    Ok((new_m, mults, marks))
}

pub fn direct_factorization(m: &Array2<f64>, method: FactorizationType) -> Result<(Array2<f64>, Array2<f64>), Error> {
    if !m.is_square() {
        return Err(Error::BadIn);
    }
    let n = m.nrows();
    let (mut l, mut u) = (Array2::<f64>::eye(n), Array2::<f64>::eye(n));
    let mut sum: f64;
    for k in 0..n {
        sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, k])).fold(0., |ac, el, eu| ac + el * eu );
        let val = m[(k, k)] - sum;
        if val.is_zero() {
            return Err(Error::DivBy0);
        }
        l[(k, k)] = match method {
            FactorizationType::Crout => val,
            FactorizationType::Doolittle => 1.,
            FactorizationType::Cholesky => {
                if val < 0. {
                    return Err(Error::ComplexNumber);
                }
                val.sqrt()
            },
        };
        u[(k, k)] = match method {
            FactorizationType::Doolittle => val,
            FactorizationType::Crout => 1.,
            FactorizationType::Cholesky => {
                if val < 0. {
                    return Err(Error::ComplexNumber);
                }
                val.sqrt()
            },
        };

        for i in k+1..n {
            sum = Zip::from(l.slice(s![i, ..k])).and(u.slice(s![..k, k]))
                .fold(0., |ac, el, eu| ac + el * eu );
            l[(i, k)] = (m[(i, k)]-sum)/u[(k, k)];
        }

        for i in k+1..n {
            sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, i]))
                .fold(0., |ac, el, eu| ac + el * eu );
            u[(k, i)] = (m[(k, i)]-sum)/l[(k, k)];
        }
    }

    Ok((l, u))
}

pub fn direct_factorization_with_complex(m: &Array2<f64>) -> Result<(Array2<Complex<f64>>, Array2<Complex<f64>>), Error> {
    let n = m.nrows();
    let (mut l, mut u) = (Array2::<Complex<f64>>::eye(n), Array2::<Complex<f64>>::eye(n));
    let mut sum: Complex<f64>;
    for k in 0..n {
        sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, k])).fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
        let val = (Complex::<f64>::from_real(m[(k, k)]) - sum).sqrt();
        if val.is_zero() {
            return Err(Error::DivBy0);
        }
        l[(k, k)] = val;
        u[(k, k)] = val;

        for i in k+1..n {
            sum = Zip::from(l.slice(s![i, ..k])).and(u.slice(s![..k, k])).fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
            l[(i, k)] = (Complex::<f64>::from_real(m[(i, k)])-sum)/u[(k, k)];
        }

        for i in k+1..n {
            sum = Zip::from(l.slice(s![k, ..k])).and(u.slice(s![..k, i])).fold(Complex::<f64>::zero(), |ac, el, eu| ac + el * eu );
            u[(k, i)] = (Complex::<f64>::from_real(m[(k, i)])-sum)/l[(k, k)];
        }
    }

    Ok((l, u))
}

/*pub fn jacobi(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, _tol: f64, max_it: usize) -> Result<(Array1<f64>, usize), Error> {
    let n = b.len();
    let tol = _tol.abs();
    let mut x0 = _x0.clone();
    println!("0 --- {}", x0);
    let (mut err, mut i) = (f64::infinity(), 0usize);
    let mut x1 = Array1::<f64>::zeros(n);
    while err > tol && i < max_it {
        x1 = new_jacobi_set(&a, &b, &x0)?;
        println!("{} --- {}", i+1, x1);
        err = (&x1 - &x0).norm();
        x0.clone_from(&x1);
        i += 1;
    }
    Ok((x1, i))
}

fn new_jacobi_set(a: &Array2<f64>, b: &Array1<f64>,x: &Array1<f64>) -> Result<Array1<f64>, Error> {
    let n = b.len();
    let mut xn = Array1::<f64>::zeros(n);
    let mut sum: f64;
    for i in 0..n {
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        sum = Zip::from(a.slice(s![i, ..])).and(x.slice(s![..]))
            .fold(0., |ac, a_ij, x_j| ac + a_ij * x_j) - a[(i, i)];
        xn[i] = (b[i] - sum) / a[(i, i)];
    }
    Ok(xn)
}

pub fn gauss_seidel(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, w: f64, _tol: f64, max_it: usize) -> Result<(Array1<f64>, usize), Error> {
    let n = b.len();
    let tol = _tol.abs();
    let mut x0 = _x0.clone();
    println!("0 --- {}", x0);
    let (mut err, mut i) = (f64::infinity(), 0usize);
    let mut x1 = Array1::<f64>::zeros(n);
    while err > tol && i < max_it {
        x1 = new_gauss_set(&a, &b, &x0, w)?;
        err = (&x1 - &x0).norm_l2();
        println!("{} --- {} --- {}", i+1, x1, err);
        x0.clone_from(&x1);
        i += 1;
    }
    Ok((x1, i))
}

fn new_gauss_set(a: &Array2<f64>, b: &Array1<f64>,x: &Array1<f64>, w: f64) -> Result<Array1<f64>, Error> {
    let n = b.len();
    let mut xn = Array1::<f64>::zeros(n);
    let mut sum: f64;
    for i in 0..n {
        if a[(i, i)].is_zero() {
            return Err(Error::DivBy0);
        }
        sum = Zip::from(a.slice(s![i, ..i])).and(xn.slice(s![..i]))
            .fold(0., |ac, a_ij, x_j| ac + a_ij * x_j);
        sum = Zip::from(a.slice(s![i, i+1..])).and(x.slice(s![i+1..]))
            .fold(sum, |ac, a_ij, x_j| ac + a_ij * x_j);
        xn[i] = (1. - w) * x[i] - w * (b[i] + sum) / a[(i, i)];
    }
    Ok(xn)
}*/

pub fn iterate(a: &Array2<f64>, b: &Array1<f64>, _x0: &Array1<f64>, method: IterationType, _tol: f64, max_it: usize) -> Result<(Array1<f64>, f64), Error> {
    let n = b.len();
    if !a.is_square() || n != a.nrows() {
        return Err(Error::BadIn);
    }
    let tol = _tol.abs();
    let mut x_n = _x0.clone();
    let mut x_n1 = Array1::<f64>::zeros(n);
    let d = Array2::<f64>::from_diag(&a.diag());
    let u = Array2::<f64>::from_shape_fn((n, n), |(i, j)| {
        if i < j {
            return -a[(i, j)]
        } 0.
    });
    let l = Array2::<f64>::from_shape_fn((n, n), |(i, j)| {
        if i > j {
            return -a[(i, j)]
        } 0.
    });
    let (t, c) = match method {
        IterationType::Jacobi => {
            let d_inv = d.inv().unwrap();
            (d_inv.dot(&(&l + &u)), d_inv.dot(b)) },
        IterationType::GaussSeidel => {
            let dl_inv = (&d - &l).inv().unwrap();
            (dl_inv.dot(&u), dl_inv.dot(b)) },
        IterationType::SOR(w) => {
            let dl_inv = (&(&d - &(&l * w))).inv().unwrap();
            (dl_inv.dot(&(&(&d * (1. - w)) + &(&u * w))), dl_inv.dot(b) * w) },
    };
    let mut i = 0usize;
    let mut err = f64::infinity();
    let spec = match spectral_radius(&t) {
        Ok(e) => e,
        Err(_) => 0.
    };
    println!("{}", spec);
    println!("{:04E} | {:^4E} | {}", i, err, x_n);
    while err > tol && i < max_it {
        x_n1 = &t.dot(&x_n) + &c;
        err = (&x_n1 - &x_n).norm();
        x_n.clone_from(&x_n1);
        i += 1;
        println!("{:^4} | {:.2E} | {}", i, err, x_n1);
    }

    println!("{}\n{}\n{}", d, u, l);


    Ok((x_n1, spec))
}


