use num_traits::{Zero};
use nalgebra::{ComplexField};
use crate::linear_equations::utilities::{Error, swap_rows, swap_entire_cols};
use ndarray::{Array2, Array1, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::mem;

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

fn forward_substitution<T: ComplexField>(A: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>, Error> {
    if !A.is_square() {
        return Err(Error::BadIn);
    }
    let n = b.len();
    let mut x = Array1::<T>::zeros(n);
    if A[(0, 0)].is_zero() {
        return Err(Error::DivBy0);
    }
    x[0] = b[0]/A[(0, 0)];
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
    }
    Ok(x)
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
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        let mults = Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .par_apply_collect(|mut row_i| {
                let mul = row_i[0]/row_k[0];
                Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
                mul
            });
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
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        let mults = Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .par_apply_collect(|mut row_i| {
                let mul = row_i[0]/row_k[0];
                Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
                mul
            });
        println!("k: {}\nU: \n{}\nmults:\n{}\n----------", k, new_m, mults);
    }
    Ok(new_m)
}

pub fn elimination_with_total_pivoting(m: &Array2<f64>) -> Result<Array2<f64>, Error> {
    let n = m.nrows();
    let mut new_m = Array2::<f64>::zeros((n, n));
    new_m.clone_from(m);
    let mut marks = Array1::<f64>::linspace(0., (n - 1) as f64, n);
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
        let row_k = Zip::from(new_m.slice_mut(s![k, k..])).par_apply_collect(|x| *x);
        let mults = Zip::from(new_m.slice_mut(s![k+1.., k..]).genrows_mut())
            .par_apply_collect(|mut row_i| {
                let mul = row_i[0]/row_k[0];
                Zip::indexed(row_i).par_apply(|j, mut e| *e -= mul * row_k[j]);
                mul
            });
        println!("k: {}\nU: \n{}\nmults:\n{}\nmarks:\n{}\n----------", k, new_m, mults, marks);
    }
    Ok(new_m)
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

