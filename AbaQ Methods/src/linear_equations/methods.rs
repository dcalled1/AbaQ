use nalgebra::base::{DMatrix, DVector};
use num_traits::{Zero};
use nalgebra::{ComplexField};
use crate::linear_equations::utilities::Error;

fn back_substitution<T: ComplexField>(A: DMatrix<T>, b: DVector<T>) -> Result<DVector<T>, Error> {
    let n = b.len();
    if n != A.ncols() || n != A.nrows() {
        return Err(Error::BadIn);
    }
    let mut x = DVector::zeros(n);
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

fn forward_substitution<T: ComplexField>(A: DMatrix<T>, b: DVector<T>) -> Result<DVector<T>, Error> {
    let n = b.len();
    if n != A.ncols() || n != A.nrows() {
        return Err(Error::BadIn);
    }
    let mut x = DVector::zeros(n);
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

