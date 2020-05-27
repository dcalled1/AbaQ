use ndarray::{Array2, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::mem;
use ndarray_linalg::{Norm, c64};
use nalgebra::{ComplexField, Complex};

#[derive(Debug)]
pub enum Error {
    DivBy0,
    BadIn,
    MultipleSolution,
    ComplexNumber,
}

#[derive(PartialEq, Eq)]
pub enum FactorizationType {
    Cholesky,
    Crout,
    Doolittle,
}

pub enum IterationType {
    Jacobi,
    GaussSeidel,
    SOR(f64),
}

pub struct LUStage<T: ComplexField> {
    l: Array2<T>,
    u: Array2<T>,
    k: usize,
}

pub struct LUStages<T: ComplexField> {
    stages: Vec<LUStage<T>>,
    initial_matrix: Array2<f64>,
}

impl <T: ComplexField> LUStages<T> {
    pub fn new(m: &Array2<f64>) -> LUStages<f64> {
        LUStages {
            stages: Vec::<LUStage<f64>>::new(),
            initial_matrix: m.clone(),
        }
    }

    pub fn new_with_complex(m: &Array2<f64>) -> LUStages<c64> {
        LUStages {
            stages: Vec::<LUStage<c64>>::new(),
            initial_matrix: m.clone(),
        }
    }

    pub fn registry(&mut self, l: &Array2<T>, u: &Array2<T>, k: usize) {
        self.stages.push(LUStage{
            l: l.clone(),
            u: u.clone(),
            k
        })
    }
}

pub(crate) fn swap_rows(m: &mut Array2<f64>, a: usize, b: usize, start: usize) {
    let (mut ra, mut rb) = m.multi_slice_mut((s![a, start..], s![b, start..]));
    Zip::from(&mut ra).and(&mut rb).par_apply(|ea, eb| mem::swap(ea, eb));
}

pub(crate) fn swap_cols(m: &mut Array2<f64>, a: usize, b: usize, start: usize) {
    let (mut ca, mut cb) = m.multi_slice_mut((s![start.., a], s![start.., b]));
    Zip::from(&mut ca).and(&mut cb).par_apply(|ea, eb| mem::swap(ea, eb));
}

pub(crate) fn swap_entire_cols(m: &mut Array2<f64>, a: usize, b: usize) {
    let (mut ca, mut cb) = m.multi_slice_mut((s![.., a], s![.., b]));
    Zip::from(&mut ca).and(&mut cb).par_apply(|ea, eb| mem::swap(ea, eb));
}

pub fn spectral_radius(m: &Array2<f64>) -> Result<f64, Error> {
    if !m.is_square() {
        return Err(Error::BadIn);
    }
    let n = m.nrows();
    let mut b_k = Array1::<f64>::ones(n);
    let mut b_k1 = m.dot(&b_k);
    b_k = &b_k1 / b_k1.norm();
    for _ in 0..10 {
        b_k1 = m.dot(&b_k);
        b_k = &b_k1 / b_k1.norm();

    }
    Ok(b_k.dot(&b_k1) / b_k.dot(&b_k))
}



