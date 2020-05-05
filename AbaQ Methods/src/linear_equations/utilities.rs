use ndarray::{Array2, Zip};
use ndarray::prelude::*;
use ndarray::parallel::prelude::*;
use std::mem;

pub enum Error {
    DivBy0,
    BadIn,
    MultipleSolution,
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