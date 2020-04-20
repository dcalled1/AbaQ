use num_traits::{abs, Float};

#[derive(Debug, Copy, Clone)]
pub enum Pessimistic {
    MaxIterationsReached,
    DivBy0,
    FunctionOutOfDomain,
    ComplexRoot,
    MultipleRoot,
    InvalidInput,
    InvalidFunction,
}

#[derive(Debug, Copy, Clone)]
pub enum Optimistic {
    RootFound,
    RootApproxFound,
    IntervalFound,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Error {
    Absolute,
    Relative
}

pub(crate) fn calc_error(x_prev: f64, x_act: f64, error_type: Error) -> f64 {
    if error_type == Error::Relative && abs(x_act) > Float::epsilon() {
        abs((x_act - x_prev)/x_act)
    } else {
        abs(x_act - x_prev)
    }
}


