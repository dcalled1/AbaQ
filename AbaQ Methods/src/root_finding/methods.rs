use num_traits::{Float, abs};
use crate::root_finding::register::Logbook;
use crate::root_finding::utilities::{Optimistic, Pessimistic, Error, calc_error};


/*
pub trait RunnableMethod {
    fn run(&mut self) -> Logbook;
    fn run_with_no_log(&mut self);
    fn next(&mut self) -> Option<f64>;
    fn pursue(&mut self) -> bool;
}*/

//Methods

pub fn incremental_search(f: fn(f64)->f64, x0: f64, dx: f64, n:u32)
    -> Result<(Option<f64>, Option<(f64, f64)>, Optimistic), Pessimistic> {
    let (mut xa, mut xb) = (x0, x0 + dx);
    let (mut ya, mut yb) = (f(xa), f(xb));

    let mut i = 0u32;
    while ya * yb > 0f64 && i < n {
        xa = xb;
        xb = xa + dx;
        ya = f(xa);
        yb = f(xb);
        i += 1;
    }
    if ya == 0f64 {
        return Ok((Some(xa), None, Optimistic::RootFound));
    }
    if yb == 0f64 {
        return Ok((Some(xb), None, Optimistic::RootFound));
    }
    if ya * yb < 0f64 {
        return Ok((None, Some((xa, xb)), Optimistic::IntervalFound));
    }
    Err(Pessimistic::MaxIterationsReached)
}

pub fn bisection(f: fn(f64)->f64, _xu: f64, _xl: f64, tol: f64, n:u32, error_type: Error)
                          -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xl".into(), "xm".into(), "xu".into(),
                                           "yl".into(), "ym".into(), "yu".into(), "error".into()]);
    let (mut xu, mut xl) = if _xu > _xl {
        (_xu, _xl)
    } else {
        (_xl, _xu)
    };
    let (mut yu, mut yl) = (f(xu), f(xl));
    if yu == 0f64 {
        return (Ok((xu, 0, Optimistic::RootFound)), logbook);
    }
    if yl == 0f64 {
        return (Ok((xl, 0, Optimistic::RootFound)), logbook);
    }
    if yu*yl > 0f64 {
        return (Err(Pessimistic::InvalidInput), logbook);
    }
    let mut xm = (xl + xu)/2f64;
    let (mut err, mut ym): (f64, f64) = (Float::infinity(), f(xm));
    let mut i = 1u32;
    logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    let mut xaux: f64 = 0f64;
    while ym != 0f64 && err > tol && i < n {
        if yl * ym < 0f64 {
            xu = xm;
            yu = ym;
        } else {
            xl = xm;
            yl = ym;
        }
        xaux = xm;
        xm = (xl + xu)/2f64;
        ym = f(xm);
        err = calc_error(xm, xaux, error_type);
        i += 1;
        logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    }
    if ym == 0f64 {
        return (Ok((xm, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xm, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn false_position(f: fn(f64)->f64, _xu: f64, _xl: f64, tol: f64, n:u32, error_type: Error)
                 -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xl".into(), "xm".into(), "xu".into(),
                                           "yl".into(), "ym".into(), "yu".into(), "error".into()]);
    let (mut xu, mut xl) = if _xu > _xl {
        (_xu, _xl)
    } else {
        (_xl, _xu)
    };
    let (mut yu, mut yl) = (f(xu), f(xl));
    if yu == 0f64 {
        return (Ok((xu, 0, Optimistic::RootFound)), logbook);
    }
    if yl == 0f64 {
        return (Ok((xl, 0, Optimistic::RootFound)), logbook);
    }
    if yu*yl > 0f64 {
        return (Err(Pessimistic::InvalidInput), logbook);
    }
    let mut xm = xl - yl * (xu - xl)/(yu - yl);
    let (mut err, mut ym): (f64, f64) = (Float::infinity(), f(xm));
    let mut i = 1u32;
    logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    let mut xaux: f64 = 0f64;
    while ym != 0f64 && err > tol && i < n {
        if yl * ym < 0f64 {
            xu = xm;
            yu = ym;
        } else {
            xl = xm;
            yl = ym;
        }
        xaux = xm;
        xm = xl - yl * (xu - xl)/(yu - yl);
        ym = f(xm);
        err = if error_type == Error::Relative && abs(xm) > Float::epsilon() {
            abs((xm - xaux)/xm)
        } else {
            abs(xm - xaux)
        };
        i += 1;
        logbook.registry(i, vec![xl, xm, xu, yl, ym, yu, err]);
    }
    if ym == 0f64 {
        return (Ok((xm, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xm, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn fixed_point(f: fn(f64)->f64, g: fn(f64)->f64, _xa: f64,  tol: f64, n:u32, error_type: Error)
                      -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, err]);
    let mut xn = 0f64;
    while y != 0f64 && err > tol && i < n {
        xn = g(xa);
        y = f(xn);
        err = if error_type == Error::Relative && abs(xn) > Float::epsilon() {
            abs((xn - xa)/xn)
        } else {
            abs(xn - xa)
        };
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

pub fn newton(f: fn(f64)->f64, df: fn(f64)->f64, _xa: f64,  tol: f64, n:u32, error_type: Error)
                   -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "y\'i".into(), "error".into()]);
    let mut xa = _xa;
    let (mut y, mut dy) = (f(xa), df(xa));
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, dy, err]);
    let mut xn = 0f64;
    while y != 0f64 && dy != 0f64 && err > tol && i < n {
        xn = xa - y/dy;
        y = f(xn);
        dy = df(xn);
        err = if error_type == Error::Relative && abs(xn) > Float::epsilon() {
            abs((xn - xa)/xn)
        } else {
            abs(xn - xa)
        };
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, dy, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if dy == 0f64 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}

pub fn secant(f: fn(f64)->f64, _x0: f64, _x1: f64,  tol: f64, n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let (mut x0, mut x1) = (_x0, _x1);
    let mut y0 = f(x0);
    if y0 == 0f64 {
        return (Ok((x0, 0, Optimistic::RootFound)), logbook)
    }
    let mut y1 = f(x1);
    let mut err: f64 = Float::infinity();
    let mut i = 1u32;
    let mut x2 = 0f64;
    logbook.registry(0, vec![x0, y0, err]);
    logbook.registry(1, vec![x1, y1, err]);
    while y1 != 0f64 && y1 != y0 && err > tol && i < n {
        x2 = x1 - y1 * (x1 - x0)/(y1 - y0);
        err = if error_type == Error::Relative && abs(x2) > Float::epsilon() {
            abs((x2 - x1)/x2)
        } else {
            abs(x2 - x1)
        };
        x0 = x1;
        y0 = y1;
        x1 = x2;
        y1 = f(x1);
        i += 1;
        logbook.registry(i, vec![x1, y1, err]);
    }
    if y1 == 0f64 {
        return (Ok((x1, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x1, i, Optimistic::RootApproxFound)), logbook);
    }
    if y0 == y1 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn multiple_root(f: fn(f64)->f64, df: fn(f64)->f64, d2f: fn(f64)->f64,_xa: f64,  tol: f64,
                     n:u32, error_type: Error)
                     -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "y\'i".into(),
                                           "y\'\'i".into(), "error".into()]);
    let mut xa = _xa;
    let (mut y, mut dy, mut d2y) = (f(xa), df(xa), d2f(xa));
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, dy, d2y, err]);
    let mut xn = 0f64;
    while y != 0f64 && dy.powi(2) != y * d2y && err > tol && i < n {
        xn = xa - y * dy /(dy.powi(2) - y * d2y);
        y = f(xn);
        dy = df(xn);
        d2y = d2f(xn);
        err = if error_type == Error::Relative && abs(xn) > Float::epsilon() {
            abs((xn - xa)/xn)
        } else {
            abs(xn - xa)
        };
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, dy, d2y, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if dy.powi(2) == y * d2y {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn steffensen(f: fn(f64)->f64, _xa: f64,  tol: f64, n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "z\'i".into(),
                                           "error".into()]);
    let mut xa = _xa;
    let mut y = f(xa);
    let mut z = f(xa + y);
    let mut err: f64 = Float::infinity();
    let mut i = 0u32;
    logbook.registry(i, vec![xa, y, z, err]);
    let mut xn = 0f64;
    while y != 0f64 && y != z && err > tol && i < n {
        xn = xa - y.powi(2)/(z - y);
        y = f(xn);
        z = f(xn + y);
        err = if error_type == Error::Relative && abs(xn) > Float::epsilon() {
            abs((xn - xa)/xn)
        } else {
            abs(xn - xa)
        };
        xa = xn;
        i += 1;
        logbook.registry(i, vec![xa, y, z, err]);
    }
    if y == 0f64 {
        return (Ok((xa, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((xa, i, Optimistic::RootApproxFound)), logbook);
    }
    if y == z {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}


pub fn muller(f: fn(f64)->f64, _x0: f64, _x1: f64, _x2: f64, tol: f64, n:u32, error_type: Error)
              -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let (mut x0, mut x1, mut x2) = (_x0, _x1, _x2);
    let mut y0 = f(x0);
    if y0 == 0f64 {
        return (Ok((x0, 0, Optimistic::RootFound)), logbook)
    }
    let mut y1 = f(x1);
    if y1 == 0f64 {
        return (Ok((x1, 0, Optimistic::RootFound)), logbook)
    }
    let mut y2 = f(x2);
    let (mut h0, mut h1) = (x1 - x0, x2 - x1);
    if h0 == 0f64 || h1 == 0f64 {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let (mut d0, mut d1) = ((y1 - y0)/h0, (y2 - y2)/h1);
    if abs(h1 + h0) < Float::epsilon() {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let mut a = (d1 - d0)/(h1 + h0);
    let mut b = a * h1 + d1;
    let mut err: f64 = Float::infinity();
    let mut i = 2u32;
    logbook.registry(0, vec![x0, y0, err]);
    logbook.registry(1, vec![x1, y1, err]);
    logbook.registry(2, vec![x2, y2, err]);
    let mut x3 = 0f64;
    while y2 != 0f64 && b.powi(2) >= 4f64 * a * y2
        && x0 != x1 && x1 != x2 && x0 != x2 && err > tol && i < n {
        x3 = if b < 0f64 {
            x2 + 2f64 * y2 / (b - (b.powi(2) - 4f64 * a * y2).sqrt())
        } else {
            x2 + 2f64 * y2 / (b + (b.powi(2) - 4f64 * a * y2).sqrt())
        };
        err = if error_type == Error::Relative && abs(x3) > Float::epsilon() {
            abs((x3 - x2)/x3)
        } else {
            abs(x3 - x2)
        };
        x0 = x1;
        y0 = y1;
        x1 = x2;
        y1 = y2;
        x2 = x3;
        y2 = f(x2);
        i += 1;
        logbook.registry(i, vec![x2, y2, err]);
    }
    if y1 == 0f64 {
        return (Ok((x2, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x2, i, Optimistic::RootApproxFound)), logbook);
    }
    if x0 == x1 || x1 == x2 || x0 == x2 {
        return (Err(Pessimistic::MultipleRoot), logbook);
    }
    if b.powi(2) < 4f64 * a * y2 {
        return (Err(Pessimistic::ComplexRoot), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)
}

pub fn accelerated_fixed_point(f: fn(f64)->f64, g: fn(f64)->f64, _x0: f64,
                               tol: f64, n:u32, error_type: Error)
                                -> (Result<(f64, u32, Optimistic), Pessimistic>, Logbook) {
    let mut logbook = Logbook::new(n, vec!["xi".into(), "yi".into(), "error".into()]);
    let mut x0 = _x0;
    let mut x1 = g(x0);
    let mut x2 = g(x1);
    logbook.registry(0, vec![x0, f(x0), Float::infinity()]);
    logbook.registry(1, vec![x1, f(x1), if error_type == Error::Relative && abs(x1) > Float::epsilon() {
                    abs((x1 - x0)/x1)
                } else {
                    abs(x1 - x0)
                }]);
    logbook.registry(2, vec![x2, f(x2), if error_type == Error::Relative && abs(x2) > Float::epsilon() {
                    abs((x2 - x1)/x2)
                } else {
                    abs(x2 - x1)
                }]);
    if x2 + x0 == 2f64 * x1 {
        return (Err(Pessimistic::DivBy0), logbook);
    }
    let mut x3 = x2 - (x2 - x1).powi(2)/(x2 - 2f64 * x1 + x0);
    let mut y = f(x3);
    let mut err: f64 = if error_type == Error::Relative && abs(x3) > Float::epsilon() {
        abs((x3 - x2)/x3)
    } else {
        abs(x3 - x2)
    };
    let mut i = 3u32;
    logbook.registry(i, vec![x3, y, err]);
    while y != 0f64 && err > tol && i < n {
        x0 = x3;
        x1 = g(x0);
        x2 = g(x1);
        if x2 + x0 == 2f64 * x1 {
            return (Err(Pessimistic::DivBy0), logbook);
        }
        x3 = x2 - (x2 - x1).powi(2)/(x2 - 2f64 * x1 + x0);
        y = f(x3);
        err = if error_type == Error::Relative && abs(x3) > Float::epsilon() {
            abs((x3 - x2)/x3)
        } else {
            abs(x3 - x2)
        };
        i += 1;
        logbook.registry(i, vec![x3, y, err]);
    }
    if y == 0f64 {
        return (Ok((x3, i, Optimistic::RootFound)), logbook);
    }
    if err <= tol {
        return (Ok((x3, i, Optimistic::RootApproxFound)), logbook);
    }
    (Err(Pessimistic::MaxIterationsReached), logbook)

}

/*
//Incremental Search

#[derive(Debug)]
pub struct IncrementalSearch {
    xa: f64,
    xb: f64,
    ya: f64,
    yb: f64,
    f: fn(f64) -> f64,
    n: u32,
    i: u32,
    dx: f64,
    stop: Option<StopCause>,
}

impl IncrementalSearch {
    pub fn new(f: fn(f64)->f64, x0: f64, dx: f64, n:u32) -> IncrementalSearch {
        let (xb, ya) = (x0+dx, f(x0));
        let yb = f(xb);
        IncrementalSearch{xa: x0, xb, f, n, dx, i: 0, ya, yb, stop: None}
    }
}

impl RunnableMethod for IncrementalSearch {
    fn run(&mut self) -> Logbook {
        let mut ended = false;
        let mut log = Logbook::new(self.n);
        while !ended {
            ended = self.pursue();
            log.registry(self.i, vec![self.xa, self.ya, self.xb, self.yb]);
        }
        log
    }

    fn run_with_no_log(&mut self) {
        let mut ended = false;
        while !ended {
            ended = self.pursue();
        }
    }

    fn next(&mut self) -> Option<f64> {
        if self.stop != None {return None}
        if self.ya*self.yb < 0f64 {
            self.stop = Some(IntervalFound);
            return None
        }
        if self.i >= self.n {
            self.stop = Some(MaxIterationsReached);
            return None
        }
        self.xa = self.xb;
        self.ya = self.yb;
        self.xb = self.xa + self.dx;
        self.yb = (self.f)(self.xb);
        self.i += 1;
        Some(self.xb)
    }
    fn pursue(&mut self) -> bool {
        if self.next() == None {
            return true
        }
        false
    }

}


//Bisection

#[derive(Debug)]
pub struct Bisection {
    xu: f64,
    xl: f64,
    xm: f64,
    yu: f64,
    yl: f64,
    ym: f64,
    f: fn(f64) -> f64,
    n: u32,
    i: u32,
    err: f64,
    tol: f64,
    error_type: Error,
    stop: Option<StopCause>,
}

impl Bisection {
    pub fn new(f: fn(f64)->f64, _xu: f64, _xl: f64, tol: f64, n:u32, error_type: Error) -> Bisection {
        let (xu, xl) = if _xu < _xl {
            (_xl, _xu)
        } else {
            (_xu, _xl)
        };
        let (yu, yl) = (f(xu), f(xl));
        let xm = (xu + xl)/2f64;
        let ym = f(xm);
        let status = if yu * yl > 0f64 {
            Some(StopCause::InvalidInput)
        } else if yu == 0f64 {
            Some(StopCause::RootFound)
        } else if yl == 0f64 {
            Some(StopCause::RootFound)
        } else {
            None
        };
        Bisection{xu, xl, f, n, i: 0, yu, yl,xm, ym,
            stop: status, tol: abs(tol), err: Float::infinity(), error_type}
    }
}

impl RunnableMethod for Bisection {
    fn run(&mut self) -> Logbook {
        let mut ended = false;
        let mut log = Logbook::new(self.n);
        while !ended {
            log.registry(self.i, vec![self.xu, self.yu, self.xm, self.ym, self.xl, self.yl, self.err]);
            ended = self.pursue();
        }
        log
    }

    fn run_with_no_log(&mut self) {
        let mut ended = false;
        while !ended {
            ended = self.pursue();
        }
    }

    fn next(&mut self) -> Option<f64> {
        if self.stop != None {
            return None;
        }
        if self.ym == 0f64 {
            self.stop = Some(RootFound);
            return None
        }
        if self.err < self.tol {
            self.stop = Some(RootApproxFound);
            return None
        }
        if self.i >= self.n {
            self.stop = Some(MaxIterationsReached);
            return None
        }

        let ans: f64;
        if self.yl * self.ym < 0f64 {
            self.xu = self.xm;
            self.yu = self.ym;

        } else {
            self.xl = self.xm;
            self.yl = self.ym;
        }
        self.xm = (self.xu + self.xl)/2f64;
        self.ym = (self.f)(self.xm);

        self.i += 1;
        Some(self.xm)
    }
    fn pursue(&mut self) -> bool {
        if self.next() == None {
            return true
        }
        false
    }

} */