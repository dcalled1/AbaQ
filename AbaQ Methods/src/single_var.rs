use crate::single_var::StopCause::{IntervalFound, MaxIterationsReached};
use num_traits::{Float, abs};

#[derive(Debug)]
pub enum StopCause {
    RootFound,
    RootApproxFound,
    MaxIterationsReached,
    DivBy0,
    FunctionOutOfDomain,
    IntervalFound,
    ComplexRoot,
    MultipleRoot,
}

#[derive(Debug)]
pub enum Error {
    Absolute,
    Relative
}


pub struct Log {
    vars: Vec<f64>,
    i: u32
}
pub struct Logbook {
    regs: Vec<Log>
}

impl Log {
    fn new(i: u32, vars: Vec<f64>) -> Log {
        Log{i, vars}
    }
}

impl Logbook {
    fn new(n: u32) -> Logbook {
        Logbook{regs: Vec::with_capacity(n as usize)}
    }

    fn registry(&mut self, i: u32, vars: Vec<f64>) {
        self.regs.push(Log::new(i, vars));
    }
}

pub trait RunnableMethod {
    fn run(&mut self) -> Logbook;
    fn run_with_no_log(&mut self);
    fn next(&mut self) -> Option<f64>;
    fn pursue(&mut self) -> bool;
}


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
//TODO

#[derive(Debug)]
pub struct Bisection {
    xa: f64,
    xb: f64,
    ya: f64,
    yb: f64,
    f: fn(f64) -> f64,
    n: u32,
    i: u32,
    err: f64,
    tol: f64,
    error_type: Error,
    stop: Option<StopCause>,
}

impl Bisection {
    pub fn new(f: fn(f64)->f64, xa: f64, xb: f64, tol: f64, n:u32, error_type: Error) -> Bisection {
        let (ya, yb) = (f(x0), f(xb));
        Bisection{xa, xb, f, n, i: 0, ya, yb, stop: None, tol: abs(tol), err: Float::infinity(), error_type}
    }
}

impl RunnableMethod for Bisection {
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