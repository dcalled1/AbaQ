use crate::single_var::RunnableMethod;

mod single_var;

fn main() {
    let mut is = single_var::IncrementalSearch::new(|x: f64| x*2f64, -5f64, 0.3f64, 70);
    is.run();
    println!("{:?}",&is)
}
