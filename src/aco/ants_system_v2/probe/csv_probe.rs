use serde::Serialize;

use crate::aco::ants_system_v2::probe::Probe;
use crate::aco::ants_system_v2::Solution;

#[derive(Serialize)]
struct BestSolutionRecord{
    from: usize,
    to: usize,
    iter: usize,
}
#[derive(Serialize)]
struct FMatrixRecord {
    from: usize,
    to: usize,
    iter: usize,
    val: f64
}

pub struct CsvProbe {
    iteration: usize,
    best_sols: Vec<BestSolutionRecord>,
    pher: Vec<FMatrixRecord>,
    best_sol: Solution
}

impl CsvProbe {
    pub fn new() -> CsvProbe {
        CsvProbe {
            iteration: 0,
            best_sols: vec![],
            pher: vec![],
            best_sol: Solution::default()
        }
    }

    pub fn flush(&mut self) {
        let mut wrt = csv::WriterBuilder::new().from_path("best.csv").unwrap();

        for record in self.best_sols.iter() {
            wrt.serialize(record).expect("Could not serialize record");
        }
        wrt.flush().expect("Could not flush data");

    }
}

impl Probe for CsvProbe {
    fn on_new_best(&mut self, best_sol: &Solution) {
        self.best_sol = best_sol.clone();
    }

    fn on_current_best(&mut self, best: &Solution) {

    }

    fn on_iteration_start(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn on_iteration_end(&mut self, iteration: usize) {
        for (i, row) in self.best_sol.matrix.row_iter().enumerate() {
            for (j, val ) in row.iter().enumerate() {
                if *val < 0.5 {continue;}

                self.best_sols.push(BestSolutionRecord {
                    from: i,
                    to: j,
                    iter: iteration
                })
            }
        }
    }

    fn on_end(&mut self) {
        self.flush();
    }
}