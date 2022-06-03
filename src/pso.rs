use std::borrow::Borrow;
use itertools::iterate;
use num::{NumCast, One};

pub mod particle;
pub mod probe;
pub mod swarm;
pub mod util;

use crate::pso::probe::console_probe::ConsoleProbe;
use crate::pso::probe::csv_probe::CsvProbe;
use crate::pso::probe::json_probe::JsonProbe;
use crate::pso::probe::multi_probe::MultiProbe;
use crate::pso::probe::probe::Probe;
use crate::pso::swarm::Swarm;


struct PSOAlgorithmCfg {
    /**
    Parameters:
    dimensions: number of dimension of optimized function's domain
    lower_bound: lower bound of search area in every dimension of the domain
    upper_bound: upper bound of search area in every dimension of the domain
    particle_count: number of particles to use in optimization (number of particles will be maintained throughout the algorithm's run)
    inertia_weight: specifies how much particles retain their speed from previous iteration (0 - no speed retention, 1 - no slowdown)
    cognitive_coefficient: specifies how much particles are attracted their own best positions
    social_coefficient: specifies how much particles are attracted to entire swarm's best position
    function: function to be optimized
    iterations: number of iterations, the algorithm should run for
    log_interval: specifies how often algorithm's progress is logged
    probe: used for displaying results / progress of the algorithm

    Example values:
    inertia_weight: 0.5
    cognitive_coefficient: 1.0
    social_coefficient: 3.0
    **/
    dimensions: usize,
    lower_bound: f64,
    upper_bound: f64,
    particle_count: usize,
    inertia_weight: f64,
    cognitive_coefficient: f64,
    social_coefficient: f64,
    function: fn(&Vec<f64>) -> f64,
    iterations: usize,
    log_interval: usize,
    probe: Box<dyn Probe>
}

impl Default for PSOAlgorithmCfg {
    fn default() -> Self {
        PSOAlgorithmCfg {
            dimensions: 2,
            lower_bound: -10.0,
            upper_bound: 10.0,
            particle_count: 30,
            inertia_weight: 0.5,
            cognitive_coefficient: 1.0,
            social_coefficient: 3.0,
            function: rosenbrock,
            iterations: 500,
            log_interval: 10,
            probe:Box::new(ConsoleProbe::new())
        }
    }
}

struct PSOAlgorithm {
    config: PSOAlgorithmCfg,
    swarm: Swarm
}

impl PSOAlgorithm {
    fn new(config: PSOAlgorithmCfg) -> Self {
        let swarm = Swarm::generate(config.particle_count.clone(), config.dimensions.clone(), config.lower_bound.clone(), config.upper_bound.clone(), config.function.borrow());
        PSOAlgorithm {
            config,
            swarm
        }
    }

    fn execute(&mut self) {
        self.config.probe.on_begin(&self.swarm);
        for iteration in 0..self.config.iterations {
            self.swarm.update_velocities(&self.config.inertia_weight, &self.config.cognitive_coefficient, &self.config.social_coefficient);
            self.swarm.update_positions(&self.config.function);
            self.swarm.update_best_position(&self.config.function);
            if (iteration + 1) % self.config.log_interval == 0 {
                self.config.probe.on_new_generation(&self.swarm, iteration + 1);
            }
        }
        self.config.probe.on_end(&self.swarm);
    }
}

fn rosenbrock(x: &Vec<f64>) -> f64 {
    let _100: f64 = NumCast::from(100).unwrap();
    let mut value: f64 = f64::default();
    for i in 0..x.len() {
        if i == x.len() - 1 {
            break;
        }
        let x_curr: f64 = x[i];
        let x_next: f64 = x[i+1];
        value += _100 * (x_next - (x_curr * x_curr)) * (x_next - (x_curr * x_curr)) + (f64::one() - x_curr)*(f64::one() - x_curr);
    }
    return value;
}

pub fn pso_demo() {
    let iterations = 1000;

    let console_probe = Box::new(ConsoleProbe::new());
    let csv_probe = Box::new(CsvProbe::new("pso_demo.csv", iterations));
    let json_probe = Box::new(JsonProbe::new("pso_demo.json", iterations));
    let probes : Vec<Box<dyn Probe>> = vec![console_probe, csv_probe, json_probe];

    let config = PSOAlgorithmCfg{
        dimensions: 3,
        iterations,
        log_interval: 50,
        probe: Box::new(MultiProbe::new(probes)),
        ..PSOAlgorithmCfg::default()
    };

    let mut algorithm = PSOAlgorithm::new(config);

    algorithm.execute();
}