/*

Firefly algorithm

Single-threaded implementation, proposed by Xin-She Yang in "Nature-Inspired Metaheuristic
Algorithms", 2008, ISBN ISBN 978-1-905986-10-1.

Algorithm is roughly inspired by behaviours of fireflies, a light-emitting beetles of Coleoptera
order. They emit light during twilights as a honest warning signal that they are distasteful.

A single firefly represents a possible - but not necessarily optimal - solution.

Algorithm requires a function referred to as an objective function; value of that function is
directly proportional to firefly's brightness in maximization problems and inversely proportional in
minimization problems. That value dictates firefly's brightness.

To begin, we initialize our fireflies' population by randomly placing them within area restricted by
bound parameters provided, then calculate each firefly's brightness via provided objective function.

The optimization proper can be described in pseudocode as following:

(I x denotes light intensity of given x)

while (t < MaxGeneration)
        for i = 1 : n (all n fireflies)
            for j = 1 : i (n fireflies)
                if ( I j > I i ),
                    move firefly i towards j;
                    Evaluate new solutions and update light intensity;
                end if
            end for j
        end for i
        Rank fireflies and find the current best;
    end while

Note that the number of objective function evaluations per loop is one evaluation per firefly, even
though the above pseudocode suggests it is n×n. (Based on Yang's MATLAB code.) Thus the total number
of objective function evaluations is (number of generations) × (number of fireflies).

A distinctive feature of the firefly algorithm is the phenomenon of automatic division. Since
attractiveness of solution decreases with distance, which - if reasonable light absorption
coefficient is employed - results in separation of population into subgroups, with each subgroup
centered around a different local optimum. Average distance between two adjacent firefly groups can
be expressed as -f64::powf(f64::sqrt(GAMMA), -1).

This property can be easily visualised by optimizing f(x,y) = (|x| + |y|)e^(-x^2-y^2), with gamma
parameter of 0.98. There are four local maximums, located in (+- 0.5, +- 0.5). Algorithm
approximates local optimums within ~ 30 generations, with population counting 25.

Complexity of the algorithm is O(t*n^2), t - no of iterations, n - no of fireflies. For relatively
large n, double loop should be replaced with a single loop, and the attractiveness of fireflies
sorted by a comparative algorithm of choice, bronging complexity down to O(n*t*log(n)).

Firefly algorithm has found widespread use in analysis of equilibrium states in chemistry, and as an
effective solution to semantic syntax problems in computer science.

*/

use std::f64;
use rand::{Rng, thread_rng};

pub mod probe;
pub mod auxiliary;

use probe::Probe;


pub struct FireflyAlgorithmCfg {
    dimensions: u8,
    //Nr of dimensions
    lower_bound: f64,
    //Lower search bound
    upper_bound: f64,
    //Upper search bound
    max_generations: u32,
    //Maximum amount of generations
    population_size: u32,
    //Population size
    alfa0: f64,
    //Initial randomness coefficient
    beta0: f64,
    //Attractiveness coefficient, in most cases leave as 1
    gamma: f64,
    //Light absorption coefficient
    delta: f64,
    //Randomness decrease modifier, 0<delta<1
}

impl Default for FireflyAlgorithmCfg {
    fn default() -> Self {
        FireflyAlgorithmCfg {
            dimensions: 2,
            lower_bound: -5.0,
            upper_bound: 5.0,
            max_generations: 1000,
            population_size: 25,
            alfa0: 1.0,
            beta0: 1.0,
            gamma: 0.01,
            delta: 0.97,
        }
    }
}

pub struct FireflyAlgorithm {
    pub config: FireflyAlgorithmCfg,
    pub brightness_function: fn(&Vec<f64>) -> f64,
    pub probe: Box<dyn Probe>,

}

impl FireflyAlgorithm {
    fn new(config: FireflyAlgorithmCfg, brightness_function: fn(&Vec<f64>) -> f64, probe: Box<dyn Probe>) -> Self {
        FireflyAlgorithm {
            config,
            brightness_function,
            probe,
        }
    }

    pub fn execute(&mut self) {
        self.probe.on_start();
        let mut population: Vec<Vec<f64>> = Vec::new();
        for _index in 0..self.config.population_size as usize { //Generate initial population
            let mut temp: Vec<f64> = Vec::new();
            for _dim in 0..self.config.dimensions {
                temp.push(thread_rng().gen_range(self.config.lower_bound as f64..self.config.upper_bound as f64));
            }
            population.push(temp);
        }
        let mut brightness: Vec<f64> = Vec::new();
        let temp = population.clone();
        for point in temp {
            brightness.push(1 as f64 / (self.brightness_function)(&point)); //TODO USUŃ TEMP CLONEA
        }
        let scale = self.config.upper_bound - self.config.lower_bound;
        let mut alfa = self.config.alfa0;
        let mut rng = thread_rng();
        let mut currentbest: f64 = f64::MAX;
        for generation in 0..self.config.max_generations {
            if generation % 25 == 0 {
                self.probe.on_iteration_start(&generation)
            }
            for index in 0 as usize..self.config.population_size as usize {
                for innerindex in 0 as usize..self.config.population_size as usize {
                    if brightness[index] < brightness[innerindex] {
                        let const1 = self.config.beta0 * f64::powf(f64::consts::E, -1 as f64 * self.config.gamma * f64::powi(distance(&population[index], &population[innerindex]), 2));
                        for dimension in 0 as usize..self.config.dimensions as usize {
                            population[index][dimension] += const1 * (population[innerindex][dimension] - population[index][dimension]) + self.config.alfa0 * alfa * (rng.gen_range(0.01..0.99)/*TODO DODAJ SETTING*/ - 0.5) * scale;
                        }
                        brightness[index] = 1 as f64 / (self.brightness_function)(&population[index]);
                    }
                }
            }
            alfa = alfa * self.config.delta;
            if generation % 25 == 0 { //TODO REFACTOR
                let mut maxpos = 0;
                let mut maxbright = 0 as f64;
                for index in 0 as usize..self.config.population_size as usize {
                    if brightness[index] == f64::INFINITY {
                        maxpos = index;
                        break;
                    }
                    if brightness[index] > maxbright {
                        maxbright = brightness[index];
                        maxpos = index;
                    }
                }
                if (self.brightness_function)(&population[maxpos]) < currentbest {
                    self.probe.on_new_best(&(self.brightness_function)(&population[maxpos]));
                    currentbest = (self.brightness_function)(&population[maxpos]);
                } else {
                    self.probe.on_current_best();
                }
                //println!("Gen: {}, x: {}, y: {}", generation, population[maxpos][0], population[maxpos][1]);
            }
            if generation % 25 == 0 {
                //self.probe.on_iteration_end(&generation); //TODO CHYBA TEGO NIE POTRZEBUJĘ
                println!();//TODO PO PROSTU WYPISZĘ NEWLINE USUŃ TO
            }
        }
        self.probe.on_end();
    }
}

pub fn distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 { //Distance between two points
    let mut res: f64 = 0 as f64;
    for dimension in 0..a.len() {
        res += f64::powi(a[dimension] - b[dimension], 2)
    }
    f64::sqrt(res)
}
