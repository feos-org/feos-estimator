//! The [`Estimator`] struct can be used to store multiple [`DataSet`]s for convenient parameter
//! optimization.
use super::dataset::*;
use super::FitError;
use feos_core::EosUnit;
use feos_core::EquationOfState;
use ndarray::{arr1, concatenate, Array1, ArrayView1, Axis};
use quantity::QuantityScalar;
use std::fmt;
use std::fmt::Write;
use std::rc::Rc;

/// A collection of [`DataSet`]s and weights that can be used to
/// evaluate an equation of state versus experimental data.
pub struct Estimator<U: EosUnit, E: EquationOfState> {
    data: Vec<Rc<dyn DataSet<U, E>>>,
    weights: Vec<f64>,
}

impl<U: EosUnit, E: EquationOfState> Estimator<U, E>
where
    QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
{
    /// Create a new `Estimator` given `DataSet`s and weights.
    ///
    /// The weights are normalized and used as multiplicator when the
    /// cost function across all `DataSet`s is evaluated.
    pub fn new(data: Vec<Rc<dyn DataSet<U, E>>>, weights: Vec<f64>) -> Self {
        Self { data, weights }
    }

    /// Add a `DataSet` and its weight.
    pub fn add_data(&mut self, data: &Rc<dyn DataSet<U, E>>, weight: f64) {
        self.data.push(data.clone());
        self.weights.push(weight);
    }

    /// Returns the cost of each `DataSet`.
    ///
    /// Each cost contains the inverse weight.
    pub fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError> {
        let predictions: Result<Vec<Array1<f64>>, FitError> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let w_sum = self.weights.iter().sum::<f64>();
                let w = arr1(&self.weights) / w_sum;
                Ok(d.cost(eos)? * w[i])
            })
            .collect();
        if let Ok(p) = predictions {
            let aview: Vec<ArrayView1<f64>> = p.iter().map(|pi| pi.view()).collect();
            Ok(concatenate(Axis(0), &aview)?)
        } else {
            Err(FitError::IncompatibleInput)
        }
    }

    /// Returns the stored `DataSet`s.
    pub fn datasets(&self) -> Vec<Rc<dyn DataSet<U, E>>> {
        self.data.to_vec()
    }
}
