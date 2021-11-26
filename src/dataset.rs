//! The [`DataSet`] trait provides routines that can be used for
//! optimization of parameters of equations of state given
//! a `target` which can be values from experimental data or
//! other models.
use crate::FitError;
use feos_core::EosUnit;
use feos_core::{Contributions, DensityInitialization, State};
use feos_core::{EquationOfState, MolarWeight};
use feos_core::{PhaseEquilibrium, VLEOptions};
use ndarray::{arr1, Array1};
use quantity::{Quantity, QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::fmt::{self, LowerExp};
use std::rc::Rc;

/// Utilities for working with experimental data.
///
/// Functionalities in the context of optimizations of
/// parameters of equations of state.
pub trait DataSet<U: EosUnit, E: EquationOfState>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    /// Evaluate the cost function.
    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>;
}
