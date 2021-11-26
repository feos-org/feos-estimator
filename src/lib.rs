use feos_core::EosError;
use ndarray::{arr1, concatenate, Array1, ArrayView1, Axis};
use quantity::{QuantityError, QuantityScalar};
use std::fmt;
use std::fmt::Write;
use std::num::ParseFloatError;
use std::rc::Rc;
use thiserror::Error;

mod dataset;
pub use dataset::DataSet;
mod binary_vle;
mod estimator;
mod vapor_pressure;

#[derive(Debug, Error)]
pub enum FitError {
    #[error("Missing input. Need '{needed}' to evaluate '{to_evaluate}'.")]
    MissingInput { needed: String, to_evaluate: String },
    #[error("Input has not the same amount of data as the target.")]
    IncompatibleInput,
    #[error("Keyword '{0}' unknown. Try: 'liquid density', 'vapor pressure', 'equilibrium liquid density'")]
    KeyError(String),
    #[error(transparent)]
    ShapeError(#[from] ndarray::ShapeError),
    #[error(transparent)]
    ParseError(#[from] ParseFloatError),
    #[error(transparent)]
    QuantityError(#[from] QuantityError),
    #[error(transparent)]
    EosError(#[from] EosError),
}
