use feos_core::EosError;
use quantity::QuantityError;
use std::num::ParseFloatError;
use thiserror::Error;

mod dataset;
pub use dataset::DataSet;
// mod binary_vle;
mod estimator;
mod loss;
pub use loss::Loss;
mod vapor_pressure;
mod liquid_density;
mod viscosity;
mod thermal_conductivity;
mod diffusion;

#[derive(Debug, Error)]
pub enum EstimatorError {
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
