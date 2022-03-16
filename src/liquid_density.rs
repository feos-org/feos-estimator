use super::{DataSet, EstimatorError, Loss};
use feos_core::{
    DensityInitialization, EosUnit, EquationOfState, MolarWeight, PhaseEquilibrium, SolverOptions,
    State,
};
use ndarray::{arr1, Array1};
use quantity::{QuantityArray1, QuantityScalar};
use std::collections::HashMap;
use std::rc::Rc;

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct LiquidDensity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    datapoints: usize,
}

impl<U: EosUnit> LiquidDensity<U> {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
    ) -> Result<Self, EstimatorError> {
        let datapoints = target.len();
        Ok(Self {
            target,
            temperature,
            pressure,
            datapoints,
        })
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }

    /// Returns pressure of data points.
    pub fn pressure(&self) -> QuantityArray1<U> {
        self.pressure.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E> for LiquidDensity<U> {
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature", "pressure"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, EstimatorError> {
        let moles = arr1(&[1.0]) * U::reference_moles();
        let unit = self.target.get(0);
        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let state = State::new_npt(
                eos,
                self.temperature.get(i),
                self.pressure.get(i),
                &moles,
                DensityInitialization::Liquid,
            );
            if let Ok(s) = state {
                prediction.try_set(i, s.mass_density())?;
            } else {
                prediction.try_set(i, f64::NAN * unit)?;
            }
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = self.relative_difference(eos)?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m.insert("pressure".to_owned(), self.pressure());
        m
    }
}

/// Store experimental data of liquid densities and compare to the equation of state.
#[derive(Clone)]
pub struct EquilibriumLiquidDensity<U: EosUnit> {
    pub target: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
    extrapolate: bool,
}

impl<U: EosUnit> EquilibriumLiquidDensity<U> {
    /// A new data set for liquid densities with pressures and temperatures as input.
    pub fn new(
        target: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        extrapolate: bool,
    ) -> Result<Self, EstimatorError> {
        let datapoints = target.len();
        let max_temperature = temperature
            .to_reduced(U::reference_temperature())?
            .into_iter()
            .reduce(|a, b| a.max(b))
            .unwrap()
            * U::reference_temperature();
        Ok(Self {
            target,
            temperature,
            max_temperature,
            datapoints,
            extrapolate,
        })
    }

    /// Returns temperature of data points.
    pub fn temperature(&self) -> QuantityArray1<U> {
        self.temperature.clone()
    }
}

impl<U: EosUnit, E: EquationOfState + MolarWeight<U>> DataSet<U, E>
    for EquilibriumLiquidDensity<U>
{
    fn target(&self) -> QuantityArray1<U> {
        self.target.clone()
    }

    fn target_str(&self) -> &str {
        "equilibrium liquid density"
    }

    fn input_str(&self) -> Vec<&str> {
        vec!["temperature"]
    }

    fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let unit = self.target.get(0);
        let critical_point = State::critical_point(
            eos,
            None,
            Some(self.max_temperature),
            SolverOptions::default(),
        )?;
        let t_c = critical_point.temperature;
        let rho_c = critical_point.mass_density();

        let mut prediction = Array1::zeros(self.datapoints) * unit;
        for i in 0..self.datapoints {
            let t = self.temperature.get(i);
            if let Ok(state) = PhaseEquilibrium::pure_t(eos, t, None, SolverOptions::default()) {
                prediction.try_set(i, state.liquid().mass_density())?;
            } else {
                if self.extrapolate {
                    let tr = t.to_reduced(t_c).unwrap() - 1.0;
                    let extrapolation = if tr < f64::exp(-1.0) {
                        rho_c * (1.0 + tr * tr.ln())
                    } else {
                        rho_c * 0.62
                    };
                    prediction.try_set(i, extrapolation)?;
                } else {
                    prediction.try_set(i, f64::NAN * U::reference_pressure())?
                }
            }
        }
        Ok(prediction)
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = self.relative_difference(eos)?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
    }

    fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
        let mut m = HashMap::with_capacity(2);
        m.insert("temperature".to_owned(), self.temperature());
        m
    }
}
