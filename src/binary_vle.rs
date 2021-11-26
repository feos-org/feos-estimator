use std::{fmt::LowerExp, rc::Rc};

use super::{DataSet, FitError};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, State, VLEOptions};
use ndarray::{arr1, Array1};
use ndarray_stats::QuantileExt;
use quantity::{Quantity, QuantityArray1, QuantityScalar};

enum CostFunctions {
    Pressure,
    ChemicalPotential,
    Distance,
}

/// Store experimental vapor pressure data and compare to the equation of state.
#[derive(Clone)]
pub struct BinaryTPx<U: EosUnit> {
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    liquid_molefracs: Array1<f64>,
    datapoints: usize,
}

impl<U: EosUnit> BinaryTPx<U> {
    pub fn new(
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        liquid_molefracs: Array1<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = temperature.len();
        Ok(Self {
            temperature,
            pressure,
            liquid_molefracs,
            datapoints,
        })
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (VLEOptions::default(), VLEOptions::default());

        // let reduced_temperatures = (0..self.datapoints)
        //     .map(|i| (self.temperature.get(i) * tc_inv).into_value().unwrap())
        //     .collect::<Vec<f64>>();

        // let prediction = &self.predict(eos)?;
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let prediction = PhaseEquilibrium::bubble_point_tx(
                eos,
                self.temperature.get(i),
                Some(self.pressure.get(i)),
                &arr1(&vec![xi, 1.0 - xi]),
                None,
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);

            cost[i] = ((self.pressure.get(i) - prediction) / self.pressure.get(i)).into_value()?
        }
        Ok(cost)
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryTPx<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError> {
        self.pressure_cost(eos)
    }
}

#[derive(Clone)]
pub struct BinaryTPy<U: EosUnit> {
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    vapor_molefracs: Array1<f64>,
    datapoints: usize,
}

impl<U: EosUnit> BinaryTPy<U> {
    pub fn new(
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        vapor_molefracs: Array1<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = temperature.len();
        Ok(Self {
            temperature,
            pressure,
            vapor_molefracs,
            datapoints,
        })
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (VLEOptions::default(), VLEOptions::default());

        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            let yi = self.vapor_molefracs[i];
            let prediction = PhaseEquilibrium::dew_point_tx(
                eos,
                self.temperature.get(i),
                Some(self.pressure.get(i)),
                &arr1(&vec![yi, 1.0 - yi]),
                None,
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);

            cost[i] = ((self.pressure.get(i) - prediction) / self.pressure.get(i)).into_value()?
        }
        Ok(cost)
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryTPy<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError> {
        self.pressure_cost(eos)
    }
}

#[derive(Clone)]
pub struct BinaryTPxy<U: EosUnit> {
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    liquid_molefracs: Array1<f64>,
    vapor_molefracs: Array1<f64>,
    datapoints: usize,
}

impl<U: EosUnit> BinaryTPxy<U> {
    pub fn new(
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        liquid_molefracs: Array1<f64>,
        vapor_molefracs: Array1<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = temperature.len();
        Ok(Self {
            temperature,
            pressure,
            liquid_molefracs,
            vapor_molefracs,
            datapoints,
        })
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (VLEOptions::default(), VLEOptions::default());

        let mut cost = Array1::zeros(2 * self.datapoints);
        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let yi = self.vapor_molefracs[i];

            let prediction_liquid = PhaseEquilibrium::bubble_point_tx(
                eos,
                self.temperature.get(i),
                Some(self.pressure.get(i)),
                &arr1(&vec![xi, 1.0 - xi]),
                Some(&arr1(&vec![yi, 1.0 - yi])),
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);

            let prediction_vapor = PhaseEquilibrium::dew_point_tx(
                eos,
                self.temperature.get(i),
                Some(self.pressure.get(i)),
                &arr1(&vec![yi, 1.0 - yi]),
                Some(&arr1(&vec![xi, 1.0 - xi])),
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);

            cost[i] =
                ((self.pressure.get(i) - prediction_liquid) / self.pressure.get(i)).into_value()?;
            cost[self.datapoints + i] =
                ((self.pressure.get(i) - prediction_vapor) / self.pressure.get(i)).into_value()?;
        }
        Ok(cost)
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for BinaryTPxy<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError> {
        self.pressure_cost(eos)
    }
}
