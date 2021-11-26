use std::{fmt::LowerExp, rc::Rc};

use super::{DataSet, FitError};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, State, VLEOptions};
use ndarray::Array1;
use ndarray_stats::QuantileExt;
use quantity::{Quantity, QuantityArray1, QuantityScalar};

/// Store experimental vapor pressure data and compare to the equation of state.
#[derive(Clone)]
pub struct VaporPressure<U: EosUnit> {
    vapor_pressure: QuantityArray1<U>,
    temperature: QuantityArray1<U>,
    max_temperature: QuantityScalar<U>,
    datapoints: usize,
    std_parameters: Vec<f64>,
}

impl<U: EosUnit> VaporPressure<U> {
    /// Create a new vapor pressure data set.
    ///
    /// Takes the temperature as input and possibly parameters
    /// that describe the standard deviation of vapor pressure as
    /// function of temperature. This standard deviation can be used
    /// as inverse weights in the cost function.
    pub fn new(
        vapor_pressure: QuantityArray1<U>,
        temperature: QuantityArray1<U>,
        std_parameters: Vec<f64>,
    ) -> Result<Self, FitError> {
        let datapoints = vapor_pressure.len();
        let max_temperature = *temperature
            .to_reduced(U::reference_temperature())
            .unwrap()
            .max()
            .map_err(|_| FitError::IncompatibleInput)?
            * U::reference_temperature();
        Ok(Self {
            vapor_pressure,
            temperature,
            max_temperature,
            datapoints,
            std_parameters,
        })
    }
}

impl<U: EosUnit, E: EquationOfState> DataSet<U, E> for VaporPressure<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError> {
        let critical_point =
            State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())?;
        // let tc_inv = if let Ok(critical_point) = critical_point {
        //     1.0 / critical_point.temperature
        // } else {
        //     return Err(FitError::IncompatibleInput);
        // };

        // let reduced_temperatures = (0..self.datapoints)
        //     .map(|i| (self.temperature.get(i) * tc_inv).into_value().unwrap())
        //     .collect::<Vec<f64>>();

        // let prediction = &self.predict(eos)?;
        let mut cost = Array1::zeros(self.datapoints);

        for i in 0..self.datapoints {
            let temperature = self.temperature.get(i);
            if temperature > critical_point.temperature {
                cost[i] = 5.0
                    * (temperature - critical_point.temperature)
                        .to_reduced(U::reference_temperature())
                        .unwrap();
            } else {
                let prediction =
                    PhaseEquilibrium::pure_t(eos, temperature, None, VLEOptions::default())?
                        .vapor()
                        .pressure(Contributions::Total);
                cost[i] = ((self.vapor_pressure.get(i) - prediction) / self.vapor_pressure.get(i))
                    .into_value()?
            }
        }
        Ok(cost)
    }
}
//     fn target(&self) -> QuantityArray1<U> {
//         self.target.clone()
//     }

//     fn target_str(&self) -> &str {
//         "vapor pressure"
//         // r"$p^\text{sat}$"
//     }

//     fn input_str(&self) -> Vec<&str> {
//         vec!["temperature"]
//     }

//     fn predict(&self, eos: &Rc<E>) -> Result<QuantityArray1<U>, FitError>
//     where
//         QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
//     {
//         let tc =
//             State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())
//                 .unwrap()
//                 .temperature;

//         let unit = self.target.get(0);
//         let mut prediction = Array1::zeros(self.datapoints) * unit;
//         for i in 0..self.datapoints {
//             let t = self.temperature.get(i);
//             if t < tc {
//                 let state = PhaseEquilibrium::pure_t(
//                     eos,
//                     self.temperature.get(i),
//                     None,
//                     VLEOptions::default(),
//                 );
//                 if let Ok(s) = state {
//                     prediction
//                         .try_set(i, s.liquid().pressure(Contributions::Total))
//                         .unwrap();
//                 } else {
//                     println!(
//                         "Failed to compute vapor pressure, T = {}",
//                         self.temperature.get(i)
//                     );
//                     prediction.try_set(i, f64::NAN * unit).unwrap();
//                 }
//             } else {
//                 prediction.try_set(i, f64::NAN * unit).unwrap();
//             }
//         }
//         Ok(prediction)
//     }

//     fn cost(&self, eos: &Rc<E>) -> Result<Array1<f64>, FitError>
//     where
//         QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
//     {
//         let tc_inv = 1.0
//             / State::critical_point(eos, None, Some(self.max_temperature), VLEOptions::default())
//                 .unwrap()
//                 .temperature;

//         let reduced_temperatures = (0..self.datapoints)
//             .map(|i| (self.temperature.get(i) * tc_inv).into_value().unwrap())
//             .collect();
//         let mut weights = self.weight_from_std(&reduced_temperatures);
//         weights /= weights.sum();

//         let prediction = &self.predict(eos)?;
//         let mut cost = Array1::zeros(self.datapoints);
//         for i in 0..self.datapoints {
//             if prediction.get(i).is_nan() {
//                 cost[i] = weights[i]
//                     * 5.0
//                     * (self.temperature.get(i) - 1.0 / tc_inv)
//                         .to_reduced(U::reference_temperature())
//                         .unwrap();
//             } else {
//                 cost[i] = weights[i]
//                     * ((self.target.get(i) - prediction.get(i)) / self.target.get(i))
//                         .into_value()?
//             }
//         }
//         Ok(cost)
//     }

//     fn get_input(&self) -> HashMap<String, QuantityArray1<U>> {
//         let mut m = HashMap::with_capacity(1);
//         m.insert("temperature".to_owned(), self.temperature());
//         m
//     }
// }
