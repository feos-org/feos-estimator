use super::{DataSetBinary, EstimatorError, Loss};
use feos_core::{Contributions, EosUnit, EquationOfState, PhaseEquilibrium, SolverOptions, State};
use ndarray::{arr1, Array1};
use quantity::{Quantity, QuantityArray1, QuantityScalar};
use std::{fmt::LowerExp, rc::Rc};

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "python", pyo3::pyclass)]
pub enum Cost {
    Pressure,
    ChemicalPotential,
    Distance,
}

/// Store experimental vapor pressure data and compare to the equation of state.
pub struct BinaryTPx<U: EosUnit> {
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    liquid_molefracs: Array1<f64>,
    datapoints: usize,
    cost: Cost,
}

impl<U: EosUnit> BinaryTPx<U> {
    pub fn new(
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        liquid_molefracs: Array1<f64>,
        cost: Cost,
    ) -> Result<Self, EstimatorError> {
        let datapoints = temperature.len();
        match cost {
            Cost::ChemicalPotential => Err(EstimatorError::IncompatibleInput),
            _ => Ok(Self {
                temperature,
                pressure,
                liquid_molefracs,
                datapoints,
                cost,
            }),
        }
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (SolverOptions::default(), SolverOptions::default());
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let prediction = PhaseEquilibrium::bubble_point(
                eos,
                self.temperature.get(i),
                &arr1(&vec![xi, 1.0 - xi]),
                Some(self.pressure.get(i)),
                None,
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);
            cost[i] = ((self.pressure.get(i) - prediction) / self.pressure.get(i)).into_value()?
        }
        Ok(cost)
    }

    fn distance_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let dx = 1e-4;
        let tol = 1e-9;
        let max_iter = 60;
        let options = (SolverOptions::default(), SolverOptions::default());
        let mut cost = Array1::zeros(self.datapoints);

        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let mut dxi = if xi < 0.5 { dx } else { -dx };
            let temperature = self.temperature.get(i);
            let pressure = self.pressure.get(i);
            let mut shift = 0.0;
            'iteration: for i in 0..max_iter {
                let damping = match i {
                    i if i <= 2 => 0.75,
                    i if i > 8 && shift < 1e-5 => 0.5,
                    i if i > 25 => 0.25,
                    _ => 1.0,
                };

                let xi_f = xi + shift * damping;
                let prediction = PhaseEquilibrium::bubble_point(
                    eos,
                    temperature,
                    &arr1(&vec![xi_f, 1.0 - xi_f]),
                    Some(pressure),
                    None,
                    options,
                );
                if prediction.is_err() {
                    cost[i] = 10.0;
                    break 'iteration;
                }
                let p1 = prediction.unwrap().vapor().pressure(Contributions::Total);

                if xi_f > 1.0 - dxi {
                    dxi *= -1.0
                };

                let xi_b = xi_f + dxi;
                let prediction = PhaseEquilibrium::bubble_point(
                    eos,
                    temperature,
                    &arr1(&vec![xi_b, 1.0 - xi_b]),
                    Some(pressure),
                    None,
                    options,
                );
                if prediction.is_err() {
                    cost[i] = 10.0;
                    break 'iteration;
                }
                let p2 = prediction.unwrap().vapor().pressure(Contributions::Total);
                let mut line_vec = arr1(&[dxi, (p2 - p1).to_reduced(pressure)?]);
                line_vec /= line_vec.mapv(|li| li * li).sum().sqrt();
                let exp_vec = arr1(&[xi - xi_f, (pressure - p1).to_reduced(pressure)?]);
                cost[i] = (&exp_vec * &exp_vec).sum().sqrt();
                shift = line_vec[0] * (&line_vec * &exp_vec).sum().sqrt();
                if shift > xi_f {
                    shift = xi_f
                }
                if shift < -xi_f {
                    shift = -xi_f
                }
                if shift.abs() <= tol {
                    break 'iteration;
                }
            }
        }
        Ok(cost)
    }
}

impl<U: EosUnit, E: EquationOfState> DataSetBinary<U, E> for BinaryTPx<U> {
    fn datapoints(&self) -> usize {
        self.temperature.len()
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = match self.cost {
            Cost::Pressure => self.pressure_cost(eos),
            Cost::Distance => self.distance_cost(eos),
            _ => Err(EstimatorError::IncompatibleInput),
        }?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
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
    ) -> Result<Self, EstimatorError> {
        let datapoints = temperature.len();
        Ok(Self {
            temperature,
            pressure,
            vapor_molefracs,
            datapoints,
        })
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (SolverOptions::default(), SolverOptions::default());

        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            let yi = self.vapor_molefracs[i];
            let prediction = PhaseEquilibrium::dew_point(
                eos,
                self.temperature.get(i),
                &arr1(&vec![yi, 1.0 - yi]),
                Some(self.pressure.get(i)),
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

impl<U: EosUnit, E: EquationOfState> DataSetBinary<U, E> for BinaryTPy<U> {
    fn datapoints(&self) -> usize {
        self.temperature.len()
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError>
    where
        QuantityScalar<U>: std::fmt::Display + std::fmt::LowerExp,
    {
        let mut cost = self.pressure_cost(eos)?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
    }
}

#[derive(Clone)]
pub struct BinaryTPxy<U: EosUnit> {
    temperature: QuantityArray1<U>,
    pressure: QuantityArray1<U>,
    liquid_molefracs: Array1<f64>,
    vapor_molefracs: Array1<f64>,
    datapoints: usize,
    cost: Cost,
}

impl<U: EosUnit> BinaryTPxy<U> {
    pub fn new(
        temperature: QuantityArray1<U>,
        pressure: QuantityArray1<U>,
        liquid_molefracs: Array1<f64>,
        vapor_molefracs: Array1<f64>,
        cost: Cost,
    ) -> Result<Self, EstimatorError> {
        let datapoints = temperature.len();
        Ok(Self {
            temperature,
            pressure,
            liquid_molefracs,
            vapor_molefracs,
            datapoints,
            cost,
        })
    }

    fn pressure_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let options = (SolverOptions::default(), SolverOptions::default());

        let mut cost = Array1::zeros(2 * self.datapoints);
        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let yi = self.vapor_molefracs[i];

            let prediction_liquid = PhaseEquilibrium::bubble_point(
                eos,
                self.temperature.get(i),
                &arr1(&vec![xi, 1.0 - xi]),
                Some(self.pressure.get(i)),
                Some(&arr1(&vec![yi, 1.0 - yi])),
                options,
            )?
            .vapor()
            .pressure(Contributions::Total);

            let prediction_vapor = PhaseEquilibrium::dew_point(
                eos,
                self.temperature.get(i),
                &arr1(&vec![yi, 1.0 - yi]),
                Some(self.pressure.get(i)),
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

    fn chemical_potential_cost<E: EquationOfState>(
        &self,
        eos: &Rc<E>,
    ) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let mut cost = Array1::zeros(self.datapoints);
        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let yi = self.vapor_molefracs[i];
            let temperature = self.temperature.get(i);
            let pressure = self.pressure.get(i);
            let mu_liquid = State::new_npt(
                eos,
                temperature,
                pressure,
                &(arr1(&[xi, 1.0 - xi]) * U::reference_moles()),
                feos_core::DensityInitialization::Liquid,
            )?
            .chemical_potential(Contributions::Total)
            .to_reduced(U::reference_molar_energy())?;

            let mu_vapor = State::new_npt(
                eos,
                temperature,
                pressure,
                &(arr1(&[yi, 1.0 - yi]) * U::reference_moles()),
                feos_core::DensityInitialization::Vapor,
            )?
            .chemical_potential(Contributions::Total)
            .to_reduced(U::reference_molar_energy())?;
            cost[i] = (mu_liquid - mu_vapor).mapv(|dmu| dmu * dmu).sum().sqrt();
        }
        Ok(cost)
    }

    fn distance_cost<E: EquationOfState>(&self, eos: &Rc<E>) -> Result<Array1<f64>, EstimatorError>
    where
        Quantity<f64, U>: std::fmt::Display,
    {
        let dx = 1e-4;
        let tol = 1e-9;
        let max_iter = 60;
        let options = (SolverOptions::default(), SolverOptions::default());
        let mut cost = Array1::zeros(self.datapoints);

        for i in 0..self.datapoints {
            let xi = self.liquid_molefracs[i];
            let yi = self.vapor_molefracs[i];
            let mut dxi = if xi < 0.5 { dx } else { -dx };
            let temperature = self.temperature.get(i);
            let pressure = self.pressure.get(i);
            let mut shift = 0.0;
            'iteration: for i in 0..max_iter {
                let damping = match i {
                    i if i <= 2 => 0.75,
                    i if i > 8 && shift < 1e-5 => 0.5,
                    i if i > 25 => 0.25,
                    _ => 1.0,
                };

                let xi_f = xi + shift * damping;
                let prediction = PhaseEquilibrium::bubble_point(
                    eos,
                    temperature,
                    &arr1(&vec![xi_f, 1.0 - xi_f]),
                    Some(pressure),
                    Some(&arr1(&vec![yi, 1.0 - yi])),
                    options,
                );
                if prediction.is_err() {
                    cost[i] = 10.0;
                    break 'iteration;
                }
                let p1 = prediction.unwrap().vapor().pressure(Contributions::Total);

                if xi_f > 1.0 - dxi {
                    dxi *= -1.0
                };

                let xi_b = xi_f + dxi;
                let prediction = PhaseEquilibrium::bubble_point(
                    eos,
                    temperature,
                    &arr1(&vec![xi_b, 1.0 - xi_b]),
                    Some(pressure),
                    Some(&arr1(&vec![yi, 1.0 - yi])),
                    options,
                );
                if prediction.is_err() {
                    cost[i] = 10.0;
                    break 'iteration;
                }
                let p2 = prediction?.vapor().pressure(Contributions::Total);
                let mut line_vec = arr1(&[dxi, (p2 - p1).to_reduced(pressure)?]);
                line_vec /= line_vec.mapv(|li| li * li).sum().sqrt();
                let exp_vec = arr1(&[xi - xi_f, (pressure - p1).to_reduced(pressure)?]);
                cost[i] = (&exp_vec * &exp_vec).sum().sqrt();
                shift = line_vec[0] * (&line_vec * &exp_vec).sum().sqrt();
                if shift > xi_f {
                    shift = xi_f
                }
                if shift < -xi_f {
                    shift = -xi_f
                }
                if shift.abs() <= tol {
                    break 'iteration;
                }
            }
        }
        Ok(cost)
    }
}

impl<U: EosUnit, E: EquationOfState> DataSetBinary<U, E> for BinaryTPxy<U>
where
    Quantity<f64, U>: std::fmt::Display + LowerExp,
{
    fn datapoints(&self) -> usize {
        self.temperature.len()
    }

    fn cost(&self, eos: &Rc<E>, loss: Loss) -> Result<Array1<f64>, EstimatorError> {
        let mut cost = match self.cost {
            Cost::Pressure => self.pressure_cost(eos),
            Cost::Distance => self.distance_cost(eos),
            Cost::ChemicalPotential => self.chemical_potential_cost(eos),
        }?;
        loss.apply(&mut cost.view_mut());
        Ok(cost / self.datapoints as f64)
    }
}
