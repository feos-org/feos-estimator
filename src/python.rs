use crate::EstimatorError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

impl From<EstimatorError> for PyErr {
    fn from(e: EstimatorError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}

#[macro_export]
macro_rules! impl_estimator {
    ($eos:ty, $py_eos:ty) => {
        #[pyclass(name = "Loss", unsendable)]
        #[derive(Clone)]
        pub struct PyLoss(Loss);

        #[pymethods]
        impl PyLoss {
            /// Create a linear loss function.
            /// 
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = z` and `s = 1`.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            pub fn linear() -> Self {
                Self(Loss::Linear)
            }

            /// Create a loss function according to Huber's method.
            /// 
            /// `loss = s**2 * rho(f**2 / s**2)`
            /// where `rho(z) = z if z <= 1 else 2*z**0.5 - 1`.
            /// `s` is the scaling factor.
            ///
            /// Parameters
            /// ----------
            /// scaling_factor : f64
            ///     Scaling factor for Huber loss function.
            ///
            /// Returns
            /// -------
            /// Loss
            #[staticmethod]
            #[pyo3(text_signature = "(scaling_factor)")]
            pub fn huber(scaling_factor: f64) -> Self {
                Self(Loss::Huber(scaling_factor))
            }
        }

        /// A collection of experimental data that can be used to compute
        /// cost functions and make predictions using an equation of state.
        #[pyclass(name = "DataSet", unsendable)]
        #[derive(Clone)]
        pub struct PyDataSet(Rc<dyn DataSet<SIUnit, $eos>>);

        #[pymethods]
        impl PyDataSet {
            /// Compute the cost function for each input value.
            ///
            /// The cost function that is used depends on the
            /// property. See the class constructors to learn
            /// about the cost functions of the properties.
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            ///     The cost function evaluated for each experimental data point.
            #[pyo3(text_signature = "($self, eos)")]
            fn cost<'py>(&self, eos: &$py_eos, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.cost(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the property of interest for each data point
            /// of the input as computed by the equation of state.
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// SIArray1
            ///
            /// See also
            /// --------
            /// eos_python.saft.estimator.DataSet.vapor_pressure : ``DataSet`` for vapor pressure.
            /// eos_python.saft.estimator.DataSet.liquid_density : ``DataSet`` for liquid density.
            /// eos_python.saft.estimator.DataSet.equilibrium_liquid_density : ``DataSet`` for liquid density at vapor liquid equilibrium.
            #[pyo3(text_signature = "($self, eos)")]
            fn predict(&self, eos: &$py_eos) -> PyResult<PySIArray1> {
                Ok(self.0.predict(&eos.0)?.into())
            }

            /// Return the relative difference between experimental data
            /// and prediction of the equation of state.
            ///
            /// The relative difference is computed as:
            ///
            /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            #[pyo3(text_signature = "($self, eos)")]
            fn relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.relative_difference(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the mean absolute relative difference.
            ///
            /// The mean absolute relative difference is computed as:
            ///
            /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// Float
            #[pyo3(text_signature = "($self, eos)")]
            fn mean_absolute_relative_difference(&self, eos: &$py_eos) -> PyResult<f64> {
                Ok(self.0.mean_absolute_relative_difference(&eos.0)?)
            }

            /// Create a DataSet with experimental data for vapor pressure.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for vapor pressure.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// std_parameters : List[float], optional
            ///     Parameters for temperature dependent function
            ///     that models the variance of experimental vapor pressure data.
            ///     If not provided, all data points have the same weight
            ///     for the cost function.
            ///
            /// Returns
            /// -------
            /// ``DataSet``
            ///
            /// Notes
            /// -----
            /// The function for the experimental standard deviation as
            /// a function of the reduced temperature, :math:`T^* = T / T_\text{c}`, reads
            ///
            /// .. math:: \sigma_\text{experimental} = \exp\left(-p_0 T^* + p_1\right) + p_2
            ///
            /// The critical temperature, :math:`T_\text{c}`, is computed from the equation of state.
            /// The inverse variances (i.e. :math:`\sigma^2`) are normalized and then used to
            /// weight the data points when evaluating the cost function.
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, std_parameters)")]
            fn vapor_pressure(
                target: &PySIArray1,
                temperature: &PySIArray1,
                std_parameters: Option<Vec<f64>>,
            ) -> PyResult<Self> {
                Ok(Self(Rc::new(VaporPressure::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    std_parameters.unwrap_or(vec![0.0, 0.0, 0.0]),
                )?)))
            }

            /// Create a DataSet with experimental data for liquid density.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for vapor pressure.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            /// pressure : SIArray1
            ///     Pressure for experimental data points.
            ///
            /// Returns
            /// -------
            /// DataSet
            ///
            /// Notes
            /// -----
            /// The cost function for the liquid density is the relative difference.
            ///
            /// See also
            /// --------
            /// eos_python.saft.estimator.DataSet.relative_difference
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature, pressure)")]
            fn liquid_density(
                target: &PySIArray1,
                temperature: &PySIArray1,
                pressure: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Rc::new(LiquidDensity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                    pressure.clone().into(),
                )?)))
            }

            /// Create a DataSet with experimental data for liquid density
            /// for a vapor liquid equilibrium.
            ///
            /// Parameters
            /// ----------
            /// target : SIArray1
            ///     Experimental data for vapor pressure.
            /// temperature : SIArray1
            ///     Temperature for experimental data points.
            ///
            /// Returns
            /// -------
            /// DataSet
            ///
            /// Notes
            /// -----
            /// The cost function for the liquid density is the relative difference.
            ///
            /// See also
            /// --------
            /// eos_python.saft.estimator.DataSet.relative_difference
            #[staticmethod]
            #[pyo3(text_signature = "(target, temperature)")]
            fn equilibrium_liquid_density(
                target: &PySIArray1,
                temperature: &PySIArray1,
            ) -> PyResult<Self> {
                Ok(Self(Rc::new(EquilibriumLiquidDensity::<SIUnit>::new(
                    target.clone().into(),
                    temperature.clone().into(),
                )?)))
            }

            /// Return `input` as ``Dict[str, SIArray1]``.
            #[getter]
            fn get_input(&self) -> HashMap<String, PySIArray1> {
                let mut m = HashMap::with_capacity(2);
                self.0.get_input().drain().for_each(|(k, v)| {
                    m.insert(k, PySIArray1::from(v));
                });
                m
            }

            /// Return `target` as ``SIArray1``.
            #[getter]
            fn get_target(&self) -> PySIArray1 {
                PySIArray1::from(self.0.target())
            }

            /// Return `target` as ``SIArray1``.
            #[getter]
            fn get_datapoints(&self) -> usize {
                self.0.datapoints()
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }

        /// A collection `DataSets` that can be used to compute metrics for experimental data.
        ///
        /// Parameters
        /// ----------
        /// data : List[DataSet]
        ///     The properties and experimental data points to add to
        ///     the estimator.
        /// weights : List[float]
        ///     The weight of each property. When computing the cost function,
        ///     the weights are normalized (sum of weights equals unity).
        #[pyclass(name = "Estimator", unsendable)]
        #[pyo3(text_signature = "(data, weights)")]
        pub struct PyEstimator(Estimator<SIUnit, $eos>);

        #[pymethods]
        impl PyEstimator {
            #[new]
            fn new(data: Vec<PyDataSet>, loss: Vec<PyLoss>, weights: Vec<f64>) -> Self {
                Self(Estimator::new(
                    data.iter().map(|d| d.0.clone()).collect(),
                    loss.iter().map(|l| l.0.clone()).collect(),
                    weights,
                ))
            }

            /// Compute the cost function for each ``DataSet``.
            ///
            /// The cost function is:
            /// - The relative difference between prediction and target value,
            /// - to which a loss function is applied,
            /// - and which is weighted according to the number of datapoints,
            /// - and the relative weights as defined in the Estimator object.
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            ///     The cost function evaluated for each experimental data point
            ///     of each ``DataSet``.
            #[pyo3(text_signature = "($self, eos)")]
            fn cost<'py>(&self, eos: &$py_eos, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
                Ok(self.0.cost(&eos.0)?.view().to_pyarray(py))
            }

            /// Return the properties as computed by the
            /// equation of state for each `DataSet`.
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// List[SIArray1]
            #[pyo3(text_signature = "($self, eos)")]
            fn predict(&self, eos: &$py_eos) -> PyResult<Vec<PySIArray1>> {
                Ok(self
                    .0
                    .predict(&eos.0)?
                    .iter()
                    .map(|d| PySIArray1::from(d.clone()))
                    .collect())
            }

            /// Return the relative difference between experimental data
            /// and prediction of the equation of state for each ``DataSet``.
            ///
            /// The relative difference is computed as:
            ///
            /// .. math:: \text{Relative Difference} = \frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}}
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// List[numpy.ndarray[Float]]
            #[pyo3(text_signature = "($self, eos)")]
            fn relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<Vec<&'py PyArray1<f64>>> {
                Ok(self
                    .0
                    .relative_difference(&eos.0)?
                    .iter()
                    .map(|d| d.view().to_pyarray(py))
                    .collect())
            }

            /// Return the mean absolute relative difference for each ``DataSet``.
            ///
            /// The mean absolute relative difference is computed as:
            ///
            /// .. math:: \text{MARD} = \frac{1}{N}\sum_{i=1}^{N} \left|\frac{x_i^\text{prediction} - x_i^\text{experiment}}{x_i^\text{experiment}} \right|
            ///
            /// Parameters
            /// ----------
            /// eos : PyEos
            ///     The equation of state that is used.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray[Float]
            #[pyo3(text_signature = "($self, eos)")]
            fn mean_absolute_relative_difference<'py>(
                &self,
                eos: &$py_eos,
                py: Python<'py>,
            ) -> PyResult<&'py PyArray1<f64>> {
                Ok(self
                    .0
                    .mean_absolute_relative_difference(&eos.0)?
                    .view()
                    .to_pyarray(py))
            }

            /// Return the stored ``DataSet``s.
            ///
            /// Returns
            /// -------
            /// List[DataSet]
            #[getter]
            fn get_datasets(&self) -> Vec<PyDataSet> {
                self.0
                    .datasets()
                    .iter()
                    .cloned()
                    .map(|ds| PyDataSet(ds))
                    .collect()
            }

            fn _repr_markdown_(&self) -> String {
                self.0._repr_markdownn_()
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.0.to_string())
            }
        }
    };
}
