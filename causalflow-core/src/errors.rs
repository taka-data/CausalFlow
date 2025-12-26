use thiserror::Error;

#[derive(Error, Debug)]
pub enum CausalFlowError {
    #[error("Missing feature data: feature index {0} out of bounds")]
    FeatureOutOfBounds(usize),

    #[error("Invalid data: input contains NaN or Infinity")]
    InvalidData,

    #[error("Empty data: training or inference data cannot be empty")]
    EmptyData,

    #[error("Invalid treatment: treatment values must be binary (0 or 1), found {0}")]
    InvalidTreatment(f64),

    #[error("Model not fitted: please call fit() before predicting")]
    ModelNotFitted,

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Calculation error: {0}")]
    Calculation(String),
}

#[cfg(feature = "python")]
impl From<CausalFlowError> for pyo3::PyErr {
    fn from(err: CausalFlowError) -> pyo3::PyErr {
        use pyo3::exceptions::{PyRuntimeError, PyValueError};
        match err {
            CausalFlowError::InvalidData
            | CausalFlowError::EmptyData
            | CausalFlowError::InvalidTreatment(_)
            | CausalFlowError::FeatureOutOfBounds(_) => PyValueError::new_err(err.to_string()),
            CausalFlowError::ModelNotFitted
            | CausalFlowError::Internal(_)
            | CausalFlowError::Calculation(_) => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, CausalFlowError>;
