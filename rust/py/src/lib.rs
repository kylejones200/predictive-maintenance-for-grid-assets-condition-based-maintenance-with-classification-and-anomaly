use predictive_maintenance_for_grid_assets_condition_based_maintenance_with_classification_and_anomaly_core::rolling_rms_features;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn rolling_rms_features_py<'py>(py: Python<'py>, series: PyReadonlyArray1<f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(rolling_rms_features(series.as_slice()?, window).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (series, window, iterations=500))]
fn bench_kernel_py(series: PyReadonlyArray1<f64>, window: usize, iterations: usize) -> PyResult<f64> {
    let series_buf = series.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = rolling_rms_features(&series_buf, window);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn predictive_maintenance_for_grid_assets_condition_based_maintenance_with_classification_and_anomaly_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_rms_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}
