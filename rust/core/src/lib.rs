//! Rolling RMS feature for predictive maintenance signals.

pub fn rolling_rms_features(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let w = window.max(1);
    let mut out = vec![0.0; n];
    for i in 0..n {
        let start = i.saturating_sub(w - 1);
        let slice = &series[start..=i];
        let ms = slice.iter().map(|x| x * x).sum::<f64>() / slice.len() as f64;
        out[i] = ms.sqrt();
    }
    out
}
