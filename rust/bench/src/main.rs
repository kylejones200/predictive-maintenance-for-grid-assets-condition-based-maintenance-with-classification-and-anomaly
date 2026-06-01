use predictive_maintenance_for_grid_assets_condition_based_maintenance_with_classification_and_anomaly_core::rolling_rms_features;

fn main() {
    let s: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + 50.0).collect();
    for _ in 0..2000 {
        let _ = rolling_rms_features(&s, 24);
    }
}
