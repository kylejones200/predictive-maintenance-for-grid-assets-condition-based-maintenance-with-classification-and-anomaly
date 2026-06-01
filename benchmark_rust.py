#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import rolling_rms_features  # noqa: E402

def main() -> None:
    s = np.ascontiguousarray(np.sin(np.arange(5000) * 0.01) + 50.0)
    window = 24
    t0 = time.perf_counter()
    for _ in range(200):
        rolling_rms_features(s, window)
    py_s = time.perf_counter() - t0
    try:
        import predictive_maintenance_for_grid_assets_condition_based_maintenance_with_classification_and_anomaly_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(s, window, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        rolling_rms_features(s, window),
        np.asarray(rs.rolling_rms_features_py(s, window)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
