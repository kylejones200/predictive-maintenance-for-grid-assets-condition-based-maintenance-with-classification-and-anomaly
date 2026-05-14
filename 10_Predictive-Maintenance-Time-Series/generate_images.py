#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""

import signalplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import logging

def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml
        return _yaml.safe_load(_f) or {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Set random seeds
np.random.seed(config.get('data', {}).get('seed', 42))
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except Exception:
    tf = None

# Tufte-style configuration
signalplot.apply(font_family='serif')

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
import yaml
original_savefig = plt.savefig

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith('/') and not str(filename).startswith('images/'):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

plt.savefig = savefig_tufte

# (Placeholder script – replace with real predictive maintenance visualizations as needed.)
logger.info("All images generated successfully!")
