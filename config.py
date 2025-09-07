# config.py
import os

# Project settings
PROJECT_NAME = "Retail Sales Analysis & Customer Segmentation"
VERSION = "1.0.0"
AUTHOR = "Aryan Mishra"

# Data settings
N_RECORDS = 50000
RANDOM_SEED = 42

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'visualizations')

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, VISUALIZATION_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Analysis parameters
CLUSTERING_PARAMS = {
    'n_clusters_range': (2, 8),
    'random_state': RANDOM_SEED,
    'n_init': 10
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'husl'
FIGURE_SIZE = (12, 8)
DPI = 300