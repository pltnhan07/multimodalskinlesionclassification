# src/utils.py
import yaml
import pandas as pd
import os

def load_config(config_file="config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_metadata(csv_file, metadata_columns):
    """
    Load metadata CSV and return:
    - A list/array of image IDs (assumed column 'img_id')
    - An array of metadata (only the specified columns)
    - A list/array of labels (assumed column 'diagnostic_encoded')
    """
    df = pd.read_csv(csv_file)
    img_ids = df['img_id'].values
    metadata = df[metadata_columns].values
    labels = df['diagnostic_encoded'].values
    return img_ids, metadata, labels

def get_file_paths(img_ids, image_dir, ext=".png"):
    """Construct file paths given image IDs and a directory."""
    return [os.path.join(image_dir, f"{id}{ext}") for id in img_ids]

def compute_alpha(labels, num_classes, multiplier_dict=None):
    """
    Compute per-class weights (alpha) using inverse frequency.
    Optionally adjust certain classes with multipliers.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    inverse_freq = 1.0 / counts.astype(float)
    if multiplier_dict:
        for cls, multiplier in multiplier_dict.items():
            inverse_freq[cls] *= multiplier
    alpha = inverse_freq / inverse_freq.sum()
    return alpha.tolist()
