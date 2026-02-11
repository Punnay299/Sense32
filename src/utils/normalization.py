import numpy as np
import json
import os
import logging

class AdaptiveScaler:
    """
    Robust Scaler for CSI Data.
    Uses Median and IQR to handle outliers better than MinMax or Standard scaler.
    """
    def __init__(self):
        self.median = None
        self.iqr = None
        self.min_val = None
        self.max_val = None

    def fit(self, data):
        """
        Compute statistics for the dataset.
        :param data: np.array of shape (N, Features) or (N, Seq, Features)
        """
        # Flatten time dimension if present
        if len(data.shape) == 3:
            flat_data = data.reshape(-1, data.shape[-1])
        else:
            flat_data = data
            
        logging.info(f"AdaptiveScaler: Fitting on {len(flat_data)} samples...")
        
        # 1. Compute Median
        self.median = np.median(flat_data, axis=0)
        
        # 2. Compute IQR (Inter-Quartile Range) 25th and 75th percentile
        q75, q25 = np.percentile(flat_data, [75 ,25], axis=0)
        self.iqr = q75 - q25
        
        # Avoid division by zero
        self.iqr[self.iqr == 0] = 1.0
        
        # 3. Compute Min/Max for clipping (optional fallback)
        self.min_val = np.min(flat_data, axis=0)
        self.max_val = np.max(flat_data, axis=0)
        
        logging.info("AdaptiveScaler: Fit complete.")

    def transform(self, data):
        """
        Apply Robust Scaling: (X - Median) / IQR
        Then clip to range [-3, 3] to remove extreme outliers.
        Finally scale to [0, 1] approximately for the Neural Network.
        """
        if self.median is None:
            raise ValueError("Scaler not fitted!")
            
        # (X - Median) / IQR
        scaled = (data - self.median) / self.iqr
        
        # Clip extreme outliers (e.g. glitches)
        scaled = np.clip(scaled, -3.0, 3.0)
        
        # Shift to [0, 1] range for stability (activation functions like ReLU work well with positive inputs)
        # -3 becomes 0, +3 becomes 1
        scaled = (scaled + 3.0) / 6.0
        
        return scaled

    def save(self, path):
        """Save stats to JSON"""
        stats = {
            "median": self.median.tolist(),
            "iqr": self.iqr.tolist(),
            "min": self.min_val.tolist(),
            "max": self.max_val.tolist()
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
        logging.info(f"Scaler saved to {path}")

    def load(self, path):
        """Load stats from JSON"""
        if not os.path.exists(path):
            logging.warning(f"Scaler file {path} not found.")
            return

        with open(path, 'r') as f:
            stats = json.load(f)
            
        self.median = np.array(stats["median"], dtype=np.float32)
        self.iqr = np.array(stats["iqr"], dtype=np.float32)
        self.min_val = np.array(stats["min"], dtype=np.float32)
        self.max_val = np.array(stats["max"], dtype=np.float32)
        logging.info(f"Scaler loaded from {path}")
