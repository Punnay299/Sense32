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
            
        logging.info(f"AdaptiveScaler: Fitting on {len(flat_data)} samples (GLOBAL Mode)...")
        
        # GLOBAL SCALING: Compute stats across ALL values (Features flattened too)
        # This ensures we preserve relative magnitude between Node A and Node B.
        
        # Flatten everything to 1D array
        all_values = flat_data.flatten()
        
        # 1. Compute Median
        self.median = np.median(all_values)
        
        # 2. Compute IQR
        q75, q25 = np.percentile(all_values, [75 ,25])
        self.iqr = q75 - q25
        
        if self.iqr == 0: self.iqr = 1.0
        
        # 3. Min/Max (just for logging)
        self.min_val = np.min(all_values)
        self.max_val = np.max(all_values)
        
        logging.info(f"AdaptiveScaler: Fit complete. Median={self.median:.4f}, IQR={self.iqr:.4f}")

    def transform(self, data):
        """
        Apply Robust Scaling: (X - Median) / IQR
        Then clip to range [-3, 3] to remove extreme outliers.
        Finally scale to [0, 1] approximately for the Neural Network.
        """
        if self.median is None:
            # Fallback if not fitted (e.g. strict instance norm wanted)
            # For this specific project restart, we might want purely instance norm.
            # But to keep compatibility, let's keep the error or just warn.
            pass
            
        # Standard Adaptive Scaling (Global)
        if self.median is not None:
            # LOG1P SCALING (New Feb 2026)
            # Compress spikes: log(x+1)
            # We assume data is absolute amplitude. Phase should NOT be logged if it includes negative values?
            # Wait, phase is -pi to pi. Taking log of negative is bad.
            # But this scaler is used on EVERYTHING (stacked).
            # We must be careful.
            # "data" here is the full stacked array.
            # Ideally, we should only log AMPLITUDE.
            # But the scaler is generic.
            # Let's trust that 'dataset.py' handles the splitting or we just apply log to everything and hope phase doesn't break?
            # No, log of negative phase is NaN.
            # WE MUST HANDLE THIS IN DATASET.PY instead of here?
            # Or we assume this Scaler is ONLY for Amplitude?
            # The code stacks Amp+Phase then calls transform().
            # So we CANNOT just log everything.
            
            # REVERT STRATEGY: Do NOT log here.
            # Log in dataset.py/inference.py BEFORE stacking or BEFORE calling transform.
            # But wait, transform() uses self.median which was fitted on... what?
            # If we log in dataset.py, we feed logged data to fit().
            # So this scaler just sees "some distribution" and scales it.
            # That is cleaner.
            
            # So... actually I should NOT change this file to force log?
            # The plan said "Update AdaptiveScaler to apply np.log1p".
            # But due to Phase (negative values), I should delegate that to the caller.
            # OR I can check for negative values?
            
            # Let's keep this file as "Robust Z-Score" (Median/IQR) and do the Log transform OUTSIDE.
            # Actually, let's just make it robust to NaNs if I forced it? No.
            
            # CHANGE OF PLAN: I will NOT modify this file's logic to force log.
            # I will modify dataset.py to apply log to AMPLITUDE ONLY, then stack Phase, then fit/transform.
            
            scaled = (data - self.median) / self.iqr
            scaled = np.clip(scaled, -3.0, 3.0)
            scaled = (scaled + 3.0) / 6.0
            return scaled
        else:
            return data

    @staticmethod
    def instance_norm(data):
        """
        Apply Instance Normalization (Per Window Z-Score).
        Independent of Global Statistics.
        Shape: (N, Features)
        """
        # Mean/Std per sample (axis=1) or per window? 
        # Here 'data' is usually (50, 64) -> Window.
        # We want to normalize the ENTIRE window to mean=0, std=1.
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0: std = 1e-5
        
        scaled = (data - mean) / std
        
        # Clip outliers
        scaled = np.clip(scaled, -3.0, 3.0)
        
        # Scale to [0, 1] for model (Soft range)
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
