import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import ast
import logging

class RFDataset(Dataset):
    def __init__(self, session_paths, window_ms=1000, max_packets=100):
        """
        :param session_paths: List of paths to session folders
        :param window_ms: Time window in ms to look back for RF data
        :param max_packets: Fixed number of packets to feed to model (padding/truncating)
        """
        self.window_ms = window_ms
        self.max_packets = max_packets
        self.samples = [] # List of (rf_subset_df, label_row) - optimization needed for large RAM?
                          # Better: (session_idx, frame_idx, timestamps)
        
        self.rf_data = [] # List of DataFrames, one per session
        self.labels = []  # List of DataFrames, one per session
        
        for p in session_paths:
            self._load_session(p)
            
        logging.info(f"Loaded {len(self.samples)} samples from {len(session_paths)} sessions.")

    def _load_session(self, path):
        rf_path = os.path.join(path, "rf_data.csv")
        lbl_path = os.path.join(path, "labels.csv")
        
        if not os.path.exists(rf_path) or not os.path.exists(lbl_path):
            return

        # Load RF
        df_rf = pd.read_csv(rf_path)
        # Ensure sorted
        df_rf = df_rf.sort_values("timestamp_monotonic_ms")
        
        # Load Labels
        df_lbl = pd.read_csv(lbl_path)
        
        # Filter only visible for training? Or all?
        # For regression we mostly want visible. For presence, we want all.
        # Let's keep all and return visibility flag.
        
        processed_rf_arrays = self._preprocess_rf(df_rf) # numpy matrix [Time, Feats]
        timestamps_rf = df_rf["timestamp_monotonic_ms"].values
        
        s_idx = len(self.rf_data)
        self.rf_data.append((timestamps_rf, processed_rf_arrays))
        self.labels.append(df_lbl)
        
        for idx, row in df_lbl.iterrows():
            self.samples.append((s_idx, idx))

    def _preprocess_rf(self, df):
        # Extract features. Currently just RSSI.
        # Normalize RSSI: -100 to -30 -> 0 to 1
        rssi = df["rssi"].values.astype(np.float32)
        rssi_norm = (rssi + 100) / 70.0 
        rssi_norm = np.clip(rssi_norm, 0, 1)
        return rssi_norm.reshape(-1, 1) # [T, 1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s_idx, l_idx = self.samples[idx]
        
        ts_rf_all, rf_feats_all = self.rf_data[s_idx]
        df_lbl = self.labels[s_idx]
        row = df_lbl.iloc[l_idx]
        
        ts_end = row["timestamp_monotonic_ms"]
        ts_start = ts_end - self.window_ms
        
        # Binary search for window
        idx_start = np.searchsorted(ts_rf_all, ts_start, side='left')
        idx_end = np.searchsorted(ts_rf_all, ts_end, side='right')
        
        window_feats = rf_feats_all[idx_start:idx_end]
        
        # Pad/Truncate
        L, C = window_feats.shape
        if L > self.max_packets:
            # Take latest
            window_feats = window_feats[-self.max_packets:]
        elif L < self.max_packets:
            pad_amt = self.max_packets - L
            padding = np.zeros((pad_amt, C), dtype=np.float32)
            window_feats = np.concatenate([padding, window_feats], axis=0)
            
        # To Tensor [Channels, Time] for 1D Conv
        rf_tensor = torch.from_numpy(window_feats).permute(1, 0) 
        
        # Label
        target = np.array([row["center_x"], row["center_y"]], dtype=np.float32)
        visible = 1.0 if str(row["visible"]) == 'True' else 0.0
        
        return rf_tensor, torch.from_numpy(target), torch.tensor([visible], dtype=torch.float32)
