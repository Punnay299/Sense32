import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import ast
import json
import logging

class RFDataset(Dataset):
    def __init__(self, session_paths, seq_len=50, augment=False):
        """
        :param session_paths: List of paths to session folders (e.g. ['data/session_north_1', ...])
        :param seq_len: Number of RF packets to feed as input (Time Step)
        :param augment: boolean, whether to apply data augmentation (noise, dropout)
        """
        self.seq_len = seq_len
        self.augment = augment
        self.samples = [] 
        
        # We pre-load data into memory for speed (assuming fit in RAM). 
        # For huge datasets, we'd index file offsets.
        self.sessions_data = [] # List of {'rf_feats': np.array, 'rf_times': np.array}
        self.valid_indices = [] # List of (session_idx, rf_end_idx, label_dict)
        
        for s_idx, p in enumerate(session_paths):
            self._load_session(s_idx, p)
            
        logging.info(f"RFDataset: Loaded {len(self.valid_indices)} samples from {len(session_paths)} sessions.")

    def _load_session(self, s_idx, path):
        rf_path = os.path.join(path, "rf_data.csv")
        labels_path = os.path.join(path, "labels.csv")
        cam_idx_path = os.path.join(path, "camera_index.csv")
        
        if not os.path.exists(rf_path) or not os.path.exists(labels_path):
            logging.warning(f"Skipping {path}: Missing csv files")
            return

        try:
            # 1. Load RF Data
            rf_df = pd.read_csv(rf_path)
            if len(rf_df) < self.seq_len + 10:
                logging.warning(f"Skipping {path}: Not enough RF data")
                return
                
            rf_df = rf_df.sort_values("timestamp_monotonic_ms")
            
            # Check for CSI data presence
            has_csi = "csi_amp" in rf_df.columns and rf_df["csi_amp"].iloc[0] != "[]" and isinstance(rf_df["csi_amp"].iloc[0], str)
            
            # Filter for Dominant Source (Avoid mixing multiple ESP32s)
            if "source" in rf_df.columns:
                counts = rf_df["source"].value_counts()
                if len(counts) > 0:
                    dom_source = counts.index[0]
                    # logging.info(f"Session {path}: Locking to source {dom_source} (Counts: {counts.to_dict()})")
                    rf_df = rf_df[rf_df["source"] == dom_source]
            
            rf_feats = []
            
            if has_csi:
                # Parse CSI
                for i, x in enumerate(rf_df["csi_amp"].values):
                    try:
                        val = ast.literal_eval(x)
                        if isinstance(val, list):
                            # Pad/Cut to 64
                            if len(val) >= 64: val = val[:64]
                            else: val = val + [0]*(64-len(val))
                            rf_feats.append(val)
                        else:
                            rf_feats.append([0]*64)
                    except:
                        rf_feats.append([0]*64)
                
                rf_feats = np.array(rf_feats, dtype=np.float32)
                # Normalize CSI (0-1 approx)
                rf_feats = rf_feats / 127.0
                rf_feats = np.clip(rf_feats, 0, 1)
                
            else:
                # Fallback to RSSI + RTT (2 dims) -> Pad to 64 or handle in model
                # Ideally model should handle 64. We can zero-pad.
                rssi = rf_df["rssi"].fillna(-100).values.astype(np.float32)
                rtt = rf_df.get("rtt_ms", pd.Series([0]*len(rf_df))).fillna(0).values.astype(np.float32)
                
                # Norm
                rssi = (rssi + 100) / 100.0 # 0-1
                rtt = np.clip(rtt, 0, 1000) / 1000.0
                
                # Stack
                feats_2d = np.stack([rssi, rtt], axis=1) # [T, 2]
                
                # Pad to 64 columns
                padding = np.zeros((len(feats_2d), 62), dtype=np.float32)
                rf_feats = np.concatenate([feats_2d, padding], axis=1)
            
            rf_times = rf_df["timestamp_monotonic_ms"].values
            
            self.sessions_data.append({
                'rf_feats': rf_feats,
                'rf_times': rf_times
            })
            
            # 2. Load Labels
            lbl_df = pd.read_csv(labels_path)
            
            # Sync Logic: Use camera_index.csv if available to map frame_index -> timestamp
            # But wait! pose_extractor.py ALREADY does this and saves it to labels.csv.
            # If we merge again, pandas might suffix columns (timestamp_monotonic_ms_x), breaking things.
            # So we should TRUST labels.csv if it has the column.
            
            if "timestamp_monotonic_ms" not in lbl_df.columns:
                 # Only try to rescue if missing
                if os.path.exists(cam_idx_path):
                    cam_df = pd.read_csv(cam_idx_path)
                    if "frame_index" in lbl_df.columns and "frame_index" in cam_df.columns:
                        lbl_df = pd.merge(lbl_df, cam_df[["frame_index", "timestamp_monotonic_ms"]], on="frame_index", how="inner")
            
            if "timestamp_monotonic_ms" not in lbl_df.columns:
                logging.warning(f"Skipping {path}: No timestamps in labels and no camera_index.csv")
                return 
                
            lbl_df = lbl_df.sort_values("timestamp_monotonic_ms")
            
            # Location Label
            folder_name = os.path.basename(path).lower()
            # Location Label
            folder_name = os.path.basename(path).lower()
            loc_label = 0 # Default/Room 1
            
            if "room1" in folder_name: 
                loc_label = 0
            elif "room2" in folder_name: 
                loc_label = 1
            elif "hallway" in folder_name: 
                loc_label = 2
            elif "empty" in folder_name: 
                loc_label = 3
            
            # Legacy support (optional, can keep or remove)
            elif "south" in folder_name: loc_label = 1
            elif "east" in folder_name: loc_label = 2
            elif "west" in folder_name: loc_label = 3
            
            # 3. Create Index
            for i in range(len(lbl_df)):
                row = lbl_df.iloc[i]
                ts = row["timestamp_monotonic_ms"]
                
                # Find RF Index
                idx = np.searchsorted(rf_times, ts)
                
                if idx >= self.seq_len:
                    # Valid window
                    
                    # Parse Pose
                    pose_kps = np.zeros(66, dtype=np.float32)
                    presence = 0.0
                    
                    if "keypoints_flat" in row:
                        try:
                            kps = json.loads(row["keypoints_flat"])
                            if kps and len(kps) > 0:
                                presence = 1.0
                                # Extract x, y from [x, y, z, vis, ...]
                                flat_xy = []
                                for k in range(0, len(kps), 4):
                                    flat_xy.append(kps[k])
                                    flat_xy.append(kps[k+1])
                                
                                # Fix len 66
                                if len(flat_xy) >= 66: flat_xy = flat_xy[:66]
                                else: flat_xy = flat_xy + [0.0]*(66-len(flat_xy))
                                pose_kps = np.array(flat_xy, dtype=np.float32)
                        except:
                            pass
                    
                    # Transition Logic (Hybrid)
                    # For "transition" sessions, we use visibility to toggle Room 1 vs Room 2.
                    if "transition" in folder_name:
                         if presence > 0.5: # Visible
                             loc_label = 0 # Room 1
                         else:
                             # Not Visible -> Assume they are in Room 2 (as per user scenario)
                             # AND Force Presence = 1 (They are still there, just hidden)
                             loc_label = 1 # Room 2
                             presence = 1.0 # Force Presence

                    # Store minimal data needed for __getitem__
                    lbl_dict = {
                        "pose": pose_kps,
                        "presence": presence,
                        "location": loc_label
                    }
                    
                    # We store the *true* session index (since we filtered empty sessions, indices align with self.sessions_data)
                    # wait, self.sessions_data is appended only if RF loads. 
                    # So s_idx from enumerate is NOT safe if we skipped some.
                    # Correct: Use len(self.sessions_data) - 1
                    current_s_idx = len(self.sessions_data) - 1
                    
                    self.valid_indices.append((current_s_idx, idx, lbl_dict))
                    
        except Exception as e:
            logging.error(f"Error loading session {path}: {e}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        s_idx, rf_end_idx, lbl = self.valid_indices[idx]
        
        # Get RF Window
        session = self.sessions_data[s_idx]
        rf_window = session['rf_feats'][rf_end_idx-self.seq_len : rf_end_idx] # [Seq, 64]
        
        # Augmentation
        if self.augment:
            # Gaussian Noise
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, rf_window.shape).astype(np.float32)
                rf_window = rf_window + noise
            
            # Random Dropout (simulate packet loss)
            if np.random.rand() < 0.3:
                mask = np.random.rand(self.seq_len, 1) > 0.1 # 10% dropout
                rf_window = rf_window * mask
                
        # To Torch
        rf_tensor = torch.from_numpy(rf_window) # [Seq, 64]
        # Transpose for 1D CNN: [Channels, Seq]
        rf_tensor = rf_tensor.permute(1, 0)
        
        return {
            "rf": rf_tensor,
            "pose": torch.tensor(lbl["pose"], dtype=torch.float32),
            "presence": torch.tensor([lbl["presence"]], dtype=torch.float32),
            "location": torch.tensor(lbl["location"], dtype=torch.long)
        }
