import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import ast
import json
import logging
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.normalization import AdaptiveScaler
from src.utils.csi_sanitizer import CSISanitizer

class RFDataset(Dataset):
    def __init__(self, session_paths, seq_len=50, augment=False, scaler=None):
        """
        :param session_paths: List of paths to session folders
        :param seq_len: Time steps
        :param augment: Data augmentation flag
        :param scaler: Pre-fitted AdaptiveScaler (optional). If None, will fit on loaded data.
        """
        self.seq_len = seq_len
        self.augment = augment
        self.samples = [] 
        
        # We pre-load data into memory for speed 
        self.sessions_data = [] # List of {'rf_feats': np.array, 'rf_times': np.array}
        self.valid_indices = [] # List of (session_idx, rf_end_idx, label_dict)
        
        for s_idx, p in enumerate(session_paths):
            self._load_session(s_idx, p)
            
        logging.info(f"RFDataset: Loaded {len(self.valid_indices)} samples from {len(session_paths)} sessions.")
        
        # Fit or Use Scaler
        if scaler:
            self.scaler = scaler
            logging.info("RFDataset: Using provided scaler.")
        else:
            self.scaler = AdaptiveScaler()
            # Collect all RF data to fit
            all_rf = []
            for s in self.sessions_data:
                all_rf.append(s['rf_feats'])
            
            if len(all_rf) > 0:
                full_stack = np.concatenate(all_rf, axis=0)
                self.scaler.fit(full_stack)
            else:
                logging.warning("RFDataset: No data to fit scaler!")
                
        # Transform all loaded data immediately to save time during training
        for s in self.sessions_data:
             s['rf_feats'] = self.scaler.transform(s['rf_feats'])

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
            
            # Filter for Dominant Sources (Dual Node Support)
            if "source" in rf_df.columns:
                counts = rf_df["source"].value_counts()
                unique_sources = sorted(counts.index.tolist()) # Sort by string (IP/MAC)
                
                # We expect 2 nodes. If more, take top 2.
                # If 1, duplicate? Or fail? User wants ROBUST.
                # Let's take up to 2.
                if len(unique_sources) > 2:
                     unique_sources = unique_sources[:2]
                
                logging.info(f"Session {path}: Found sources {unique_sources}")
                
            # Separate DataFrames per Source
            dfs = {}
            for src in unique_sources:
                 dfs[src] = rf_df[rf_df["source"] == src].copy()
            
            # Master Time Base (Use the first source as master)
            if len(unique_sources) > 0:
                 master_src = unique_sources[0]
                 master_times = dfs[master_src]["timestamp_monotonic_ms"].values
            else:
                 logging.warning(f"Skipping {path}: No sources found")
                 return

            # Extract Features for each source
            # We will stack them: [Src1_64, Src2_64] -> 128
            
            merged_feats = []
            
            # Pre-parse CSI for all
            parsed_csi = {}
            for src in unique_sources:
                 parsed_csi[src] = []
                 for x in dfs[src]["csi_amp"].values:
                    try:
                        val = ast.literal_eval(x)
                        if isinstance(val, list):
                            if len(val) >= 64: val = val[:64]
                            else: val = val + [0]*(64-len(val))
                            parsed_csi[src].append(val)
                        else:
                            parsed_csi[src].append([0]*64)
                    except:
                        parsed_csi[src].append([0]*64)
                 parsed_csi[src] = np.array(parsed_csi[src], dtype=np.float32)
                 
                 # --- AMPLITUDE SANITIZATION ---
                 # 1. Hampel Filter (Amplitude Spikes)
                 parsed_csi[src] = CSISanitizer.sanitize_amplitude(parsed_csi[src])
                 
                 # --- PHASE LOADING & SANITIZATION ---
                 # We need to parse 'csi_phase' if available. 
                 # The current loop iterates over 'csi_amp'. We need to iterate over 'csi_phase' too?
                 # Better: Do it in one go if possible, or parallel loop.
                 # Let's check if 'csi_phase' column exists
                 
                 parsed_phase = []
                 if "csi_phase" in dfs[src].columns:
                     for x in dfs[src]["csi_phase"].values:
                        try:
                            val = ast.literal_eval(x)
                            if isinstance(val, list):
                                if len(val) >= 64: val = val[:64]
                                else: val = val + [0]*(64-len(val))
                                parsed_phase.append(val)
                            else:
                                parsed_phase.append([0]*64)
                        except:
                            parsed_phase.append([0]*64)
                 else:
                     # Fallback if no phase data (shouldn't happen with new collection)
                     parsed_phase = [[0]*64] * len(parsed_csi[src])
                 
                 parsed_phase = np.array(parsed_phase, dtype=np.float32)
                 
                 # Sanitize Phase (Unwrap + Detrend)
                 # Only if we actually have data (non-zero)
                 if np.max(np.abs(parsed_phase)) > 0:
                    parsed_phase = CSISanitizer.sanitize_phase(parsed_phase)
                 
                 # Stack Amplitude + Phase [Seq, 128]
                 # We want [Seq, 64_Amp, 64_Phase]?
                 # networks.py expects 1D per subcarrier? 
                 # Actually, it treats input as "Features". 
                 # If we stack [Amp, Phase], we get 128 features per node.
                 # Total input to model (2 nodes) = 256 features.
                 
                 parsed_csi[src] = np.concatenate([parsed_csi[src], parsed_phase], axis=1)

            # Sync and Merge
            # FIXED LOGIC: Strict Slot Assignment
            # Slot 0 = Node A (10.42.0.149)
            # Slot 1 = Node B (10.42.0.173)
            
            target_ips = ["10.42.0.149", "10.42.0.173"] 
            
            rf_feats = []
            
            # Determine Master Time Base
            # Ideally use the one that exists. If both exist, use Node A?
            # If only B exists, use B.
            
            present_sources = [s for s in target_ips if s in unique_sources]
            if len(present_sources) == 0:
                 # Fallback for some non-standard IPs?
                 # Or just skip?
                 if len(unique_sources) > 0:
                     master_src = unique_sources[0] # Use whatever we found
                 else:
                     return 
            else:
                master_src = present_sources[0]
                
            master_times = dfs[master_src]["timestamp_monotonic_ms"].values

            for i, ts in enumerate(master_times):
                 row_feat = []
                 
                 for ip in target_ips:
                     # Check if we have data for this IP
                     # Note: We rely on 'unique_sources' containing the full IP string found in CSV
                     # Is it partial match? dataset usually has full IP if sanitized.
                     # Let's check keys in parsed_csi.
                     
                     found_key = None
                     for k in parsed_csi.keys():
                         if ip in k: # Partial match for safety (e.g. port numbers?)
                             found_key = k
                             break
                     
                     if found_key:
                         # We have data for this Node Slot
                         # Find sync index
                         if found_key == master_src:
                             # It's the master, direct index
                             idx = i
                         else:
                             # Sync
                             curr_times = dfs[found_key]["timestamp_monotonic_ms"].values
                             idx = np.searchsorted(curr_times, ts)
                             if idx >= len(curr_times): idx = len(curr_times) - 1
                             if idx < 0: idx = 0
                        
                         # Extract 128 features (64 Amp + 64 Phase)
                         raw_128 = parsed_csi[found_key][idx]
                         
                         # Apply LOG1P to Amplitude ONLY (First 64)
                         # We do this BEFORE stacking to ensure the scaler sees log-distributed data.
                         amp = raw_128[:64]
                         phase = raw_128[64:]
                         
                         # Log Transform: log(x+1) to compress spikes
                         amp_log = np.log1p(amp)
                         
                         # Re-stack
                         node_vec = np.concatenate([amp_log, phase])
                         row_feat.extend(node_vec)
                     else:
                         # Node Missing -> PAD ZEROES
                         row_feat.extend([0]*128) # 64 Amp + 64 Phase
                 
                 # --- DIFFERENCE FEATURE (Feb 2026 Bias Fix) ---
                 # We now have [AmpA_Log, PhA, AmpB_Log, PhB] in 'row_feat' (Length 256)
                 # We want to add [Diff_Amp] = AmpA_Log - AmpB_Log
                 # Indices: 
                 # AmpA_Log: 0:64
                 # AmpB_Log: 128:192
                 
                 # Check if we have enough data (2 nodes * 128 = 256)
                 if len(row_feat) == 256:
                     amp_a_log = np.array(row_feat[0:64])
                     amp_b_log = np.array(row_feat[128:192])
                     
                     # Simple Difference
                     diff_amp = amp_a_log - amp_b_log
                     
                     row_feat.extend(diff_amp) # Add 64 features -> Total 320
                 else:
                     # Should not happen given logic above, but safety
                     row_feat.extend([0]*64)

                 rf_feats.append(row_feat)

            rf_feats = np.array(rf_feats, dtype=np.float32)
            rf_times = master_times
            
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
            
            if "room1" in folder_name or "room_a" in folder_name: 
                loc_label = 0
            elif "room2" in folder_name or "room_b" in folder_name: 
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
        # Augmentation (Robustness)
        if self.augment:
            # 1. Amplitude Scaling (Simulate different signal strengths/distances)
            if np.random.rand() < 0.5:
                scale_factor = np.random.uniform(0.8, 1.2)
                rf_window = rf_window * scale_factor
                
            # 2. Gaussian Noise (Simulate RF interference)
            if np.random.rand() < 0.5:
                noise = np.random.normal(0, 0.05, rf_window.shape).astype(np.float32)
                rf_window = rf_window + noise

            # 3. Time Shift (Circular roll) - Simulate synchronization jitter
            if np.random.rand() < 0.3:
                shift = np.random.randint(-5, 5)
                rf_window = np.roll(rf_window, shift, axis=0)

            # 4. Random Dropout (Packet loss simulation)
            if np.random.rand() < 0.3:
                mask = np.random.rand(self.seq_len, 1) > 0.1 # 10% dropout
                rf_window = rf_window * mask
            
            # Clip again to ensure stability after augmentation
            rf_window = np.clip(rf_window, 0, 1)
                
        # To Torch
        # Ensure FLOAT32 (Fix for RuntimeError: Input type (double) and bias type (float))
        rf_tensor = torch.from_numpy(rf_window.astype(np.float32)) # [Seq, 64]
        # Transpose for 1D CNN: [Channels, Seq]
        rf_tensor = rf_tensor.permute(1, 0)
        
        return {
            "rf": rf_tensor,
            "pose": torch.tensor(lbl["pose"], dtype=torch.float32),
            "presence": torch.tensor([lbl["presence"]], dtype=torch.float32),
            "location": torch.tensor(lbl["location"], dtype=torch.long)
        }
