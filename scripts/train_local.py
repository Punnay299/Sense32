import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import sys
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.networks import WifiPoseModel

class WiFiPoseDataset(Dataset):
    def __init__(self, rf_path, labels_path, seq_len=50):
        self.seq_len = seq_len
        
        # Load Data
        logging.info("Loading data...")
        try:
            rf_df = pd.read_csv(rf_path)
            labels_df = pd.read_csv(labels_path)
        except Exception as e:
            logging.error(f"Error loading CSVs: {e}")
            self.valid_indices = []
            return

        # Preprocess RF
        # Sort by mono timestamp
        rf_df = rf_df.sort_values("timestamp_monotonic_ms")
        labels_df = labels_df.sort_values("timestamp_monotonic_ms") if "timestamp_monotonic_ms" in labels_df.columns else labels_df.sort_values("frame_index")
        
        # Add timestamp to labels if missing (using frame index logic or assuming synced)
        # Note: collect_data.py writes frame_index, timestamp_monotonic_ms to camera_index.csv
        # labels.csv has frame_index. We need to merge.
        
        session_dir = os.path.dirname(labels_path)
        index_path = os.path.join(session_dir, "camera_index.csv")
        if os.path.exists(index_path):
            cam_idx = pd.read_csv(index_path)
            labels_df = pd.merge(labels_df, cam_idx[["frame_index", "timestamp_monotonic_ms"]], on="frame_index")
        
        self.rf_data = rf_df[["rssi", "rtt_ms"]].fillna(-100).values
        self.rf_times = rf_df["timestamp_monotonic_ms"].values
        
        # Normalize RF: RSSI (-100 to 0) -> 0 to 1
        self.rf_data[:, 0] = (self.rf_data[:, 0] + 100) / 100.0
        # Normalize RTT: (0 to 1000ms) -> 0 to 1
        self.rf_data[:, 1] = np.clip(self.rf_data[:, 1], 0, 1000) / 1000.0
        
        self.labels = []
        self.valid_indices = []
        
        # Alignment: For each label (frame), find preceding RF window
        logging.info("Aligning RF and Video...")
        for i in range(len(labels_df)):
            ts = labels_df.iloc[i]["timestamp_monotonic_ms"]
            
            # Find RF index close to ts
            # Optimization: Use searchsorted
            idx = np.searchsorted(self.rf_times, ts)
            
            if idx >= self.seq_len:
                # We have enough history
                self.valid_indices.append((idx, len(self.labels)))
                
                # Parse label
                kps_json = labels_df.iloc[i]["keypoints_flat"]
                try:
                    kps = json.loads(kps_json)
                    if len(kps) == 0:
                        # Empty (no person)
                        self.labels.append({"pose": np.zeros(66), "presence": 0.0})
                    else:
                        # kps format: [x, y, z, vis, ...]
                        # We want just x, y for 17 points -> 34 values? 
                        # MediaPipe Body has 33 points. 33 * 4 = 132 values.
                        # Let's just fit the flat vector provided.
                        # For simplicity, let's just take x, y of first 17 (COCO style) or all 33.
                        # Model expects fixed output. Let's output all 33*2 = 66
                        
                        # Extract x, y
                        flat_xy = []
                        for j in range(0, len(kps), 4):
                            flat_xy.append(kps[j])   # x
                            flat_xy.append(kps[j+1]) # y
                        
                        # Check length, pad or cut
                        # Check length, pad or cut
                        target_len = 66 # 33 points * 2
                        if len(flat_xy) >= target_len:
                            flat_xy = flat_xy[:target_len]
                        else:
                            flat_xy = flat_xy + [0.0]*(target_len - len(flat_xy))
                            
                        self.labels.append({"pose": np.array(flat_xy), "presence": 1.0})
                except:
                   self.labels.append({"pose": np.zeros(66), "presence": 0.0})

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        rf_end_idx, label_idx = self.valid_indices[idx]
        
        # Get RF window
        rf_window = self.rf_data[rf_end_idx-self.seq_len : rf_end_idx]
        
        label = self.labels[label_idx]
        
        return {
            "rf": torch.FloatTensor(rf_window),
            "pose": torch.FloatTensor(label["pose"]),
            "presence": torch.FloatTensor([label["presence"]])
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, help="Path to single session directory")
    parser.add_argument("--all_data", action="store_true", help="Train on all sessions in data/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    sessions = []
    if args.all_data:
        base_dir = "data"
        if os.path.exists(base_dir):
            for name in os.listdir(base_dir):
                path = os.path.join(base_dir, name)
                if os.path.isdir(path) and "session_" in name:
                    sessions.append(path)
    elif args.session:
        sessions.append(args.session)
        
    if not sessions:
        logging.error("No sessions found. Use --session or --all_data")
        return
        
    datasets = []
    for s in sessions:
        logging.info(f"Loading session: {s}")
        rf_path = os.path.join(s, "rf_data.csv")
        labels_path = os.path.join(s, "labels.csv")
        
        if not os.path.exists(rf_path) or not os.path.exists(labels_path):
            logging.warning(f"Skipping {s}: Missing csv files")
            continue
            
        d = WiFiPoseDataset(rf_path, labels_path)
        if len(d) > 0:
            datasets.append(d)
            
    if not datasets:
         logging.error("No valid datasets loaded.")
         return
         
    
    # Split Train/Val (80/20)
    final_dataset = torch.utils.data.ConcatDataset(datasets)
    total_len = len(final_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(final_dataset, [train_len, val_len])
    
    logging.info(f"Total Samples: {total_len} (Train: {train_len}, Val: {val_len})")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Model (Input=2 features, Output=66 (33*2))
    model = WifiPoseModel(input_features=2, output_points=33).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion_pose = nn.MSELoss()
    criterion_pres = nn.BCELoss()
    
    logging.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss = 0
        for batch in train_loader:
            rf = batch["rf"].to(device)
            pose_gt = batch["pose"].to(device)
            pres_gt = batch["presence"].to(device)
            
            optimizer.zero_grad()
            pose_pred, pres_pred = model(rf)
            
            loss_pres = criterion_pres(pres_pred, pres_gt)
            mask = pres_gt.expand_as(pose_pred)
            loss_pose = criterion_pose(pose_pred * mask, pose_gt * mask)
            
            loss = loss_pres + loss_pose
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # VALIDATE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                rf = batch["rf"].to(device)
                pose_gt = batch["pose"].to(device)
                pres_gt = batch["presence"].to(device)
                
                pose_pred, pres_pred = model(rf)
                
                loss_pres = criterion_pres(pres_pred, pres_gt)
                mask = pres_gt.expand_as(pose_pred)
                loss_pose = criterion_pose(pose_pred * mask, pose_gt * mask)
                
                loss = loss_pres + loss_pose
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best.pth")
            
    logging.info(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")
    logging.info("Model saved to models/best.pth")

if __name__ == "__main__":
    main()

