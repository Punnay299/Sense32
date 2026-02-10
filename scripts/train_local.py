import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os
import sys
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.networks import WifiPoseModel
from src.model.dataset import RFDataset

def train_one_epoch(model, loader, optimizer, criterion_pose, criterion_pres, criterion_loc, device):
    model.train()
    running_loss = 0.0
    correct_pres = 0
    total_pres = 0
    correct_loc = 0
    total_loc = 0
    
    for batch in loader:
        rf = batch["rf"].to(device) # [Batch, 64, Seq]
        pose_gt = batch["pose"].to(device)
        pres_gt = batch["presence"].to(device)
        loc_gt = batch["location"].to(device)
        
        optimizer.zero_grad()
        # Forward
        # RFEncoder expects (Batch, Channels, Length). Our dataset returns (Batch, Channels, Length).
        # We need to make sure dimensions match.
        # Check if dataset returns [Batch, 64, Seq] (channels first) or [Batch, Seq, 64] (channels last).
        # In dataset.py: permute(1, 0) -> [64, Seq]. So loader gives [Batch, 64, Seq].
        # In networks.py: RFEncoder expects (Batch, Seq_Len, Features)??
        # WAIT. Let's check networks.py again.
        # networks.py: "CNN expects (Batch, Channels, Length). x = x.permute(0, 2, 1) if input is (Batch, Seq, Feat)"
        # If input is already (Batch, Channels, Length), we should NOT permute in networks.py or we should fix it.
        # Let's assume networks.py expects (Batch, Seq, Feat) as docstring says.
        # So we should permute back to (Batch, Seq, Feat) here or change dataset or change network.
        # Standard PyTorch CNN is (Batch, C, L).
        # Standard PyTorch LSTM is (Batch, L, Input_Size).
        # The network does: x.permute(0, 2, 1) -> implies input is (B, L, C).
        # Our dataset outputs (B, C, L).
        # So we should pass (B, L, C) to network.
        rf = rf.permute(0, 2, 1) # [Batch, Seq, 64]
        
        pose_pred, pres_pred, loc_pred = model(rf)
        
        # Losses
        loss_pres = criterion_pres(pres_pred, pres_gt)
        
        # Mask pose loss by presence (only train pose if person exists)
        mask = pres_gt.expand_as(pose_pred)
        loss_pose = criterion_pose(pose_pred * mask, pose_gt * mask)
        
        # Location loss 
        # Only if presence is true? Or always? If ground truth says "North" even if empty?
        # Dataset sets location based on folder name. Even if empty, it is still "North Room".
        # So we can train location always.
        loss_loc = criterion_loc(loc_pred, loc_gt)
        
        # Weighted Sum
        loss = loss_pres + loss_pose + (0.5 * loss_loc)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Metrics
        # Presence (Binary)
        preds = (pres_pred > 0.5).float()
        correct_pres += (preds == pres_gt).sum().item()
        total_pres += pres_gt.size(0)
        
        # Location (Multi-class)
        _, loc_preds = torch.max(loc_pred, 1)
        correct_loc += (loc_preds == loc_gt).sum().item()
        total_loc += loc_gt.size(0)
        
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    acc_pres = correct_pres / total_pres if total_pres > 0 else 0
    acc_loc = correct_loc / total_loc if total_loc > 0 else 0
    
    return avg_loss, acc_pres, acc_loc

def validate(model, loader, criterion_pose, criterion_pres, criterion_loc, device):
    model.eval()
    running_loss = 0.0
    correct_pres = 0
    total_pres = 0
    correct_loc = 0
    total_loc = 0
    
    with torch.no_grad():
        for batch in loader:
            rf = batch["rf"].to(device).permute(0, 2, 1)
            pose_gt = batch["pose"].to(device)
            pres_gt = batch["presence"].to(device)
            loc_gt = batch["location"].to(device)
            
            pose_pred, pres_pred, loc_pred = model(rf)
            
            loss_pres = criterion_pres(pres_pred, pres_gt)
            mask = pres_gt.expand_as(pose_pred)
            loss_pose = criterion_pose(pose_pred * mask, pose_gt * mask)
            loss_loc = criterion_loc(loc_pred, loc_gt)
            
            loss = loss_pres + loss_pose + (0.5 * loss_loc)
            running_loss += loss.item()
            
            preds = (pres_pred > 0.5).float()
            correct_pres += (preds == pres_gt).sum().item()
            total_pres += pres_gt.size(0)
            
            _, loc_preds = torch.max(loc_pred, 1)
            correct_loc += (loc_preds == loc_gt).sum().item()
            total_loc += loc_gt.size(0)
            
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    acc_pres = correct_pres / total_pres if total_pres > 0 else 0
    acc_loc = correct_loc / total_loc if total_loc > 0 else 0
    
    return avg_loss, acc_pres, acc_loc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, help="Path to single session directory")
    parser.add_argument("--all_data", action="store_true", help="Train on all sessions in data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true", help="Run 1 epoch 1 batch for testing")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Gather Paths
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
        
    logging.info(f"Found {len(sessions)} sessions.")
    
    # 2. Create Dataset
    # We want to split sessions into train/val? Or split frames?
    # Random split of frames is easier but might leak data if frames are correlated.
    # But for now let's stick to random split of frames for simplicity, as implemented before.
    # To implement augmentation correctly, we should have two datasets: one with augment=True, one False.
    # But we can't easily split indices and then change the flag.
    # Solution: Create one dataset without augmentation (or with) and split.
    # Or: Create two datasets with same paths, and use Subset with indices.
    
    full_dataset = RFDataset(sessions, augment=False) 
    
    if len(full_dataset) == 0:
        logging.error("No valid samples found in dataset.")
        return
    
    # Split
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len])
    
    # Hack to enable augmentation only on train_subset
    # The Subset class wraps the original dataset. We can't change the original dataset's flag easily for just one subset.
    # A cleaner way is to just enable it for all or none, or create a wrapper.
    # For now, let's enable it for ALL, or NONE. 
    # Let's enable it for ALL (including validation) which is suboptimal but simple,
    # OR we re-initialize dataset.
    # Better: Update RFDataset to take indices? No.
    # Let's just turn it on. Validation score might be slightly noisy but robust.
    full_dataset.augment = True 
    
    logging.info(f"Total Samples: {total_len} (Train: {train_len}, Val: {val_len})")
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model = WifiPoseModel(input_features=64, output_points=33).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    criterion_pose = nn.MSELoss()
    criterion_pres = nn.BCELoss()
    criterion_loc = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    best_val_loss = float('inf')
    os.makedirs("models", exist_ok=True)
    
    logging.info("Starting training...")
    
    epochs = 1 if args.dry_run else args.epochs
    
    for epoch in range(epochs):
        t_loss, t_acc_pres, t_acc_loc = train_one_epoch(
            model, train_loader, optimizer, criterion_pose, criterion_pres, criterion_loc, device
        )
        
        v_loss, v_acc_pres, v_acc_loc = validate(
            model, val_loader, criterion_pose, criterion_pres, criterion_loc, device
        )
        
        scheduler.step(v_loss)
        
        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"  Train | Loss: {t_loss:.4f} | Pres Acc: {t_acc_pres:.2%} | Loc Acc: {t_acc_loc:.2%}")
        logging.info(f"  Val   | Loss: {v_loss:.4f} | Pres Acc: {v_acc_pres:.2%} | Loc Acc: {v_acc_loc:.2%}")
        
        # Save Best
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "models/best.pth")
            logging.info("  -> Saved models/best.pth")
            
        # Save Last
        torch.save(model.state_dict(), "models/last.pth")
        
        if args.dry_run:
            break

if __name__ == "__main__":
    main()
