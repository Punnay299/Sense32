import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.dataset import RFDataset
from src.model.network import RFModel

def train(args):
    # Find all session folders
    sessions = [f for f in glob.glob(os.path.join(args.data_dir, "session_*")) if os.path.isdir(f)]
    print(f"Found {len(sessions)} sessions.")
    
    # Dataset
    full_ds = RFDataset(sessions, window_ms=1000, max_packets=100)
    if len(full_ds) == 0:
        print("No samples found. Run collect_data.py and pose_extractor.py first.")
        return

    # Split
    val_size = int(len(full_ds) * 0.2)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = RFModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion_coord = nn.MSELoss()
    criterion_presence = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    os.makedirs("models", exist_ok=True)

    print(f"Starting training on {device}...")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for rf, coord, vis in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            rf, coord, vis = rf.to(device), coord.to(device), vis.to(device)
            
            # Forward
            pred_coord, pred_presence = model(rf)
            
            # Loss: Only penalize coords if visible
            loss_c = (criterion_coord(pred_coord, coord) * vis).mean() + \
                     (criterion_coord(pred_coord, coord) * (1-vis) * 0.1).mean() # weak supervision on invisible? No, maybe just 0.
            
            # Actually, if not visible, coord is meaningless. 
            # But we want the model to predict *something*. 
            # Let's say: Loss = MSE(vis) + BCE(presence)
            # If vis=0, we ignore MSE.
            # However, PyTorch reduction is mean.
            
            loss_c = torch.mean(torch.sum((pred_coord - coord)**2, dim=1) * vis.squeeze()) / (vis.sum() + 1e-6)
            loss_p = criterion_presence(pred_presence, vis)
            
            loss = loss_c + loss_p
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for rf, coord, vis in val_loader:
                rf, coord, vis = rf.to(device), coord.to(device), vis.to(device)
                pred_coord, pred_presence = model(rf)
                
                loss_c = torch.mean(torch.sum((pred_coord - coord)**2, dim=1) * vis.squeeze()) / (vis.sum() + 1e-6)
                loss_p = criterion_presence(pred_presence, vis)
                val_loss += (loss_c + loss_p).item()
                
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_rf_model.pth")
            print("Saved Best Model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    train(args)
