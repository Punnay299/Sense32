import os
import pandas as pd
import ast

def check_data_quality():
    base_dir = "data"
    sessions = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if "session_" in d]
    
    print(f"Checking {len(sessions)} sessions for valid CSI data...")
    
    total_valid = 0
    total_empty = 0
    
    for s in sessions:
        rf_path = os.path.join(s, "rf_data.csv")
        if not os.path.exists(rf_path):
            continue
            
        try:
            df = pd.read_csv(rf_path)
            if "csi_amp" not in df.columns:
                print(f"[BAD] {os.path.basename(s)}: No CSI column.")
                continue
                
            valid_count = 0
            for x in df["csi_amp"]:
                if isinstance(x, str) and len(x) > 2 and x != "[]":
                     valid_count += 1
            
            if valid_count > 100:
                print(f"[GOOD] {os.path.basename(s)}: {valid_count} valid packets.")
                total_valid += 1
            else:
                print(f"[EMPTY] {os.path.basename(s)}: Only {valid_count} packets.")
                total_empty += 1
                
        except Exception as e:
            print(f"[ERROR] {os.path.basename(s)}: {e}")
            
    print(f"\nSummary: {total_valid} Good Sessions, {total_empty} Empty/Bad Sessions.")
    if total_valid == 0:
        print("CONCLUSION: All previous data is likely invalid (Empty).")
    else:
        print("CONCLUSION: Some data is usable, but new data is recommended.")

if __name__ == "__main__":
    check_data_quality()
