import pandas as pd
import numpy as np
import ast
import os
import sys

def parse_csi(csi_str):
    try:
        if isinstance(csi_str, str):
            val = ast.literal_eval(csi_str)
        else:
            val = csi_str
            
        if isinstance(val, list):
            # Amplitude is usually first 64? Or is this magnitude?
            # csi_amp column is usually just amplitudes
            return np.array(val, dtype=np.float32)
    except:
        pass
    return np.zeros(64, dtype=np.float32)

def analyze_session(path, label_name):
    print(f"\nAnalyzing {label_name}: {os.path.basename(path)}")
    rf_path = os.path.join(path, "rf_data.csv")
    
    if not os.path.exists(rf_path):
        print("  RGB_data.csv not found")
        return

    df = pd.read_csv(rf_path)
    print(f"  Loaded {len(df)} rows.")
    
    if "source" not in df.columns:
        print("  No 'source' column.")
        return

    sources = df["source"].unique()
    print(f"  Sources found: {sources}")
    
    # Filter for A (149) and B (173)
    node_a_ip = "10.42.0.149"
    node_b_ip = "10.42.0.173"
    
    df_a = df[df["source"].str.contains(node_a_ip, na=False)]
    df_b = df[df["source"].str.contains(node_b_ip, na=False)]
    
    # Calculate Statistics
    def get_stats(sub_df):
        if len(sub_df) == 0: return 0.0, 0.0
        
        amps = []
        for x in sub_df["csi_amp"].values:
            arr = parse_csi(x)
            if len(arr) > 0:
                amps.append(np.mean(arr)) # Mean of subcarriers for this packet
        
        if len(amps) == 0: return 0.0, 0.0
        return np.mean(amps), np.std(amps)

    mean_a, std_a = get_stats(df_a)
    mean_b, std_b = get_stats(df_b)
    
    print(f"  Node A ({node_a_ip}) -> Mean: {mean_a:.2f}, StdDev (Motion): {std_a:.4f}")
    print(f"  Node B ({node_b_ip}) -> Mean: {mean_b:.2f}, StdDev (Motion): {std_b:.4f}")
    
    # Check Variance dominance
    if std_a > std_b:
        print(f"  Motion Check: Node A has MORE variation ({std_a:.4f} > {std_b:.4f})")
    else:
        print(f"  Motion Check: Node B has MORE variation ({std_b:.4f} > {std_a:.4f})")

def main():
    data_dir = "data"
    sessions = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Get latest Room A
    room_a = [s for s in sessions if "room_a" in s or "room1" in s]
    target_a = room_a[-1] if room_a else None
    
    # Get latest Room B
    room_b = [s for s in sessions if "room_b" in s or "room2" in s]
    target_b = room_b[-1] if room_b else None
    
    if target_a: analyze_session(target_a, "Room A Session")
    if target_b: analyze_session(target_b, "Room B Session")

if __name__ == "__main__":
    main()
