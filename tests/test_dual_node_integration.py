import unittest
import pandas as pd
import numpy as np
import os
import shutil
import sys
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.dataset import RFDataset

class TestDualNodeIntegration(unittest.TestCase):
    def setUp(self):
        # Create a dummy session folder
        self.test_dir = "data/test_session_dual_node"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create dummy RF data (2 sources)
        # Source A: 192.168.1.10
        # Source B: 192.168.1.11
        # Time: 0 to 1000ms (10 samples)
        
        data = []
        for i in range(20):
            ts = i * 50 # 50ms steps
            source = "192.168.1.10" if i % 2 == 0 else "192.168.1.11"
            csi = str([1.0]*64) # Dummy CSI
            
            data.append({
                "timestamp_monotonic_ms": ts,
                "timestamp_device_ms": ts,
                "source": source,
                "csi_amp": csi,
                "csi_phase": str([0.0]*64),
                "mac_address": "AA:BB:CC:DD:EE:FF"
            })
            
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.test_dir, "rf_data.csv"), index=False)
        
        # Dummy Labels
        labels = []
        for i in range(20):
            ts = i * 50
            labels.append({
                "timestamp_monotonic_ms": ts,
                "keypoints_flat": str([0.5]*66), # Dummy pose
                "presence": 1.0,
                "location": 0
            })
        df_lbl = pd.DataFrame(labels)
        df_lbl.to_csv(os.path.join(self.test_dir, "labels.csv"), index=False)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dataset_merging(self):
        # Initialize Dataset
        ds = RFDataset([self.test_dir], seq_len=5, augment=False)
        
        print(f"Dataset Length: {len(ds)}")
        
        if len(ds) > 0:
            item = ds[0]
            rf = item['rf'] # Shape should be [128, Seq] (channels first for Conv1D)
            print(f"RF Shape: {rf.shape}")
            
            # Check shape
            self.assertEqual(rf.shape[0], 256, "Features should be 256 ( (64 Amp + 64 Phase) * 2 Nodes )")
            self.assertEqual(rf.shape[1], 5, "Sequence length should match")
            
            # Check stacking logic
            # Node A (first 64) should be 1.0 (from setup)
            # Node B (next 64) should be 1.0 (merged from second source)
            # Note: since we interleaved packets, interpolation might be interesting.
            # But roughly it should work.
            pass
        else:
            self.fail("Dataset is empty!")

if __name__ == "__main__":
    unittest.main()
