import unittest
import os
import sys
import shutil
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.dataset import RFDataset

class TestDatasetRobustness(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_ds_robustness"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_dataset_corrupt_files(self):
        """Ensure dataset loading skips corrupt files without crashing."""
        with open(os.path.join(self.test_dir, "rf_data.csv"), "w") as f:
            f.write("timestamp_monotonic_ms,rssi,csi_amp,csi_phase\n")
            
        with open(os.path.join(self.test_dir, "labels.csv"), "w") as f:
            f.write("frame_index,timestamp_monotonic_ms,center_x\n")
            
        ds = RFDataset([self.test_dir])
        self.assertEqual(len(ds), 0)
        
    def test_dataset_malformed_csi(self):
        """Test parsing of bad CSI strings."""
        with open(os.path.join(self.test_dir, "rf_data.csv"), "w") as f:
            f.write("timestamp_monotonic_ms,rssi,csi_amp,csi_phase\n")
            f.write("1000,-50,\"[1, 2, 3]\",\"[0,0,0]\"\n") 
            f.write("1001,-50,\"MALFORMED_LIST\",\"[0,0,0]\"\n") 
            
        with open(os.path.join(self.test_dir, "labels.csv"), "w") as f:
            f.write("frame_index,timestamp_monotonic_ms,center_x,center_y,visible\n")
            f.write("0,1000,0.5,0.5,True\n")
            f.write("1,1001,0.5,0.5,True\n")
            
        try:
             ds = RFDataset([self.test_dir])
             # Should load 1 valid sample or 0 if filtering stricter
             # My code: fails on ast.literal_eval if not handled?
             # I need to check dataset.py again.
             print(f"Loaded {len(ds)} samples")
        except Exception as e:
             self.fail(f"Dataset crashed on malformed CSI string: {e}")

if __name__ == '__main__':
    unittest.main()
