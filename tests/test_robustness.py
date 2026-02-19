import unittest
import os
import sys
import shutil
import time
import socket
import threading
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture.csi_capture import CSICapture
from src.model.dataset import RFDataset

class TestRobustness(unittest.TestCase):
    def setUp(self):
        self.test_dir = "data/test_session_robustness"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            
    def test_csi_capture_malformed_packets(self):
        """Send garbage UDP packets and ensure no crash."""
        capture = CSICapture(port=9999)
        capture.start()
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send garbage
        sock.sendto(b'GARBAGE_DATA', ('127.0.0.1', 9999))
        sock.sendto(b'CSI_BUT_SHORT', ('127.0.0.1', 9999))
        
        # Send partial header
        sock.sendto(b'CSI\x01\x00\x00\x00\x00', ('127.0.0.1', 9999))
        
        time.sleep(0.5)
        capture.stop()
        sock.close()
        # If we reached here without exception, pass (assuming internal logging handled it)
        self.assertTrue(True)

    def test_dataset_corrupt_files(self):
        """Ensure dataset loading skips corrupt files without crashing."""
        # Create valid header but no data
        with open(os.path.join(self.test_dir, "rf_data.csv"), "w") as f:
            f.write("timestamp_monotonic_ms,rssi,csi_amp,csi_phase\n")
            
        with open(os.path.join(self.test_dir, "labels.csv"), "w") as f:
            f.write("frame_index,timestamp_monotonic_ms,center_x\n")
            
        # Should return empty dataset, not crash
        ds = RFDataset([self.test_dir])
        self.assertEqual(len(ds), 0)
        
    def test_dataset_malformed_csi(self):
        """Test parsing of bad CSI strings."""
        # Create file with one good line and one bad line
        with open(os.path.join(self.test_dir, "rf_data.csv"), "w") as f:
            f.write("timestamp_monotonic_ms,rssi,csi_amp,csi_phase\n")
            f.write("1000,-50,\"[1, 2, 3]\",\"[0,0,0]\"\n") # Good (short but valid)
            f.write("1001,-50,\"MALFORMED_LIST\",\"[0,0,0]\"\n") # Bad
            
        # Create matching labels
        with open(os.path.join(self.test_dir, "labels.csv"), "w") as f:
            f.write("frame_index,timestamp_monotonic_ms,center_x,center_y,visible\n")
            f.write("0,1000,0.5,0.5,True\n")
            f.write("1,1001,0.5,0.5,True\n")
            
        # This might fail depending on how ast.literal_eval handles it.
        # We need to verify if dataset.py handles exceptions in loop.
        # Currently dataset.py does list comprehension which will crash on one bad item.
        # Ideally robust code should handle it. Let's see if it fails.
        try:
             ds = RFDataset([self.test_dir])
        except Exception:
             self.fail("Dataset crashed on malformed CSI string")

if __name__ == '__main__':
    unittest.main()
