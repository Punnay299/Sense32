import unittest
try:
    import torch
except ImportError:
    torch = None
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.model.networks import WifiPoseModel, TORCH_AVAILABLE
except ImportError as e:
    print(f"Import Error: {e}")
    TORCH_AVAILABLE = False
    WifiPoseModel = None
except Exception as e:
    print(f"Other Error: {e}")
    TORCH_AVAILABLE = False
    WifiPoseModel = None

class TestModel(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")
    def test_model_structure(self):
        # Input: Batch=2, Seq=50, Feat=2
        input_tensor = torch.randn(2, 50, 2)
        
        model = WifiPoseModel(input_features=2)
        
        # Test default forward
        pose, presence = model(input_tensor)
        
        # Check shapes
        # Pose should be (2, 34) by default based on code (17 points * 2)
        # Wait, in networks.py I defined PoseRegressor output as output_points * 2 (17*2=34)
        # But in train_local.py I redefined the head to 66 (33*2).
        # Let's test the default definition first.
        
        self.assertEqual(pose.shape, (2, 34))
        self.assertEqual(presence.shape, (2, 1))
        
        # Check presence range [0, 1]
        self.assertTrue(torch.all(presence >= 0))
        self.assertTrue(torch.all(presence <= 1))

    def test_variable_sequence_length(self):
        # LSTM should handle variable lengths but our CNN is 1D across time dim (dim 2 after permute)
        # CNN input (B, C, L). If L changes, Output L changes.
        # LSTM input (B, L, C). 
        # Wait, RFEncoder:
        # x = x.permute(0, 2, 1) -> (B, 2, L)
        # CNN: (B, 32, L) -> (B, 64, L)
        # x.permute(0, 2, 1) -> (B, L, 64)
        # LSTM input (B, L, 64) -> output (B, L, Hidden) -> Last hidden state
        # So yes, it should work with any length L.
        
        input_tensor = torch.randn(1, 100, 2)
        model = WifiPoseModel(input_features=2)
        pose, _ = model(input_tensor)
        self.assertEqual(pose.shape, (1, 34))

if __name__ == '__main__':
    unittest.main()
