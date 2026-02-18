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
        # Input: Batch=2, Seq=50, Feat=256 (Dual Node: 128 Amp + 128 Phase)
        input_tensor = torch.randn(2, 50, 256)
        
        # Initialize with 256 input features
        model = WifiPoseModel(input_features=256, output_points=33)
        
        # Test default forward
        pose, presence, location = model(input_tensor)
        
        # Check shapes
        # Pose should be (2, 66) (33 points * 2)
        self.assertEqual(pose.shape, (2, 66))
        self.assertEqual(presence.shape, (2, 1))
        self.assertEqual(location.shape, (2, 4)) # 4 Classes

        
        # Check presence range [0, 1]
        self.assertTrue(torch.all(presence >= 0))
        self.assertTrue(torch.all(presence <= 1))

    def test_variable_sequence_length(self):
        # LSTM should handle variable lengths
        
        input_tensor = torch.randn(1, 100, 256)
        model = WifiPoseModel(input_features=256, output_points=33)
        pose, _, _ = model(input_tensor)
        self.assertEqual(pose.shape, (1, 66))

if __name__ == '__main__':
    unittest.main()
