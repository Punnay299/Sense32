
import unittest
import torch
import sys
import os
import shutil
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.networks import WifiPoseModel, TORCH_AVAILABLE

class TestIntegration(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch not installed")
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_training_step(self):
        """Perform a single synthetic training step to verify backward pass works."""
        model = WifiPoseModel(input_features=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion_pose = torch.nn.MSELoss()
        
        # Synthetic Batch: 4 samples, 50 timesteps, 2 features
        inputs = torch.randn(4, 50, 2)
        # Synthetic Labels: 4 samples, 34 coords (17*2)
        labels_pose = torch.randn(4, 34)
        labels_presence = torch.ones(4, 1)

        model.train()
        optimizer.zero_grad()
        
        pred_pose, pred_presence = model(inputs)
        
        loss = criterion_pose(pred_pose, labels_pose)
        loss.backward()
        optimizer.step()
        
        # Assert loss is a number (not NaN)
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.item() >= 0)

if __name__ == '__main__':
    unittest.main()
