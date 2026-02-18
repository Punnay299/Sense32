
import unittest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.pose import PoseEstimator

class TestVision(unittest.TestCase):
    def setUp(self):
        # We can test with a dummy frame
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    @patch('src.vision.pose.mp_pose')
    def test_pose_estimator_init(self, mock_mp_pose):
        """Test that PoseEstimator initializes without error."""
        # Setup mock
        mock_mp_pose.Pose.return_value = MagicMock()
        
        try:
            estimator = PoseEstimator()
            self.assertIsNotNone(estimator)
        except Exception as e:
            self.fail(f"PoseEstimator init failed: {e}")

    @patch('src.vision.pose.mp_pose')
    def test_process_frame_structure(self, mock_mp_pose):
        """Test that process_frame returns expected structure."""
        # Setup mock return values
        mock_pose_instance = MagicMock()
        mock_mp_pose.Pose.return_value = mock_pose_instance
        
        # Mock results
        mock_results = MagicMock()
        mock_results.pose_landmarks.landmark = [MagicMock(x=0.5, y=0.5, z=0.0, visibility=1.0)] * 33
        mock_pose_instance.process.return_value = mock_results
        
        estimator = PoseEstimator()
        results = estimator.process_frame(self.frame)
        
        flat = estimator.get_keypoints_flat(results)
        self.assertIsNotNone(flat)
        self.assertEqual(len(flat), 33 * 4)
            
        com = estimator.get_center_of_mass(results)
        self.assertIsNotNone(com)
        self.assertEqual(len(com), 2)


if __name__ == '__main__':
    unittest.main()
