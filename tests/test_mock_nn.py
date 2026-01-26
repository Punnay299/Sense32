
import unittest
import sys
import types
from unittest.mock import patch

class TestMockNN(unittest.TestCase):
    def test_mock_nn_fallback(self):
        """Test that RFEncoder initializes correctly even if torch is missing."""
        with patch.dict(sys.modules, {'torch': None, 'torch.nn': None}):
            # Removing torch from checks
            # We need to reload the module to trigger the conditional import logic
            if 'src.model.networks' in sys.modules:
                del sys.modules['src.model.networks']
            
            try:
                from src.model.networks import RFEncoder, TORCH_AVAILABLE, nn
                
                self.assertFalse(TORCH_AVAILABLE, "TORCH_AVAILABLE should be False when torch is missing")
                
                # Check that nn.Module is a class we can inherit from
                self.assertTrue(isinstance(nn.Module, type), "nn.Module should be a class/type")
                
                # Try instantiating RFEncoder
                # It should raise ImportError because the class enforces torch presence
                with self.assertRaises(ImportError):
                    encoder = RFEncoder()
                
            except ImportError as e:
                self.fail(f"Importing networks.py failed without torch: {e}")
            except TypeError as e:
                self.fail(f"RFEncoder inheritance failed: {e}")
                
if __name__ == '__main__':
    unittest.main()
