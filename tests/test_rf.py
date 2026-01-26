import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.capture.rf_interface import MockRFCapture, LinuxPollingCapture

class TestRFCapture(unittest.TestCase):
    def test_mock_capture(self):
        rf = MockRFCapture()
        rf.start()
        time.sleep(0.2) 
        rf.stop()
        
        q = rf.get_queue()
        self.assertFalse(q.empty(), "Mock RF queue should not be empty")
        
        pkt = q.get()
        self.assertIn('rssi', pkt)
        self.assertIn('source', pkt)
        self.assertEqual(pkt['source'], 'mock_gen')

    @patch('builtins.open')
    @patch('subprocess.check_output')
    def test_linux_polling_capture(self, mock_subprocess, mock_open):
        # Mock /proc/net/wireless
        mock_file = MagicMock()
        # Header + Data line
        mock_file.__enter__.return_value.read.return_value = \
            "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n" \
            " face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n" \
            " wlan0: 0000   50.  -60.  -256        0      0      0      0      0        0"
        mock_open.return_value = mock_file
        
        # Mock ping
        mock_subprocess.return_value.decode.return_value = "64 bytes from 8.8.8.8: seq=1 ttl=118 time=12.5 ms"
        
        rf = LinuxPollingCapture(interface="wlan0")
        rf.start()
        time.sleep(0.15)
        rf.stop()
        
        q = rf.get_queue()
        self.assertFalse(q.empty(), "Linux RF queue should not be empty")
        
        pkt = q.get()
        self.assertEqual(pkt['source'], 'linux_poll')
        self.assertEqual(pkt['rssi'], -60.0)
        self.assertEqual(pkt['rtt_ms'], 12.5)

if __name__ == '__main__':
    unittest.main()
