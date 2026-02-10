
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock scapy before import
sys.modules['scapy.all'] = MagicMock()
from src.capture.rf_interface import ScapyRFCapture

class TestScapyCapture(unittest.TestCase):
    @patch('src.capture.rf_interface.SCAPY_AVAILABLE', True)
    @patch('src.capture.rf_interface.open', create=True) 
    def test_initialization(self, mock_open):
        # Mock /proc/net/wireless read
        mock_open.return_value.__enter__.return_value = ["  wlan0: ..."]
        
        # Test 1: Explicit interface
        rf = ScapyRFCapture(interface="mon0")
        self.assertEqual(rf.interface, "mon0")


    @patch('src.capture.rf_interface.SCAPY_AVAILABLE', True)
    def test_packet_handler(self):
        rf = ScapyRFCapture(interface="wlan0")
        rf.running = True
        
        # Mock Packet
        pkt = MagicMock()
        pkt.haslayer.return_value = True 
        
        # Universal Layer Mock
        # scapy pkt[Layer] returns a layer object.
        # We make it return the SAME mock for all layers, which has all needed attributes.
        layer_mock = MagicMock()
        layer_mock.dBm_AntSignal = -55
        layer_mock.addr2 = "00:11:22:33:44:55"
        layer_mock.info = b"TestSSID"
        
        pkt.__getitem__.return_value = layer_mock
        
        # Capture generated data
        emitted_data = []
        rf.callback = lambda d: emitted_data.append(d)
        
        rf._packet_handler(pkt)
        
        self.assertEqual(len(emitted_data), 1)
        data = emitted_data[0]
        self.assertEqual(data['rssi'], -55)
        self.assertEqual(data['source'], 'scapy_sniff')

        self.assertEqual(data['ssid'], 'TestSSID')

if __name__ == '__main__':
    unittest.main()
