import socket
import struct
import numpy as np
import logging
import time
from .rf_interface import RFInterface

class CSICapture(RFInterface):
    """
    Captures CSI data from ESP32 via UDP.
    PROTOCOL:
    [Header: 3 bytes "CSI"]
    [RSSI: 1 byte (int8)]
    [Timestamp: 4 bytes (uint32)]
    [Len: 2 bytes (uint16)]
    [Payload: Len bytes] relative to ESP32 data format
    """
    def __init__(self, port=8888, ip="0.0.0.0", callback=None):
        super().__init__(callback)
        self.port = port
        self.ip = ip

        self.sock = None
        self.pkt_count = 0

    def _run(self):
        while self.running:
            # 1. Connection/Bind Loop
            self.sock = None
            while self.running:
                try:
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024) 
                    self.sock.bind((self.ip, self.port))
                    self.sock.settimeout(1.0)
                    logging.info(f"CSI Capture listening on {self.ip}:{self.port}")
                    break
                except Exception as e:
                    logging.error(f"Failed to bind UDP port {self.port}: {e}. Retrying in 2s...")
                    if self.sock:
                        self.sock.close()
                    time.sleep(2)
            
            if not self.running: break
            
            # 2. Receive Loop
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(4096)
                    # logging.debug(f"DEBUG: Pkt {len(data)} from {addr}")  
                    
                    if len(data) < 10: 
                        logging.warning(f"Short packet: {len(data)}")
                        continue

                    # Parse Header
                    if data[0:3] != b'CSI': 
                        logging.warning(f"Invalid Header: {data[0:3]}")
                        continue
                    
                    try:
                        rssi = struct.unpack('b', data[3:4])[0]
                        ts_device = struct.unpack('<I', data[4:8])[0]
                        data_len = struct.unpack('<H', data[8:10])[0]
                        
                        # logging.info(f"DEBUG: RSSI {rssi} Len {data_len}")
                        
                        if len(data) < 10 + data_len:
                            logging.warning(f"Incomplete packet from {addr}: needed {10+data_len}, got {len(data)}")
                            continue
                            
                        raw_payload = data[10:10+data_len]
                        
                        complex_data = np.frombuffer(raw_payload, dtype=np.int8)
                        
                        if len(complex_data) % 2 != 0:
                            complex_data = complex_data[:-1]
                        
                        real = complex_data[0::2]
                        imag = complex_data[1::2]
                        
                        csi_amp = np.sqrt(real.astype(np.float32)**2 + imag.astype(np.float32)**2).tolist()
                        csi_phase = np.arctan2(imag.astype(np.float32), real.astype(np.float32)).tolist()
                        
                        pkt = {
                            'source': f"esp32_{addr[0]}",
                            'timestamp_device_ms': ts_device,
                            'timestamp_monotonic_ms': time.monotonic() * 1000,
                            'rssi': int(rssi),
                            'mac_address': 'esp32_captured',
                            'csi_amp': csi_amp,
                            'csi_phase': csi_phase
                        }
                        
                        self._emit(pkt)
                        
                        self.pkt_count += 1
                        if self.pkt_count % 100 == 0:
                            logging.info(f"CSI Capture: Processed {self.pkt_count} packets. Last from {addr}")

                    except struct.error:
                         logging.warning("Malformed CSI packet header.")
                    except Exception as e:
                         logging.error(f"Error parsing CSI payload: {e}")
                         continue


                except socket.timeout:
                    # Just retry receiving, don't rebind
                    continue
                except OSError as e:
                    logging.error(f"UDP Socket error: {e}. Rebinding...")
                    break # Break inner loop, go to outer loop (rebind)
                except Exception as e:
                    logging.error(f"CSI UDP Error: {e}")
                    time.sleep(0.1)
            
            # Close socket before rebinding loop
            if self.sock:
                self.sock.close()
