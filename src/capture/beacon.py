import socket
import time
import threading
import json
import logging

class LabelSyncBeacon:
    def __init__(self, port=5001, interval=2.0, broadcast_ip='<broadcast>'):
        self.port = port
        self.interval = interval
        self.broadcast_ip = broadcast_ip
        self.sock = None
        self.running = False
        self.thread = None
        self.seq = 0
        self.log_callback = None # Fn(seq, ts_monotonic, payload)

    def start(self, log_callback):
        """
        Start broadcasting beacons.
        :param log_callback: Function accepting (seq_id, timestamp_monotonic_ms, payload_str)
        """
        self.log_callback = log_callback
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"Beacon started on port {self.port}")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
        logging.info("Beacon stopped.")

    def _run(self):
        while self.running:
            try:
                ts_mono = time.monotonic() * 1000
                payload = {
                    "type": "SYNC_BEACON",
                    "id": "LAPTOP_MASTER",
                    "seq": self.seq,
                    "ts_sent": ts_mono
                }
                msg = json.dumps(payload).encode('utf-8')
                
                self.sock.sendto(msg, (self.broadcast_ip, self.port))
                
                # Log usage
                if self.log_callback:
                    self.log_callback(self.seq, ts_mono, str(payload))
                
                self.seq += 1
                time.sleep(self.interval)
                
            except Exception as e:
                logging.error(f"Beacon send error: {e}")
                time.sleep(1.0)
