import threading
import os
import queue
import time
import json
import socket
import math
import random
import logging
import subprocess
import re
import platform
import shlex
from abc import ABC, abstractmethod

# Try importing Scapy, handle if not present
try:
    from scapy.all import sniff, RadioTap, Dot11
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

if SCAPY_AVAILABLE:
    from scapy.all import sniff, RadioTap, Dot11, Dot11Beacon, Dot11ProbeResp


class RFInterface(ABC):
    def __init__(self, callback=None):
        """
        :param callback: Function to call when data receives (optional)
                         Signature: callback(data_dict)
        """
        self.callback = callback
        self.running = False
        self.thread = None
        self.packet_queue = queue.Queue() # For polling usage

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"{self.__class__.__name__} started.")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info(f"{self.__class__.__name__} stopped.")

    @abstractmethod
    def _run(self):
        pass

    def _emit(self, data):
        """Helper to send data to callback and queue"""
        # Add monotonic timestamp if not present
        if 'timestamp_monotonic_ms' not in data:
            data['timestamp_monotonic_ms'] = time.monotonic() * 1000
        
        if self.callback:
            self.callback(data)
        self.packet_queue.put(data)

    def get_queue(self):
        return self.packet_queue


class MockRFCapture(RFInterface):
    """Generates synthetic RSSI data simulating a person walking."""
    def _run(self):
        t0 = time.time()
        while self.running:
            # Simulate walking approach/retreat using sine wave
            now = time.time()
            elapsed = now - t0
            
            # Base RSSI around -60dBm, varying between -40 and -80
            # Period of 10 seconds for a "pass"
            raw_rssi = -60 + 20 * math.sin(elapsed * 2 * math.pi / 10.0)
            
            # Add noise
            noisy_rssi = raw_rssi + random.uniform(-5, 5)
            
            data = {
                'source': 'mock_gen',
                'timestamp_device_ms': now * 1000,
                'mac_address': '00:11:22:33:44:55',
                'rssi': int(noisy_rssi),
                'csi_amp': [], # CSI not simulated in simple mock yet
                'csi_phase': []
            }
            self._emit(data)
            time.sleep(0.05) # 20Hz


class UDPRFCapture(RFInterface):
    """
    Listens for line-delimited JSON over UDP.
    Format expected: {"rssi": -55, "mac": "...", "ts": 12345, "csi": [...]}
    """
    def __init__(self, port=5000, ip="0.0.0.0", callback=None):
        super().__init__(callback)
        self.port = port
        self.ip = ip
        self.sock = None

    def _run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        self.sock.settimeout(1.0)
        logging.info(f"UDP Sniffer listening on {self.ip}:{self.port}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                line = data.decode('utf-8').strip()
                if not line:
                    continue
                    
                try:
                    # Parse JSON
                    pkt = json.loads(line)
                    # Normalize keys
                    out = {
                        'source': f"udp_{addr[0]}",
                        'timestamp_device_ms': pkt.get('ts', 0),
                        'rssi': pkt.get('rssi', -100),
                        'mac_address': pkt.get('mac', 'unknown'),
                        'csi_amp': pkt.get('csi_amp', []),
                        'csi_phase': pkt.get('csi_phase', [])
                    }
                    self._emit(out)
                except json.JSONDecodeError:
                    logging.warning(f"Malformed UDP packet: {line}")
                    
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"UDP Socket error: {e}")
                time.sleep(1)

        self.sock.close()


class LinuxPollingCapture(RFInterface):
    """
    Active Polling Capture for Linux.
    Uses /proc/net/wireless for RSSI and ping for RTT.
    """
    def __init__(self, target_ip="8.8.8.8", interface=None, callback=None):
        super().__init__(callback)
        self.target_ip = target_ip
        
        # Robust Interface Detection
        if interface is None:
            self.interface = self._detect_interface()
            logging.info(f"Auto-detected Wi-Fi Interface: {self.interface}")
        else:
            self.interface = interface
            
        # Signal Smoothing (Simple Moving Average)
        self.rssi_history = []
        self.history_len = 5

    def _detect_interface(self):
        """Finds the first wireless interface."""
        try:
             # Try /proc/net/wireless first
             with open("/proc/net/wireless", "r") as f:
                 for line in f:
                     if ":" in line:
                         # Format: " wlan0: ..."
                         return line.split(":")[0].strip()
        except: pass
        
        # Fallback to standard
        return "wlan0"

    def _get_rssi(self):
        try:
            # /proc/net/wireless is efficient
            # Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE
            #  face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22
            #   wlp2s0: 0000   45.  -65.  -256        0      0      0      0      0        0
            
            with open("/proc/net/wireless", "r") as f:
                content = f.read()
            
            for line in content.splitlines():
                if self.interface in line or ":" in line:
                    parts = line.split()
                    # Determine which column is level. 
                    # Usually parts[0] is Interface:
                    # parts[2] is link, parts[3] is level (dBm)
                    
                    # Robust finding: look for negative number approx -30 to -100
                    # Standard format: Interface: status link level noise ...
                    # wlan0: 0000 50. -60. -256
                    if len(parts) >= 4:
                        level_str = parts[3].rstrip('.')
                        val = float(level_str)
                        self.rssi_history.append(val)
                        if len(self.rssi_history) > self.history_len:
                            self.rssi_history.pop(0)
                        return sum(self.rssi_history) / len(self.rssi_history)
        except Exception:
            pass
            
        # Fallback: iwconfig
        try:
            cmd = f"iwconfig {shlex.quote(self.interface)}"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            match = re.search(r"Signal level=(-?\d+)", output)
            if match:
                return float(match.group(1))
        except:
            pass
            
        # Fallback 2: nmcli (Common on Fedora/Ubuntu)
        # Output: "        signal:                                 70" (scale 0-100) or bars
        # nmcli -f IN-USE,SSID,SIGNAL dev wifi
        try:
            cmd = "nmcli -f IN-USE,SIGNAL dev wifi"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            # Look for line starting with '*' (in use)
            for line in output.splitlines():
                if line.strip().startswith('*'):
                    # parts: *, SSID, SIGNAL
                    parts = line.split()
                    if len(parts) >= 3:
                        signal_strength = int(parts[-1])
                        # Convert 0-100 to dBm? nmcli signal is often quality.
                        # Approx: dBm = (quality / 2) - 100
                        return (signal_strength / 2.0) - 100.0
        except:
            pass

        # Fallback 3: iw (Modern Linux)
        try:
            cmd = f"iw dev {self.interface} link"
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            match = re.search(r"signal:\s+(-?\d+)", output)
            if match:
                return float(match.group(1))
        except:
            pass

        return -100.0


    def _get_rtt(self):
        try:
            cmd = ["ping", "-c", "1", "-W", "0.2", self.target_ip]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
            match = re.search(r"time=(\d+(?:\.\d+)?)", output)
            if match:
                return float(match.group(1))
        except:
            return -1.0
        return -1.0

    def _run(self):
        logging.info("Starting Linux Polling Capture...")
        while self.running:
            start_t = time.time()
            
            rssi = self._get_rssi()
            rtt = self._get_rtt()
            
            data = {
                'source': 'linux_poll',
                'timestamp_device_ms': time.time() * 1000,
                'rssi': rssi,
                'rtt_ms': rtt,
                'mac_address': 'router',
                'csi_amp': [],
                'csi_phase': []
            }
            self._emit(data)
            
            # Simple rate limiting
            delta = time.time() - start_t
            if delta < 0.05:
                time.sleep(0.05 - delta)

class ScapyRFCapture(RFInterface):
    """
    High-Performance Passive Sniffer using Scapy.
    Captures raw RSSI from Beacon frames.
    Requires Root/Sudo privileges and Monitor Mode usually.
    """
    def __init__(self, interface=None, callback=None):
        super().__init__(callback)
        self.interface = interface
        if self.interface is None:
            self.interface = self._detect_interface()
        
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy not installed. Run 'pip install scapy'")

    def _detect_interface(self):
        """Finds the first wireless interface."""
        # Method 1: Check /sys/class/net/ for wireless interfaces
        try:
            interfaces = os.listdir('/sys/class/net/')
            for iface in interfaces:
                # Common wireless prefixes/logic
                path = os.path.join('/sys/class/net/', iface, 'wireless')
                if os.path.exists(path) or iface.startswith('wl'):
                     logging.info(f"Auto-detected Interface: {iface}")
                     return iface
        except: pass

        try:
             # Try /proc/net/wireless first
             with open("/proc/net/wireless", "r") as f:
                 for line in f:
                     if ":" in line:
                         # Format: " wlan0: ..."
                         return line.split(":")[0].strip()
        except: pass
        
        # Fallback to standard
        logging.warning("No wireless interface detected. Defaulting to wlan0.")
        return "wlan0"

            
    def _packet_handler(self, pkt):
        if not self.running: return
        
        if pkt.haslayer(RadioTap):
            # Extract RSSI
            # RadioTap field 'dBm_AntSignal' is common
            try:
                rssi = pkt[RadioTap].dBm_AntSignal
                # Sometimes it's a byte, sometimes signed int. 
                # Scapy usually handles decoding, but let's ensure it's reasonable
                if rssi is None: return
            except:
                return
                
            # Extract MAC
            mac = "unknown"
            ssid = ""
            if pkt.haslayer(Dot11):
                mac = pkt[Dot11].addr2 # Source address (Sender)
                
            if pkt.haslayer(Dot11Beacon):
                try:
                    ssid = pkt[Dot11Beacon].info.decode('utf-8', errors='ignore')
                except: pass
                
            data = {
                'source': 'scapy_sniff',
                'timestamp_device_ms': time.time() * 1000,
                'rssi': int(rssi),
                'rtt_ms': -1, # RTT not available in passive sniffing
                'mac_address': mac,
                'ssid': ssid,
                'csi_amp': [], 
                'csi_phase': []
            }
            self._emit(data)

    def _channel_hopper(self):
        """Rotates between common 2.4GHz channels to find traffic."""
        channels = [1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 5, 10]
        i = 0
        while self.running:
            try:
                ch = channels[i]
                subprocess.run(["iw", "dev", self.interface, "set", "channel", str(ch)], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                i = (i + 1) % len(channels)
            except: pass
            time.sleep(0.5) # Hop every 500ms

    def _run(self):
        logging.info(f"Starting Scapy Sniffer on {self.interface}...")
        
        # Ensure Interface is Up
        try:
             subprocess.run(["ip", "link", "set", self.interface, "up"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except: pass

        # Attempt to Enable Monitor Mode (Best Effort)
        try:
             # Check if iw exists
             subprocess.run(["iw", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
             
             # Try setting monitor mode
             logging.info(f"Attempting to enable Monitor Mode on {self.interface}...")
             subprocess.run(["ip", "link", "set", self.interface, "down"], check=True)
             subprocess.run(["iw", "dev", self.interface, "set", "type", "monitor"], check=True)
             subprocess.run(["ip", "link", "set", self.interface, "up"], check=True)
             logging.info("Monitor mode enabled successfully.")
        except Exception as e:
             logging.warning(f"Could not set Monitor Mode (optional but recommended): {e}")
             # Try bringing up anyway
             try: subprocess.run(["ip", "link", "set", self.interface, "up"])
             except: pass
             
        # Start Channel Hopper
        self.hopper_thread = threading.Thread(target=self._channel_hopper)
        self.hopper_thread.daemon = True
        self.hopper_thread.start()

        try:
            # monitor=True is handled by kwargs, but interface needs to be in monitor mode os-side mostly.
            # We assume user (or guide) sets monitor mode if they want all traffic.
            # But standard managed mode can capture beacons if on valid channel.
            sniff(iface=self.interface, prn=self._packet_handler, store=False, 
                  stop_filter=lambda x: not self.running)
        except OSError as e:
            if "Network is down" in str(e):
                logging.error(f"Interface {self.interface} is DOWN.")
                logging.info("Attempting to bring it UP...")
                try:
                     subprocess.check_call(["ip", "link", "set", self.interface, "up"])
                     logging.info("Interface brought UP. Retrying sniff...")
                     sniff(iface=self.interface, prn=self._packet_handler, store=False, 
                           stop_filter=lambda x: not self.running)
                except Exception as e2:
                     logging.error(f"Failed to bring interface up: {e2}")
            else:
                 logging.error(f"Scapy Sniff failed: {e}")
        except Exception as e:
            logging.error(f"Scapy Sniff failed: {e}")
            logging.error("Ensure you are running with 'sudo' and interface exists.")






