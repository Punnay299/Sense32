
from scapy.all import sniff, RadioTap, Ether
import logging
import sys

# Setup logging
logging.basicConfig(format='%(message)s', level=logging.INFO)

def pkt_callback(pkt):
    if pkt.haslayer(RadioTap):
        print(f"[RadioTap] RSSI: {pkt[RadioTap].dBm_AntSignal} | Len: {len(pkt)}")
        sys.exit(0) # Success
    elif pkt.haslayer(Ether):
        print(f"[Ethernet] No RSSI (Cooked Frame) | Len: {len(pkt)}")
    else:
        print(f"[Unknown] {pkt.summary()}")

if __name__ == "__main__":
    iface = "wlp8s0" # Hardcoded based on user logs
    print(f"Sniffing on {iface}...")
    try:
        sniff(iface=iface, prn=pkt_callback, count=10, timeout=5)
        print("Timeout reached.")
    except Exception as e:
        print(f"Error: {e}")
