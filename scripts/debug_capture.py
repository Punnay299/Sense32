import os
import sys
import time
from scapy.all import sniff, get_if_list, conf

print("="*60)
print(" RF CAPTURE DIAGNOSTIC TOOL")
print("="*60)

# Check 1: Permissions
try:
    uid = os.getuid()
    print(f"[Check 1] User ID: {uid}")
    if uid != 0:
        print("FAIL: Not running as root (sudo). Scapy cannot sniff.")
        # sys.exit(1) # Continue anyway to show other info
    else:
        print("PASS: Running as root.")
except:
    print("WARN: Could not check UID.")

# Check 2: Interfaces
print("\n[Check 2] Available Interfaces:")
interfaces = get_if_list()
for iface in interfaces:
    print(f"  - {iface}")

# Check 3: Scapy Sniff
print("\n[Check 3] Attempting to sniff 5 packets (Timeout 5s)...")
try:
    packets = sniff(count=5, timeout=5)
    print(f"Result: Captured {len(packets)} packets.")
    if len(packets) > 0:
        print("PASS: Packet capture is WORKING.")
        print(f"Sample Packet Summary: {packets[0].summary()}")
    else:
        print("FAIL: No packets captured. Interface might be down or in wrong mode.")
except Exception as e:
    print(f"CRITICAL ERROR during sniff: {e}")

print("\n"="*60)
