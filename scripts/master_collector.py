import os
import time
import subprocess
import sys
import socket
import argparse

# ------------------------------------------------------------------
# CONFIGURATION: 15 Sessions Total
# ------------------------------------------------------------------
ZONES = []

# Room A: 5 Sessions
for i in range(1, 6):
    ZONES.append({"name": f"room_a_{i:02d}", "desc": f"Room A (Session {i}/5): Randomized Walk/Sit", "duration": 60})

# Room B: 5 Sessions
for i in range(1, 6):
    ZONES.append({"name": f"room_b_{i:02d}", "desc": f"Room B (Session {i}/5): Randomized Walk/Sit", "duration": 60})

# Hallway: 4 Sessions
for i in range(1, 5):
    ZONES.append({"name": f"hallway_{i:02d}", "desc": f"Hallway (Session {i}/4): Transition/Loitering", "duration": 60})

# Empty (Noise): 5 Sessions (Crucial for AI Suppression)
for i in range(1, 6):
    ZONES.append({"name": f"empty_noise_{i:02d}", "desc": f"Empty Noise (Session {i}/5): NO HUMANS in Room A. Silence.", "duration": 60, "no_cam": True})


def find_esp32_ips(port=8888, duration=5):
    """Listens for ESP32 packets to auto-discover IPs."""
    print(f"\n>> Scanning for ESP32s on Port {port} for {duration} seconds...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.0)
    try:
        sock.bind(("0.0.0.0", port))
    except Exception as e:
        print(f"!! Error binding port {port}: {e}")
        return []

    devices = set()
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            try:
                data, addr = sock.recvfrom(4096)
                ip = addr[0]
                if ip not in devices:
                    print(f"   [FOUND] ESP32 at {ip}")
                    devices.add(ip)
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
    
    return list(devices)

def main():
    print("=========================================")
    print("   Wi-Fi Sensing: Master Data Collector  ")
    print("       (Scaled: 15 Sessions)             ")
    print("=========================================")
    
    # 1. Discovery
    ips = find_esp32_ips()
    
    if len(ips) == 0:
        print("!! NO ESP32s FOUND. Ensure they are powered on and sending data.")
        manual = input("Enter IPs manually (comma separated) or 'q' to quit: ")
        if manual.lower() == 'q': return
        ips = [x.strip() for x in manual.split(',')]
    
    ip_str = ",".join(ips)
    print(f"\n>> Target IPs: {ip_str}")
    
    # 2. Cleanup Confirmation
    print("\nThis script will:")
    print("1. DELETE all existing data (as requested).")
    print("2. Guide you through collecting 15 datasets.")
    print("=========================================")
    
    confirm = input("Are you sure you want to DELETE ALL OLD DATA and start fresh? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborting.")
        return

    # Delete Data
    print("\n>> Deleting data/ directory contents...")
    os.system("rm -rf data/*")
    print(">> Done.")
    
    # 3. Collection Loop
    total = len(ZONES)
    for i, zone in enumerate(ZONES):
        print("\n" + "="*50)
        print(f"Task {i+1}/{total}: {zone['desc']}")
        print("="*50)
        
        input(f"Press ENTER when you are ready to start recording ({zone['duration']}s)...")
        
        # Countdown
        for j in range(3, 0, -1):
            print(f"Starting in {j}...")
            time.sleep(1)
        
        print(">> RECORDING STARTED!")
        
        cmd = [
            sys.executable, "scripts/auto_collect.py",
            "--name", zone['name'],
            "--duration", str(zone['duration']),
            "--ip", ip_str # Passing Comma-Separated IPs
        ]
        
        if zone.get("no_cam", False):
            cmd.append("--no_cam")
        
        subprocess.run(cmd)
        
        print("\n>> SAVED.")
        if i < total - 1:
            print("Take a break. Get ready for the next task.")
            
    print("\n" + "="*50)
    print("Collection Complete!")
    print("Next Steps:")
    print("1. python3 scripts/process_all_data.py --force_relabel")
    print("2. python3 scripts/train_local.py --all_data --epochs 50")
    print("=========================================")

if __name__ == "__main__":
    main()
