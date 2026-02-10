import os
import time
import subprocess
import sys

# Configuration based on User Request
# Room 1: 3 Videos (Walk 30s, Sit 30s)
# Room 2: 1 Video (Walk 30s, Sit 30s) -> "1 video of second room"
# Hallway: 2 Videos (Walk between Room 1 & 2) -> "2 videos of hallway"
# Empty: 1 Video -> "1 video of empty room"

ZONES = [
    {"name": "room1_01", "desc": "Room 1 (Session 1): Walk 30s -> Sit 30s", "duration": 60},
    {"name": "room1_02", "desc": "Room 1 (Session 2): Walk 30s -> Sit 30s", "duration": 60},
    {"name": "room1_03", "desc": "Room 1 (Session 3): Walk 30s -> Sit 30s", "duration": 60},
    {"name": "room2_01", "desc": "Room 2 (Session 1): Walk 30s -> Sit 30s", "duration": 60},
    {"name": "hallway_01", "desc": "Hallway (Session 1): Between Rooms", "duration": 60},
    {"name": "hallway_02", "desc": "Hallway (Session 2): Between Rooms", "duration": 60},
    {"name": "empty_01", "desc": "Empty Room (Baseline): No one present", "duration": 60},
]

ESP32_IP = "10.42.0.173"

def main():
    print("=========================================")
    print("   Wi-Fi Sensing: Master Data Collector  ")
    print("=========================================")
    print(f"Target ESP32 IP: {ESP32_IP}")
    print("\nThis script will:")
    print("1. DELETE all existing data (as requested).")
    print("2. Guide you through collecting 7 datasets.")
    print("=========================================")
    
    confirm = input("Are you sure you want to DELETE ALL OLD DATA and start fresh? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborting.")
        return

    # Delete Data
    print("\n>> Deleting data/ directory contents...")
    os.system("rm -rf data/*")
    print(">> Done.")
    
    total = len(ZONES)
    for i, zone in enumerate(ZONES):
        print("\n" + "="*40)
        print(f"Task {i+1}/{total}: {zone['desc']}")
        print("="*40)
        
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
            "--ip", ESP32_IP
        ]
        
        subprocess.run(cmd)
        
        print("\n>> SAVED.")
        if i < total - 1:
            print("Take a break. Get ready for the next task.")
            
    print("\n" + "="*40)
    print("Collection Complete!")
    print("Next Steps:")
    print("1. python3 scripts/process_all_data.py")
    print("2. python3 scripts/train_local.py")
    print("="*40)

if __name__ == "__main__":
    main()
