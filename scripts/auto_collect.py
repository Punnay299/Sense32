import argparse
import subprocess
import time
import os
import signal
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True, help="ESP32 IP")
    parser.add_argument("--name", type=str, required=True, help="Session Name")
    parser.add_argument("--duration", type=int, default=60, help="Seconds to record")
    args = parser.parse_args()
    
    print(f"Stats: Auto-Collecting Session '{args.name}' for {args.duration}s")
    
    # 1. Start Traffic Generator (Background)
    # We use the same rate as inference (100Hz)
    print(" >> Starting Traffic Generator...")
    gen_cmd = [sys.executable, "scripts/traffic_generator.py", "--ip", args.ip, "--rate", "100", "--duration", str(args.duration + 5)]
    gen_proc = subprocess.Popen(gen_cmd)
    
    # 2. Start Data Collector (Foreground)
    print(" >> Starting Data Collector...")
    # rf_mode esp32 port 8888 matches our setup
    col_cmd = [sys.executable, "scripts/collect_data.py", "--name", args.name, "--duration", str(args.duration), "--rf_mode", "esp32", "--csi_port", "8888"]
    
    try:
        col_proc = subprocess.run(col_cmd, check=True)
    except subprocess.CalledProcessError:
        print("Error in collection!")
    except KeyboardInterrupt:
        print("Stopping...")
        
    # Cleanup
    print(" >> Stopping Generator...")
    gen_proc.terminate()
    gen_proc.wait()
    
    print("Done.")

if __name__ == "__main__":
    main()
