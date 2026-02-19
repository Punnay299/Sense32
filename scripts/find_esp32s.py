import socket
import struct
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Find ESP32 CSI Nodes")
    parser.add_argument("--port", type=int, default=8888, help="UDP Port to listen on")
    args = parser.parse_args()

    print(f"Listening for ESP32s on port {args.port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", args.port))
    except Exception as e:
        print(f"Error binding port: {e}")
        return

    sock.settimeout(1.0)
    
    devices = {} # IP -> Last Seen

    try:
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                ip = addr[0]
                
                if ip not in devices:
                    print(f"[NEW] Found Device: {ip}")
                
                devices[ip] = time.time()
                
                # Check for legacy (stale) devices
                now = time.time()
                # Clear screen or just print updates?
                # simple print for now
                
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        
    print("\n--- Summary ---")
    if not devices:
        print("No devices found. Ensure ESP32 is powered and traffic generator is running?")
    for ip, last_seen in devices.items():
        print(f"Device: {ip} (Last seen containing CSI data)")

if __name__ == "__main__":
    main()
