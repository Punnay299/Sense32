import time
import socket
import argparse
import threading
import sys

def send_traffic(target_ip, port, rate, size, duration):
    """Sends UDP traffic to a specific target."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload = b'A' * size
    interval = 1.0 / rate
    
    print(f"[Thread] Sending to {target_ip}:{port} at {rate} Hz")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while True:
            t0 = time.time()
            
            if duration > 0 and (t0 - start_time) > duration:
                break
                
            try:
                sock.sendto(payload, (target_ip, port))
                packet_count += 1
            except Exception as e:
                print(f"[{target_ip}] Send error: {e}")
                
            t1 = time.time()
            elapsed = t1 - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            if packet_count % 200 == 0:
                 # Print status update occasionally
                 pass
                 
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"[{target_ip}] Finished. Sent: {packet_count}")

def main():
    parser = argparse.ArgumentParser(description="High-Rate UDP Traffic Generator for ESP32 CSI")
    parser.add_argument("--ip", type=str, required=True, help="ESP32 IP Address (comma-separated for multiple)")
    parser.add_argument("--port", type=int, default=8888, help="Target Port (optional, ESP32 CSI callback captures all)")
    parser.add_argument("--rate", type=int, default=100, help="Packets per second (Hz)")
    parser.add_argument("--size", type=int, default=10, help="Payload size in bytes (Small = Better Rate)")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=infinite)")
    
    args = parser.parse_args()
    
    targets = [ip.strip() for ip in args.ip.split(',')]
    
    print(f"Starting Multi-Target Traffic Generator")
    print(f"Targets: {targets}")
    print(f"Rate: {args.rate} Hz per target")
    print("Press Ctrl+C to stop.")
    
    threads = []
    
    try:
        for ip in targets:
            t = threading.Thread(target=send_traffic, args=(ip, args.port, args.rate, args.size, args.duration))
            t.daemon = True
            t.start()
            threads.append(t)
            
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all threads...")
        # Threads are daemon, will exit when main exits
    
if __name__ == "__main__":
    main()
