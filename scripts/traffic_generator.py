import time
import socket
import argparse
import random
import sys

def main():
    parser = argparse.ArgumentParser(description="High-Rate UDP Traffic Generator for ESP32 CSI")
    parser.add_argument("--ip", type=str, required=True, help="ESP32 IP Address")
    parser.add_argument("--port", type=int, default=8888, help="Target Port (optional, ESP32 CSI callback captures all)")
    parser.add_argument("--rate", type=int, default=100, help="Packets per second (Hz)")
    parser.add_argument("--size", type=int, default=1000, help="Payload size in bytes")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=infinite)")
    
    args = parser.parse_args()
    
    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Payload
    payload = b'A' * args.size
    
    interval = 1.0 / args.rate
    
    print(f"Starting traffic generator: {args.rate} Hz -> {args.ip}:{args.port}")
    print("Press Ctrl+C to stop.")
    
    start_time = time.time()
    packet_count = 0
    
    try:
        while True:
            t0 = time.time()
            
            # Check duration
            if args.duration > 0 and (t0 - start_time) > args.duration:
                break
                
            # Send
            try:
                sock.sendto(payload, (args.ip, args.port))
                packet_count += 1
            except Exception as e:
                print(f"Send error: {e}")
                
            # Sleep remainder of interval
            t1 = time.time()
            elapsed = t1 - t0
            sleep_time = interval - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            if packet_count % 100 == 0:
                print(f"Sent {packet_count} packets...", end='\r')
                
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()
        print(f"Total Sent: {packet_count}")

if __name__ == "__main__":
    main()
