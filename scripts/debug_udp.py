import socket
import time

def debug_udp():
    IP = "0.0.0.0"
    PORT = 8888
    
    print(f"Listening on {IP}:{PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        sock.bind((IP, PORT))
    except Exception as e:
        print(f"ERROR BINDING: {e}")
        return

    count = 0
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            print(f"[{count}] Received {len(data)} bytes from {addr}")
            print(f"    Header: {data[0:3]}  (Expected: b'CSI')")
            
            if count >= 5:
                print("Success! Packets are reaching Python.")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    sock.close()

if __name__ == "__main__":
    debug_udp()
