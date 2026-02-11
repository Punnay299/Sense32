import socket
import sys

def listen_udp(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind to 0.0.0.0 to listen on all interfaces
    try:
        sock.bind(('0.0.0.0', port))
        print(f"Listening on 0.0.0.0:{port}...")
    except Exception as e:
        print(f"Error binding: {e}")
        return

    count = 0
    while True:
        try:
            data, addr = sock.recvfrom(4096)
            count += 1
            if count % 10 == 0:
                print(f"Received {count} packets. Last from {addr}, len={len(data)}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    port = 8888 # Default port
    listen_udp('0.0.0.0', port)
