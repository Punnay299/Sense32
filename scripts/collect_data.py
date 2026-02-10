import argparse
import os
import sys
import time
import csv
import cv2
import queue
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture.camera import CameraCapture, MockCameraCapture
from src.capture.rf_interface import MockRFCapture, UDPRFCapture, LinuxPollingCapture, ScapyRFCapture, HybridRFCapture
from src.capture.csi_capture import CSICapture

from src.capture.beacon import LabelSyncBeacon

def main():
    parser = argparse.ArgumentParser(description="Collect synchronized RF and Video data.")
    parser.add_argument("--name", type=str, required=True, help="Session name (e.g. 'walk_01')")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--rf_mode", type=str, default="mock", choices=["mock", "udp", "linux", "scapy", "esp32", "hybrid"], help="RF Capture mode")

    parser.add_argument("--udp_port", type=int, default=5000, help="UDP Sniffer port (for UDPRFCapture)")
    parser.add_argument("--csi_port", type=int, default=8888, help="UDP CSI port (for CSICapture)")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera Device ID")
    parser.add_argument("--beacon", action="store_true", help="Enable Sync Beacon")
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    parser.add_argument("--mock_cam", action="store_true", help="Use Mock Camera")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup Output Directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join("data", f"session_{args.name}_{timestamp_str}")
    os.makedirs(session_dir, exist_ok=True)
    logging.info(f"Saving data to {session_dir}")

    # CSV Files
    f_rf = open(os.path.join(session_dir, "rf_data.csv"), "w", newline='')
    writer_rf = csv.writer(f_rf)
    writer_rf.writerow(["timestamp_monotonic_ms", "timestamp_device_ms", "source", "rssi", "rtt_ms", "mac_address", "ssid", "csi_amp", "csi_phase"])

    f_cam = open(os.path.join(session_dir, "camera_index.csv"), "w", newline='')
    writer_cam = csv.writer(f_cam)
    writer_cam.writerow(["frame_index", "timestamp_monotonic_ms", "timestamp_wall_ms"])

    f_beacon = None
    writer_beacon = None
    if args.beacon:
        f_beacon = open(os.path.join(session_dir, "beacons.csv"), "w", newline='')
        writer_beacon = csv.writer(f_beacon)
        writer_beacon.writerow(["seq_id", "timestamp_monotonic_ms", "payload"])

    # Initialize Modules
    if args.mock_cam:
        cam = MockCameraCapture(width=640, height=480)
    else:
        cam = CameraCapture(device_id=args.cam_id)
    
    if args.rf_mode == "mock":
        rf = MockRFCapture()
    elif args.rf_mode == "udp":
        rf = UDPRFCapture(port=args.udp_port)
    elif args.rf_mode == "linux":
        rf = LinuxPollingCapture()
    elif args.rf_mode == "scapy":
        rf = ScapyRFCapture()
    elif args.rf_mode == "esp32":
        rf = CSICapture(port=args.csi_port)
    elif args.rf_mode == "hybrid":
        # Hybrid: ESP32 (CSI) + Laptop (RSSI)
        rf = HybridRFCapture(csi_port=args.csi_port)

    else:
        # Default fallback
        rf = MockRFCapture()

    beacon = None
    if args.beacon:
        beacon = LabelSyncBeacon(port=5001)

    # Start Modules
    cam.start()
    rf.start()

    def beacon_log(seq, ts, payload):
        writer_beacon.writerow([seq, ts, payload])
        f_beacon.flush()

    if beacon:
        beacon.start(beacon_log)

    # Video Writer
    # Wait for first frame to get size
    while cam.read()[0] is None:
        time.sleep(0.1)
    
    first_frame, _, _ = cam.read()
    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(session_dir, "video.mp4")
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

    logging.info("Recording started. Press Ctrl+C to stop early.")
    start_time = time.time()
    frame_idx = 0
    last_frame_ts = 0
    packet_count = 0
    check_done = False

    try:
        while (time.time() - start_time) < args.duration:
            # 1. Process RF Queue
            while not rf.get_queue().empty():
                try:
                    pkt = rf.get_queue().get()
                    packet_count += 1
                    writer_rf.writerow([
                        pkt['timestamp_monotonic_ms'],
                        pkt.get('timestamp_device_ms', 0),
                        pkt.get('source', ''),
                        pkt.get('rssi', -100),
                        pkt.get('rtt_ms', -1),
                        pkt.get('mac_address', ''),
                        pkt.get('ssid', ''),
                        str(pkt.get('csi_amp', [])),
                        str(pkt.get('csi_phase', []))
                    ])
                except Exception as e:
                    logging.error(f"Failed to write RF packet: {e}")
            
            # 2. Process Camera
            frame, ts_mono, ts_wall = cam.read()
            
            # Only write if we have a valid frame AND it's a new frame (different timestamp)
            if frame is not None and ts_mono > last_frame_ts:
                last_frame_ts = ts_mono
                
                out_video.write(frame)
                writer_cam.writerow([frame_idx, ts_mono, ts_wall])
                frame_idx += 1
                
                # Show preview
                if not args.headless:
                    cv2.imshow('Recording', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Flush periodically
            if frame_idx % 30 == 0:
                f_rf.flush()
                f_cam.flush()
                
            # Aim for ~30FPS loop
            time.sleep(1/30.0)

            # Safety Check: If after 20 seconds we have 0 packets, WARN USER
            if not check_done and (time.time() - start_time) > 20.0:
                check_done = True
                if packet_count == 0:
                     logging.error("!!!" * 20)
                     logging.error("WARNING: NO RF DATA RECEIVED IN 20 SECONDS!")
                     logging.error("Check your interface/permissions. Output csv will be empty.")
                     logging.error("!!!" * 20)
                else:
                     logging.info(f"Verified: Received {packet_count} packets in first 20s. Capture looks good.")

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Stopping modules...")
        cam.stop()
        rf.stop()
        if beacon:
            beacon.stop()
        
        out_video.release()
        f_rf.close()
        f_cam.close()
        if f_beacon:
            f_beacon.close()
        cv2.destroyAllWindows()
        logging.info(f"Session saved to {session_dir}")

if __name__ == "__main__":
    main()
