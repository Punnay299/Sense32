import argparse
import os
import sys
import time

import cv2
import torch
import numpy as np
import collections
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.capture.rf_interface import LinuxPollingCapture, MockRFCapture, ScapyRFCapture

from src.capture.camera import CameraCapture
from src.model.networks import WifiPoseModel
from src.vision.pose import PoseEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_mode", default="mock", choices=["mock", "linux", "scapy", "esp32"])

    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--port", type=int, default=8888, help="UDP Port for ESP32 CSI")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP IP to bind to")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load Model (Input=2 features, Output=66 (33*2))
    # Match training: output_points=33
    # Neural Network (Trained on 64 features)
    model = WifiPoseModel(input_features=64, output_points=33).to(device)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        logging.info("Model loaded.")
    else:
        logging.warning("Model not found, using random weights.")
    
    model.eval()
    
    # Init Sensors
    if args.rf_mode == "linux":
        rf = LinuxPollingCapture()
        input_feats = 2
    elif args.rf_mode == "scapy":
        rf = ScapyRFCapture()
        input_feats = 2
    elif args.rf_mode == "esp32":
        from src.capture.csi_capture import CSICapture
        rf = CSICapture(port=args.port, ip=args.ip)
        input_feats = 64
    else:
        rf = MockRFCapture() # Mock needs update to send 64 feats if needed, or we adapt
        # For now, let's assume mock sends 2 feats and we pad, or we update mock.
        # Let's auto-pad in loop.
        input_feats = 64 # Model expects 64

        
    cam = CameraCapture(device_id=0)
    
    rf.start()
    cam.start()
    
    # Buffer for RF data (Seq Len = 50)
    rf_buffer = collections.deque(maxlen=50)
    
    locked_source = None
    if args.rf_mode == "esp32" and args.ip != "0.0.0.0":
         # If user specified IP, lock to it immediately if desired? No, let's auto-lock.
         pass
    
    # FPS Calculation
    prev_time = time.time()
    frame_count = 0
    fps = 0

    logging.info("Starting Inference Loop...")
    try:
        while True:
            # 1. Update RF Buffer (Limit specific number of packets per frame to avoid freezing)
            packets_processed = 0
            # Logic: Collect packets, sort by timestamp
            while not rf.get_queue().empty():
                pkt = rf.get_queue().get()
                
                # Auto-Lock Logic: Lock onto the first valid source
                if locked_source is None:
                     locked_source = pkt['source']
                     logging.info(f" locked_source set to: {locked_source}")
                     print(f">> LOCKED ONTO SOURCE: {locked_source}")
                
                # Filter unknown sources
                if pkt['source'] != locked_source:
                     continue
                
                # Check for CSI
                if 'csi_amp' in pkt and len(pkt['csi_amp']) > 0:
                     raw_csi = np.array(pkt['csi_amp'], dtype=np.float32)
                     packets_processed += 1   
                     # Debug Stats (Once per sec roughly)
                     if packets_processed == 1 and frame_count % 30 == 0:
                         logging.info(f"CSI Stats: Min={np.min(raw_csi):.1f}, Max={np.max(raw_csi):.1f}, Mean={np.mean(raw_csi):.1f}")

                     # Robust Normalization (Match Training Data!)
                     # Training uses: x / 127.0 -> clip(0, 1)
                     # OLD (Broken): Z-Score
                     csi = raw_csi / 127.0
                     csi = np.clip(csi, 0, 1)
                         
                     # Resize to 64 if needed
                     if len(csi) < 64:
                         csi = np.pad(csi, (0, 64-len(csi)))
                     elif len(csi) > 64:
                         csi = csi[:64]
                     rf_buffer.append(csi)
                else:
                    # Fallback or Mock
                    t = time.time()
                    vec = np.zeros(64, dtype=np.float32)
                    for i in range(64):
                        vec[i] = np.sin(i * 0.1 + t * 2)
                    rf_buffer.append(vec)
                
            # Log ONLY if buffer is filling up slowly or stalling
            if len(rf_buffer) < 50 and frame_count % 30 == 0:
                 logging.info(f"Buffered {len(rf_buffer)}/50 packets. Waiting for more...")

            # 2. Frame
            frame, _, _ = cam.read()
            if frame is None:
                continue
            
            h, w, _ = frame.shape

            # FPS
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                fps = frame_count / (curr_time - prev_time)
                frame_count = 0
                prev_time = curr_time
                logging.info(f"FPS: {fps:.2f} | Buffer: {len(rf_buffer)}/50")

            # 3. Ground Truth (MediaPipe) - For Hybrid Logic
            # We use MediaPipe to check if person is TRULY visible to camera
            # If so, we can show skeleton (and maybe verify RF).
            # If NOT, we rely on RF.
            
            # 4. Inference
            if len(rf_buffer) == 50:
                input_tensor = torch.FloatTensor([list(rf_buffer)]).to(device) # (1, 50, 64)
                with torch.no_grad():
                    pred_pose, pred_pres, pred_loc = model(input_tensor)
                    
                presence_prob = pred_pres.item()
                pose_coords = pred_pose.cpu().numpy()[0]
                loc = torch.argmax(pred_loc, dim=1).item()
                loc_names = ["Room 1", "Room 2", "Hallway", "Empty"]
                loc_text = loc_names[loc] if loc < len(loc_names) else "Unknown"
                
                # Color code
                if loc == 0: color = (0, 255, 0) # Green Room 1
                elif loc == 1: color = (255, 0, 0) # Blue Room 2
                elif loc == 2: color = (0, 255, 255) # Yellow Hallway
                else: color = (100, 100, 100) # Gray Empty
                
                # Hybrid Visualization Logic
                # Logic: 
                # If Presence > 0.5:
                #    If Camera sees person (we don't have direct boolean from camera class, need to run pose estimator)
                #    Run PoseEstimator on frame.
                #    If PoseEstimator finds person -> Draw Skeleton (Green) + "Visual Confirmed"
                #    Else -> Draw Skeleton (Red/RF) + "Non-LOS: " + Location
                
                # For efficiency/demo, let's just draw RF predictions 
                
                if presence_prob > 0.5:
                    
                    # Define connections (MediaPipe Pose topology)
                    CONNECTIONS = [
                        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), # Arms
                        (11, 23), (12, 24), (23, 24),                     # Torso
                        (23, 25), (25, 27), (27, 29), (29, 31),           # Left Leg
                        (24, 26), (26, 28), (28, 30), (30, 32)            # Right Leg
                    ]
                    
                    # Draw Lines
                    for start_idx, end_idx in CONNECTIONS:
                        if start_idx * 2 + 1 < len(pose_coords) and end_idx * 2 + 1 < len(pose_coords):
                            x1 = int(pose_coords[start_idx * 2] * w)
                            y1 = int(pose_coords[start_idx * 2 + 1] * h)
                            x2 = int(pose_coords[end_idx * 2] * w)
                            y2 = int(pose_coords[end_idx * 2 + 1] * h)
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                    # Draw keypoints
                    for i in range(0, len(pose_coords), 2):
                        x = int(pose_coords[i] * w)
                        y = int(pose_coords[i+1] * h)
                        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    
                    # Draw Status
                    cv2.putText(frame, f"RF: Person Detected ({presence_prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Verbose Debug Stats
                    y_off = 90
                    cv2.putText(frame, f"Presence Prob: {presence_prob:.2f}", (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    probs = torch.softmax(pred_loc, dim=1).cpu().numpy()[0]
                    for i, name in enumerate(loc_names):
                        p = probs[i]
                        c = (0, 255, 0) if i == loc else (200, 200, 200)
                        cv2.putText(frame, f"{name}: {p:.2f}", (10, y_off + 25 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                    
                else:
                    cv2.putText(frame, f"RF: Empty ({presence_prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
            else:
                 # If buffer is not full, we can keep the LAST prediction
                 # This prevents flickering "Buffering..." if packets drop for 1 frame
                 # But sticking is what the user complained about. 
                 # Let's show "Buffering" but maybe keep the drawing?
                 # No, user wants to know why it's stuck. Buffering is the truth.
                 msg = f"Buffering RF... {len(rf_buffer)}/50"
                 cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                 if len(rf_buffer) > 0:
                     # Show last packet info
                     cv2.putText(frame, "Receiving Data...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                 else:
                     cv2.putText(frame, "Waiting for packets...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Visualize Raw CSI (Small graph in bottom corner)
            if len(rf_buffer) > 0:
                last_csi = rf_buffer[-1] # shape (64,)
                # Draw box
                graph_x = 10
                graph_y = h - 110
                graph_w = 200
                graph_h = 100
                cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
                
                # Auto-scale for visualization
                c_min = np.min(last_csi)
                c_max = np.max(last_csi)
                if c_max - c_min < 0.001: c_max = c_min + 1 # avoid div/0
                
                # Plot points
                for i in range(len(last_csi) - 1):
                    # Normalize to 0-1 for graph
                    y1 = (last_csi[i] - c_min) / (c_max - c_min)
                    y2 = (last_csi[i+1] - c_min) / (c_max - c_min)
                    
                    pt1 = (int(graph_x + (i / 64.0) * graph_w), int(graph_y + graph_h - y1 * graph_h))
                    pt2 = (int(graph_x + ((i+1) / 64.0) * graph_w), int(graph_y + graph_h - y2 * graph_h))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
                cv2.putText(frame, f"CSI Range: [{c_min:.1f}, {c_max:.1f}]", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Wi-Fi Human Detection (Hybrid)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # time.sleep(0.01) # Remove sleep to go as fast as possible
            
    except KeyboardInterrupt:
        pass
    finally:
        rf.stop()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
