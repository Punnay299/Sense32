import argparse
import os
import sys
import time
import collections
import logging
import cv2
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.capture.rf_interface import LinuxPollingCapture, MockRFCapture, ScapyRFCapture
from src.capture.camera import CameraCapture
from src.model.networks import WifiPoseModel
from src.vision.pose import PoseEstimator
from src.utils.normalization import AdaptiveScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_mode", default="esp32", choices=["mock", "linux", "scapy", "esp32"])
    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--port", type=int, default=8888, help="UDP Port for ESP32 CSI")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP IP to bind to")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load Model (Dual Node = 128 Features)
    model = WifiPoseModel(input_features=128, output_points=33).to(device)
    if os.path.exists(args.model):
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
            logging.info("Model loaded.")
        except:
             logging.warning("Model mismatch (likely old 64-feat model). using random weights for now.")
    else:
        logging.warning("Model not found, using random weights.")
    model.eval()
    
    # 2. Load Scaler
    scaler = AdaptiveScaler()
    scaler_path = "models/scaler.json"
    if os.path.exists(scaler_path):
        scaler.load(scaler_path)
    else:
        logging.warning("Scaler not found! Using uncalibrated data.")

    # 3. Init Sensors
    if args.rf_mode == "esp32":
        from src.capture.csi_capture import CSICapture
        rf = CSICapture(port=args.port, ip=args.ip)
    else:
        rf = MockRFCapture()

    cam = CameraCapture(device_id=0)
    pose_estimator = PoseEstimator() # Helper for "Hallway" logic

    rf.start()
    cam.start()

    # 4. State Management (Dual Node)
    nodes = {} # Map 'IP' -> 'NodeID' (A or B)
    node_buffers = {
        'A': collections.deque(maxlen=50),
        'B': collections.deque(maxlen=50)
    }
    node_variance = {'A': 0.0, 'B': 0.0}
    
    fps = 0
    frame_count = 0
    prev_time = time.time()
    
    logging.info("Waiting for Dual-Node Traffic...")

    try:
        while True:
            # --- RF Processing ---
            while not rf.get_queue().empty():
                pkt = rf.get_queue().get()
                src = pkt['source']
                
                # Auto-Assign Nodes
                if src not in nodes:
                    if len(nodes) == 0:
                        nodes[src] = 'A'
                        logging.info(f"Node A Detected: {src}")
                    elif len(nodes) == 1:
                        nodes[src] = 'B'
                        logging.info(f"Node B Detected: {src}")
                    else:
                        # Ignore 3rd node?
                        continue
                
                node_id = nodes[src]
                
                if 'csi_amp' in pkt and len(pkt['csi_amp']) >= 64:
                    raw_csi = np.array(pkt['csi_amp'][:64], dtype=np.float32)
                    
                    # Store Raw Variance (for Logic Rules)
                    # Simple variance of amplitude across subcarriers (or over time?)
                    # Over time is better, but instant variance across subcarriers also indicates multipath complexity.
                    # Let's use simple mean amplitude as a proxy for "Activity" or Variance of buffer later.
                    
                    # Normalize
                    csi_in = raw_csi.reshape(1, -1)
                    try:
                        csi_out = scaler.transform(csi_in)
                        csi = csi_out[0]
                    except:
                        csi = raw_csi / 127.0
                    
                    node_buffers[node_id].append(csi)

            # --- Frame Processing ---
            frame, _, _ = cam.read()
            if frame is None: continue
            h, w, _ = frame.shape

            # FPS
            frame_count += 1
            if time.time() - prev_time >= 1.0:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()

            # --- Logic Rules ---
            # 1. Camera visible?
            cam_results = pose_estimator.process_frame(frame)
            is_camera_visible = bool(cam_results.pose_landmarks)
            
            # 2. RF Variance (Activity Level)
            # 2. RF Variance (Activity Level)
            # Calculate variance of the LAST 50 frames in buffer
            
            for nid in ['A', 'B']:
                if len(node_buffers[nid]) > 10:
                    buff_arr = np.array(node_buffers[nid])
                    # Variance along time axis (activity)
                    var_val = np.mean(np.var(buff_arr, axis=0))
                    node_variance[nid] = var_val
                else:
                    node_variance[nid] = 0.0

            # --- Calibration Phase ---
            pass # Logic handled below in Decision Tree section
            
            # Calibration State
            if 'baseline' not in locals():
                 baseline = {'A': 0.001, 'B': 0.001} # Defaults
                 calib_buffer = {'A': [], 'B': []}
                 is_calibrated = False

            if not is_calibrated:
                if len(node_buffers['A']) > 10 and len(node_buffers['B']) > 10:
                    calib_buffer['A'].append(node_variance['A'])
                    calib_buffer['B'].append(node_variance['B'])
                    
                    if len(calib_buffer['A']) > 60: # ~2 seconds of data
                        baseline['A'] = np.mean(calib_buffer['A'])
                        baseline['B'] = np.mean(calib_buffer['B'])
                        is_calibrated = True
                        logging.info(f"Calibration Complete. Baselines: A={baseline['A']:.5f}, B={baseline['B']:.5f}")
                
                location_text = "CALIBRATING (Stand Still)"
                final_color = (0, 165, 255)
            
            else:
                # Normal Operation
                
                # Activity = Current Var - Baseline (how much MORE variance than static?)
                act_a = max(0, node_variance['A'] - baseline['A'])
                act_b = max(0, node_variance['B'] - baseline['B'])
                
                # Sensitivity Threshold (e.g. 2x baseline or fixed +0.005)
                # Let's use a fixed delta above baseline
                min_activity = 0.002 
                
                if is_camera_visible:
                    location_text = "Hallway (Visual)"
                    final_color = (0, 255, 255)
                elif act_a < min_activity and act_b < min_activity:
                     location_text = "Empty/Static"
                     final_color = (50, 50, 50)
                else:
                    # Compare Relative Activity
                    diff = act_a - act_b
                    
                    if diff > 0.002: 
                        location_text = "Room A (RF-A)"
                        final_color = (0, 255, 0)
                    elif diff < -0.002:
                        location_text = "Room B (RF-B)"
                        final_color = (255, 0, 0)
                    else:
                        location_text = "Uncertain/Both"
                        final_color = (255, 0, 255)
                        
                cv2.putText(frame, f"Act A: {act_a:.4f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Act B: {act_b:.4f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # --- Inference (If Buffers Full) ---
            if len(node_buffers['A']) == 50 and len(node_buffers['B']) == 50:
                # Merge: Stack (50, 64) and (50, 64) -> (50, 128)
                t_a = torch.tensor(np.array(node_buffers['A']), dtype=torch.float32)
                t_b = torch.tensor(np.array(node_buffers['B']), dtype=torch.float32)
                merged = torch.cat([t_a, t_b], dim=1) # [50, 128]
                
                input_tensor = merged.unsqueeze(0).to(device) # [1, 50, 128]
                
                with torch.no_grad():
                    pred_pose, pred_pres, pred_loc = model(input_tensor)
                
                pose_coords = pred_pose.cpu().numpy()[0]
                
                # Draw Stick Figure (RF)
                # Only draw if "present"
                if "Empty" not in location_text:
                    # Scale logic or simple draw
                     for i in range(0, len(pose_coords), 2):
                        x = int(pose_coords[i] * w)
                        y = int(pose_coords[i+1] * h)
                        if x > 0 and y > 0:
                            cv2.circle(frame, (x, y), 4, final_color, -1)

            # --- Visualization UI ---
            cv2.putText(frame, f"LOC: {location_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, final_color, 2)
            cv2.putText(frame, f"Var A: {node_variance['A']:.4f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"Var B: {node_variance['B']:.4f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

            # Missing Node Warning
            if len(nodes) < 2:
                cv2.putText(frame, f"Waiting for nodes... ({len(nodes)}/2)", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Valid IP Map display
            y_ip = 120
            # print(f"DEBUG NODES: {list(nodes.keys())}") 
            for ip, nid in nodes.items():
                 cv2.putText(frame, f"Node {nid}: {ip}", (10, y_ip), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                 y_ip += 20

            cv2.imshow("Dual-Node Wi-Fi Sensing", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        rf.stop()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
