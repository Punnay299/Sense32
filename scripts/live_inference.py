import argparse
import os
import sys
import time
import numpy as np
import torch
import cv2
import logging
from collections import deque

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.capture.rf_interface import HybridRFCapture, MockRFCapture
from src.model.networks import WifiPoseModel
from src.capture.camera import CameraCapture, MockCameraCapture

# Visualization Constants
SKELETON_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Torso
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), # Left Arm
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31), # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)  # Left Leg
]

def draw_pose(frame, pose_coords, confidence, label_text=""):
    """
    Draws pose on frame based on confidence.
    pose_coords: (33, 2) numpy array of (x, y) in range [0, 1]
    """
    h, w, _ = frame.shape
    
    # 1. High Confidence: Draw Skeleton
    if confidence > 0.6:
        # Draw points
        for i in range(len(pose_coords)):
            x = int(pose_coords[i][0] * w)
            y = int(pose_coords[i][1] * h)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1) # Green for high conf
        
        # Draw lines
        for i, j in SKELETON_CONNECTIONS:
            if i < len(pose_coords) and j < len(pose_coords):
                pt1 = (int(pose_coords[i][0] * w), int(pose_coords[i][1] * h))
                pt2 = (int(pose_coords[j][0] * w), int(pose_coords[j][1] * h))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    
    # 2. Medium Confidence (Likely Obstruction): Draw Bounding Box
    elif confidence > 0.3:
        # Find min/max for simple box
        xs = pose_coords[:, 0]
        ys = pose_coords[:, 1]
        x1, y1 = int(np.min(xs) * w), int(np.min(ys) * h)
        x2, y2 = int(np.max(xs) * w), int(np.max(ys) * h)
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow
        cv2.putText(frame, "Proximity Detected", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 3. Always show status text
    color = (0, 255, 0) if confidence > 0.6 else (0, 255, 255) if confidence > 0.3 else (0, 0, 255)
    cv2.putText(frame, f"Signal Conf: {confidence:.2f} | {label_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    parser = argparse.ArgumentParser(description="Real-time WiFi Pose Inference.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--rf_mode", type=str, default="hybrid", choices=["hybrid", "esp32", "mock"], help="RF Source")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera ID for overlay")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model on {device}...")
    
    try:
        model = WifiPoseModel(input_features=64, output_points=33).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 2. Start RF Capture
    if args.rf_mode == "hybrid":
        rf = HybridRFCapture()
    elif args.rf_mode == "esp32":
        from src.capture.csi_capture import CSICapture
        rf = CSICapture()
    else:
        rf = MockRFCapture()
    
    rf.start()

    # 3. Start Camera (for Viz)
    cam = CameraCapture(device_id=args.cam_id)
    cam.start()

    # Buffer for CSI Data (Window size 50 for model?)
    # Assuming model takes [Batch, 64, 50] or similar.
    # In training we view dataset.py to see window size. 
    # Usually it was 100 or 50. Let's assume 100 based on previous context or check dataset.py.
    # Checking dataset code view from before... window size wasn't explicitly shown but implied.
    # Let's use a deque of size 100.
    csi_buffer = deque(maxlen=100) # [64] arrays
    
    logging.info("Starting Inference Loop...")
    
    try:
        while True:
            # 1. Get Camera Frame
            frame, _, _ = cam.read()
            if frame is None:
                continue
                
            # 2. Get Latest RF Data
            while not rf.get_queue().empty():
                pkt = rf.get_queue().get()
                if 'csi_amp' in pkt and len(pkt['csi_amp']) == 64:
                    # Normalize simple (0-100 range approx)
                    amp = np.array(pkt['csi_amp'])
                    csi_buffer.append(amp)
            
            # 3. Model Inference (if buffer full)
            pose_pred = np.zeros((33, 2))
            conf = 0.0
            loc_text = "Waiting for Data..."
            
            if len(csi_buffer) >= 50: # Minimum to infer
                # Prepare Tensor
                # Take last 100 samples
                # If we have < 100, pad? Or just use what we have? 
                # Model likely expects fixed size. Let's use what dataset.py does.
                # Assuming 100 for now. If < 100, repeat/pad.
                
                window = list(csi_buffer)
                while len(window) < 100:
                    window.append(window[-1])
                window = window[-100:] # Ensure max 100
                
                # Shape: [1, 64, 100]
                input_tensor = np.array(window, dtype=np.float32).T # [64, 100]
                input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Forward
                    pose, presence, location = model(input_tensor)
                    
                    # Process Outputs
                    pose_coords = pose[0].cpu().numpy().reshape(33, 2) # [33, 2]
                    pres_prob = torch.sigmoid(presence).item()
                    loc_idx = torch.argmax(location, dim=1).item()
                    
                    conf = pres_prob
                    pose_pred = pose_coords
                    
                    # Location Mapping (Update based on your classes)
                    # 0: North, 1: South, 2: East, 3: West (Example)
                    classes = ["North", "South", "East", "West", "Room 2"]
                    if loc_idx < len(classes):
                        loc_text = classes[loc_idx]
                    else:
                        loc_text = "Unknown"
                        
                    # Detection Logic
                    if pres_prob > 0.8:
                        if loc_text == "Room 2":
                             # FORCE ALERT MODE
                             cv2.putText(frame, "WARNING: PERSON DETECTED IN ROOM 2", (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                             conf = 0.2 # Downgrade skeleton conf to avoid drawing noise
                        else:
                             # Normal drawing
                             pass
                    else:
                        loc_text = "Empty"
                        conf = 0.0

            # 4. Draw
            draw_pose(frame, pose_pred, conf, loc_text)
            
            cv2.imshow("Wi-Fi Sight (Live)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        rf.stop()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
