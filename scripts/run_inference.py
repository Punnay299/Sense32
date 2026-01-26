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
from src.capture.rf_interface import LinuxPollingCapture, MockRFCapture
from src.capture.camera import CameraCapture
from src.model.networks import WifiPoseModel
from src.vision.pose import PoseEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_mode", default="mock", choices=["mock", "linux"])
    parser.add_argument("--model", default="models/best.pth")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load Model (Input=2 features, Output=66 (33*2))
    # Match training: output_points=33
    model = WifiPoseModel(input_features=2, output_points=33).to(device)
    
    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, map_location=device))
        logging.info("Model loaded.")
    else:
        logging.warning("Model not found, using random weights.")
    
    model.eval()
    
    # Init Sensors
    if args.rf_mode == "linux":
        rf = LinuxPollingCapture()
    else:
        rf = MockRFCapture()
        
    cam = CameraCapture(device_id=0)
    pose_estimator = PoseEstimator() # For ground truth visualization comparison
    
    rf.start()
    cam.start()
    
    # Buffer for RF data (Seq Len = 50)
    rf_buffer = collections.deque(maxlen=50)
    
    logging.info("Starting Inference Loop...")
    try:
        while True:
            # 1. Update RF Buffer
            while not rf.get_queue().empty():
                pkt = rf.get_queue().get()
                # Normalize features same as training
                rssi_norm = (pkt.get('rssi', -100) + 100) / 100.0
                rtt_norm = np.clip(pkt.get('rtt_ms', -1), 0, 1000) / 1000.0
                rf_buffer.append([rssi_norm, rtt_norm])
                
            # 2. Frame
            frame, _, _ = cam.read()
            if frame is None:
                continue
                
            # 3. Ground Truth (Green)
            results = pose_estimator.process_frame(frame)
            frame = pose_estimator.draw_landmarks(frame, results)
                
            # 4. Inference
            if len(rf_buffer) == 50:
                input_tensor = torch.FloatTensor([list(rf_buffer)]).to(device) # (1, 50, 2)
                with torch.no_grad():
                    pred_pose, pred_pres = model(input_tensor)
                    
                presence_prob = pred_pres.item()
                pose_coords = pred_pose.cpu().numpy()[0]
                
                # Visualize Prediction (Red)
                # If presence high
                if presence_prob > 0.5:
                    h, w, _ = frame.shape
                    
                    # Draw points
                    for i in range(0, len(pose_coords), 2):
                        x = int(pose_coords[i] * w) # Assuming normalization 0-1, but model output is linear. 
                        y = int(pose_coords[i+1] * h) # We should normalized labels in training to 0-1.
                        # Assuming training did normalize (we put raw coords in training, need to fix training normalization if raw pixels used)
                        # Wait, in training I didn't normalize pixels. I used MediaPipe output. 
                        # MediaPipe x,y are normalized [0,1]. Perfect.
                        
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    
                    cv2.putText(frame, f"RF Presence: {presence_prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "RF: No Person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:
                 cv2.putText(frame, "Buffering RF...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Wi-Fi Human Detection", frame)
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
