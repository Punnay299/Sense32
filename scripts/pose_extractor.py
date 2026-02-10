import argparse
import os
import sys
import csv
import cv2
import json
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.pose import PoseEstimator

def main():
    parser = argparse.ArgumentParser(description="Extract Pose Labels from Video Session.")
    parser.add_argument("--session", type=str, required=True, help="Path to session directory (e.g. data/session_...)")
    args = parser.parse_args()
    
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    video_path = os.path.join(args.session, "video.mp4")
    index_path = os.path.join(args.session, "camera_index.csv")
    
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return

    output_path = os.path.join(args.session, "labels.csv")
    
    # Initialize Pose Model
    pose_estimator = PoseEstimator(min_detection_confidence=0.5, model_complexity=2)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0 
    
    # Load Timestamp Index if available
    cam_timestamps = {}
    if os.path.exists(index_path):
        try:
            df_idx = pd.read_csv(index_path)
            # Map frame_index -> timestamp_monotonic_ms
            cam_timestamps = dict(zip(df_idx["frame_index"], df_idx["timestamp_monotonic_ms"]))
            logging.info(f"Loaded {len(cam_timestamps)} timestamps from index file.")
        except Exception as e:
            logging.warning(f"Failed to read camera_index.csv: {e}")
    
    logging.info(f"Processing {total_frames} frames from {video_path}...")
    
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "timestamp_monotonic_ms", "visible", "keypoints_flat", "center_x", "center_y"])
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate timestamp
            if frame_idx in cam_timestamps:
                ts_ms = cam_timestamps[frame_idx]
            else:
                # Fallback Estimate
                ts_ms = int((frame_idx / fps) * 1000)
            
            # Blind Labeling Logic (NLOS)
            # If "room2" or "hallway", we assume the user IS present but camera can't see them.
            # We force visible=True, but leave keypoints empty (zeros).
            is_nlos = "room2" in args.session.lower() or "hallway" in args.session.lower()
            
            if is_nlos:
                 # Force Presence
                 visible = True
                 # We use a special marker or just zeros. 
                 # dataset.py checks "if kps and len(kps) > 0: presence = 1.0"
                 # So we need to provide a dummy list of zeros to trigger presence=1
                 # 33 landmarks * 4 (x,y,z,vis) = 132 zeros
                 kps_list = [0.0] * 132 
                 cx, cy = 0.5, 0.5 # Default center
                 
                 writer.writerow([frame_idx, ts_ms, visible, json.dumps(kps_list), cx, cy])
                 
                 if frame_idx % 100 == 0:
                        print(f"Processed {frame_idx}/{total_frames} (Blind)", end='\r')
                 frame_idx += 1
                 continue

            try:
                results = pose_estimator.process_frame(frame)
                
                visible = False
                kps_list = []
                cx, cy = -1.0, -1.0
                
                if results.pose_landmarks:
                    visible = True
                    # Extract all 33 landmarks (x, y, z, visibility)
                    for lm in results.pose_landmarks.landmark:
                        kps_list.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    # Center of mass (approximate with hip center)
                    left_hip = results.pose_landmarks.landmark[23]
                    right_hip = results.pose_landmarks.landmark[24]
                    cx = (left_hip.x + right_hip.x) / 2
                    cy = (left_hip.y + right_hip.y) / 2
                
                writer.writerow([frame_idx, ts_ms, visible, json.dumps(kps_list), cx, cy])
                
            except Exception as e:
                logging.error(f"Error processing frame {frame_idx}: {e}")
                # Write empty row to maintain sync?
                # Best to write empty result
                writer.writerow([frame_idx, ts_ms, False, "[]", -1, -1])
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames}", end='\r')
            
            frame_idx += 1
            
    cap.release()
    logging.info(f"\nDone. Labels saved to {output_path}")

if __name__ == "__main__":
    main()
