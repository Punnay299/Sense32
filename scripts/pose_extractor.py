import argparse
import os
import sys
import csv
import cv2
import json
import logging
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vision.pose import PoseEstimator

def main():
    parser = argparse.ArgumentParser(description="Extract Pose Labels from Video Session.")
    parser.add_argument("--session", type=str, required=True, help="Path to session directory (e.g. data/session_...)")
    args = parser.parse_args()

    video_path = os.path.join(args.session, "video.mp4")
    if not os.path.exists(video_path):
        # Fallback to checking if frames exist or warn
        logging.error(f"Video file not found: {video_path}")
        return

    output_path = os.path.join(args.session, "labels.csv")
    
    # Initialize Pose Model
    pose_estimator = PoseEstimator(min_detection_confidence=0.5, model_complexity=2)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Processing {total_frames} frames from {video_path}...")
    
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "pose_visible", "keypoints_flat", "center_x", "center_y"])
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = pose_estimator.process_frame(frame)
            
            visible = False
            kps_str = "[]"
            cx, cy = -1.0, -1.0
            
            if results.pose_landmarks:
                visible = True
                kps = pose_estimator.get_keypoints_flat(results)
                kps_str = json.dumps(kps)
                
                center = pose_estimator.get_center_of_mass(results)
                if center:
                    cx, cy = center
            
            writer.writerow([frame_idx, visible, kps_str, cx, cy])
            
            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx}/{total_frames}", end='\r')
            
            frame_idx += 1
            
    cap.release()
    logging.info(f"\nDone. Labels saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
