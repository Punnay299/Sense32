import argparse
import os
import sys
import cv2
import csv
import ast
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, required=True, help="Path to session folder")
    args = parser.parse_args()

    video_path = os.path.join(args.session, "video.mp4")
    labels_path = os.path.join(args.session, "labels.csv")

    if not os.path.exists(labels_path):
        print("labels.csv not found. Run pose_extractor.py first.")
        return

    # Load Labels
    labels = []
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['frame_index'] = int(row['frame_index'])
            row['visible'] = row['visible'] == 'True'
            row['center_x'] = float(row['center_x'])
            row['center_y'] = float(row['center_y'])
            # We don't parse full keypoints for editing, just center
            labels.append(row)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_idx = 0

    state = {
        'frame': None,
        'dirty': False
    }

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            h, w = state['frame'].shape[:2]
            nx, ny = x / w, y / h
            labels[current_idx]['center_x'] = nx
            labels[current_idx]['center_y'] = ny
            labels[current_idx]['visible'] = True
            state['dirty'] = True
            print(f"Frame {current_idx}: Set Center to ({nx:.2f}, {ny:.2f})")

    cv2.namedWindow("Label Assist")
    cv2.setMouseCallback("Label Assist", mouse_callback)

    while True:
        # seek if needed
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != current_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        state['frame'] = frame.copy()  # Keep clean copy for underlying data
        
        # Draw Overlay
        display_frame = frame.copy()
        lbl = labels[current_idx]
        h, w = display_frame.shape[:2]
        
        if lbl['visible']:
            cx, cy = int(lbl['center_x'] * w), int(lbl['center_y'] * h)
            cv2.circle(display_frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(display_frame, "VISIBLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "NOT VISIBLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.putText(display_frame, f"Frame: {current_idx}/{total_frames}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Label Assist", display_frame)
        
        key = cv2.waitKey(20) & 0xFF
        
        if key == 27: # ESC
            break
        elif key == ord('n') or key == 83: # 'n' or Right Arrow
            current_idx = min(current_idx + 1, total_frames - 1)
        elif key == ord('b') or key == 81: # 'b' or Left Arrow
            current_idx = max(current_idx - 1, 0)
        elif key == ord('v'):
            labels[current_idx]['visible'] = not labels[current_idx]['visible']
            state['dirty'] = True
        elif key == ord('s'):
            print("Saving...")
            with open(labels_path, 'w', newline='') as f_out:
                keys = labels[0].keys()
                writer = csv.DictWriter(f_out, fieldnames=keys)
                writer.writeheader()
                writer.writerows(labels)
            print("Saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
