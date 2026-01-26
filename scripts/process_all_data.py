
import os
import subprocess
import glob
import logging
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_relabel", action="store_true", help="Force re-running pose extraction even if labels.csv exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # 1. Find all sessions in data/
    sessions = glob.glob(os.path.join("data", "session_*"))
    logging.info(f"Found {len(sessions)} sessions.")
    
    # 2. Process Videos (Extract Labels)
    for session in sessions:
        labels_file = os.path.join(session, "labels.csv")
        video_file = os.path.join(session, "video.mp4")
        
        if not os.path.exists(video_file):
            logging.warning(f"Skipping {session}: No video.mp4")
            continue
            
        if not os.path.exists(labels_file) or args.force_relabel:
            logging.info(f"Processing {session}...")
            # Call pose_extractor.py
            cmd = ["./venv/bin/python", "scripts/pose_extractor.py", "--session", session]
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to process {session}: {e}")
                continue
        else:
            logging.info(f"Skipping {session}: Labels already exist.")
            
    logging.info("All data processed. Ready for training.")

if __name__ == "__main__":
    main()
