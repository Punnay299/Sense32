import os
import subprocess
import glob
import logging
import argparse
import sys
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_relabel", action="store_true", help="Force re-running pose extraction even if labels.csv exists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    
    # 1. Find all sessions in data/
    if not os.path.exists("data"):
        logging.error("No 'data' directory found.")
        return

    sessions = glob.glob(os.path.join("data", "session_*"))
    logging.info(f"Found {len(sessions)} sessions.")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 2. Process Videos (Extract Labels)
    pbar = tqdm(sessions, desc="Processing Sessions")
    for session in pbar:
        labels_file = os.path.join(session, "labels.csv")
        video_file = os.path.join(session, "video.mp4")
        
        pbar.set_postfix({"Current": os.path.basename(session)[:15]})
        
        if not os.path.exists(video_file):
            logging.warning(f"Skipping {session}: No video.mp4")
            skip_count += 1
            continue
            
        if not os.path.exists(labels_file) or args.force_relabel:
            # logging.info(f"Processing {session}...") # Reduce log spam
            # Call pose_extractor.py
            cmd = [sys.executable, "scripts/pose_extractor.py", "--session", session]
            try:
                # Run subprocess and capture output to avoid cluttering tqdm unless error
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                success_count += 1
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to process {session}: {e.stderr}")
                fail_count += 1
                continue
        else:
            # logging.info(f"Skipping {session}: Labels already exist.")
            skip_count += 1
            
    pbar.close()
            
    logging.info("------------------------------------------------Data Processing Complete------------------------------------------------")
    logging.info(f"Success: {success_count} | Failed: {fail_count} | Skipped (Existing): {skip_count}")
    logging.info("Ready for training.")

if __name__ == "__main__":
    main()
