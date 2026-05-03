#!/usr/bin/env python3
import os
import time
import glob
import logging
from datetime import datetime

# Configuration
# Dict of: folder_path -> {"extension": str, "keep": int}
WATCH_DIRS = {
    "pixel-space-stable-rms-baseline/ckpts": {"extension": ".safetensors", "keep": 4},
    "pixel-space-stable-rms-baseline/previews":          {"extension": ".png",          "keep": 12},
}
SLEEP_TIME = 300  # Check every 5 minutes (in seconds)
LOG_FILE = "checkpoint_cleanup_tagger.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def ensure_folders_exist():
    """Ensure all watched folders exist."""
    for folder in WATCH_DIRS:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.info(f"Created folder: {folder}")

def get_files_by_time(folder, extension):
    """Get all files with given extension in folder, sorted by modification time (oldest first)."""
    files = glob.glob(os.path.join(folder, f"*{extension}"))
    return sorted(files, key=os.path.getmtime)

def cleanup_old_checkpoints():
    """For each watched directory, keep only the most recent N files and delete the rest."""
    for folder, cfg in WATCH_DIRS.items():
        extension = cfg["extension"]
        keep_recent = cfg["keep"]

        logging.info(f"[{folder}] Checking for *{extension} files (keep={keep_recent})")
        checkpoint_files = get_files_by_time(folder, extension)

        if not checkpoint_files:
            logging.info(f"[{folder}] No files found")
            continue

        total_files = len(checkpoint_files)
        logging.info(f"[{folder}] Found {total_files} file(s)")

        if total_files <= keep_recent:
            logging.info(f"[{folder}] Total files ({total_files}) <= keep ({keep_recent}), no cleanup needed")
            continue

        # Files to delete are all except the last keep_recent
        files_to_delete = checkpoint_files[:-keep_recent]
        files_to_keep = checkpoint_files[-keep_recent:]

        logging.info(f"[{folder}] Keeping {len(files_to_keep)} most recent file(s):")
        for file_path in files_to_keep:
            logging.info(f"  - {os.path.basename(file_path)}")

        logging.info(f"[{folder}] Deleting {len(files_to_delete)} old file(s):")
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                logging.info(f"  ✓ Deleted {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"  ✗ Failed to delete {os.path.basename(file_path)}: {str(e)}")

def main():
    """Main cleanup function."""
    logging.info("Starting checkpoint cleanup watchdog")
    logging.info(f"Watching {len(WATCH_DIRS)} director(ies)")
    ensure_folders_exist()
    
    while True:
        try:
            cleanup_old_checkpoints()
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
        
        # Sleep before next check
        logging.info(f"Sleeping for {SLEEP_TIME} seconds")
        time.sleep(SLEEP_TIME)

if __name__ == "__main__":
    main()