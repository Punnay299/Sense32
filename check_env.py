
import sys
import subprocess
import importlib
import os

print("="*60, flush=True)
print("             Wi-Fi CSI Environment Checker", flush=True)
print("="*60, flush=True)

# 1. Python Check
print(f"[INFO] Python: {sys.version.split()[0]}", flush=True)
if sys.version_info < (3, 8):
    print("[FAIL] Python 3.8+ required.", flush=True)
else:
    print("[PASS] Python version ok.", flush=True)

# 2. Pip Packages Check
REQUIRED = ["numpy", "cv2", "torch", "mediapipe"]
INSTALLED = {}

print("-" * 20, flush=True)
print("[INFO] Checking pip packages...", flush=True)

def get_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

try:
    import importlib.metadata
except ImportError:
    print("[WARN] importlib.metadata not found (Python < 3.8?)", flush=True)

for pkg in REQUIRED:
    # Handle cv2 name mapping
    pip_name = "opencv-python" if pkg == "cv2" else pkg
    ver = get_version(pip_name)
    if ver:
        print(f"[PASS] {pip_name}: {ver}", flush=True)
        INSTALLED[pkg] = True
    else:
        print(f"[FAIL] {pip_name} NOT FOUND.", flush=True)
        INSTALLED[pkg] = False

# 3. Import Checks
print("-" * 20, flush=True)
print("[INFO] Verifying imports...", flush=True)

# Numpy
try:
    import numpy as np
    print(f"[PASS] numpy imported.", flush=True)
except ImportError as e:
    print(f"[FAIL] numpy import failed: {e}", flush=True)

# OpenCV
try:
    import cv2
    print(f"[PASS] cv2 imported.", flush=True)
except ImportError as e:
    print(f"[FAIL] cv2 import failed: {e}", flush=True)

# PyTorch
try:
    import torch
    print(f"[PASS] torch imported (CUDA: {torch.cuda.is_available()}).", flush=True)
except ImportError as e:
    print(f"[FAIL] torch import failed: {e}", flush=True)

# MediaPipe (The Problem Child)
print("-" * 20, flush=True)
print("[INFO] Diagnosing MediaPipe...", flush=True)
try:
    import mediapipe
    print(f"[PASS] 'import mediapipe' works. Path: {os.path.dirname(mediapipe.__file__)}", flush=True)
    
    # Check solutions
    try:
        import mediapipe.python.solutions as mp_solutions
        print("[PASS] 'import mediapipe.python.solutions' works.", flush=True)
    except ImportError:
        print("[FAIL] 'import mediapipe.python.solutions' FAILED.", flush=True)
        
        # Check standard
        try:
            import mediapipe.solutions
            print("[PASS] 'import mediapipe.solutions' works.", flush=True)
        except ImportError:
            print("[FAIL] 'import mediapipe.solutions' FAILED.", flush=True)
            
            # Check submodule
            try:
                from mediapipe.python.solutions import pose
                print("[PASS] 'from mediapipe.python.solutions import pose' works.", flush=True)
            except ImportError:
                print("[FAIL] ALL mediapipe imports failed.", flush=True)
                
except ImportError as e:
    print(f"[FAIL] 'import mediapipe' failed: {e}", flush=True)

print("="*60, flush=True)
print("Done.", flush=True)
