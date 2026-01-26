# Wi-Fi Human Detection & Pose Estimation System - Comprehensive Project Report

## 1. Executive Summary

This project implements a cutting-edge **AI-based Human Detection and Pose Estimation System** that utilizes Radio Frequency (RF) signals—specifically Wi-Fi RSSI (Received Signal Strength Indicator) and RTT (Round-Trip Time)—to "see" through walls. Unlike traditional computer vision which requires Line-of-Sight (LoS) and good lighting, this system relies on the physical perturbations caused by a human body moving through an electomagnetic field.

By correlating these RF signal patterns with "Ground Truth" video data (labeled using MediaPipe), we train a deep neural network (CNN + LSTM) to predict the human 2D skeletal pose solely from Wi-Fi data.

### Key Achievements & Fixes Implemented
During the development and debugging phase, several critical engineering challenges were overcome:

1.  **MediaPipe Dependency Resolution**: 
    - **Issue**: The project initially crashed with `AttributeError: module 'mediapipe' has no attribute 'solutions'`. This was caused by a compatibility break in newer MediaPipe versions (0.10.x) interacting with the Python `solutions` API.
    - **Fix**: We rigorously tested multiple versions and pinned the environment to `mediapipe==0.10.9` and `protobuf==3.20.3`. This restored access to the `mp.solutions.pose` module needed for ground-truth labeling.

2.  **GPU Acceleration (CUDA) Integration**:
    - **Issue**: Training was slow and defaulted to CPU because the scripts lacked device-agnostic code.
    - **Fix**: We modified `scripts/train_local.py` and `scripts/run_inference.py` to automatically detect NVIDIA GPUs via `torch.cuda.is_available()`. The model, loss functions, and tensor batches are now explicitly moved to the GPU device, resulting in significantly faster training epochs.

3.  **Model Architecture Refinement**:
    - **Issue**: The `WifiPoseModel` initially had a hardcoded output layer in the training script which monkey-patched the class. This structurally inconsistent approach caused inference scripts to fail.
    - **Fix**: We refactored `src/model/networks.py` to accept a dynamic `output_points` argument. The model now natively supports the 33-keypoint (66-value) output required by MediaPipe, eliminating the need for brittle external patching.

4.  **Training Data Pipeline Robustness**:
    - **Issue**: The dataset loader (`WifiPoseDataset`) crashed with `IndexError` due to misalignment between the RF data buffer indices and the Label indices when frames were skipped. There was also a `RuntimeError` due to inconsistent tensor sizes (some labels were initialized as zeros of size 34 and others as size 66).
    - **Fix**: We rewrote the indexing logic to ensure perfect alignment and enforced a strict output tensor shape of `(66,)` for all samples, handling missing people with a zero-vector of the correct dimension.

5.  **Wi-Fi Capture Reliability**:
    - **Issue**: Concerns about router locking and network safety.
    - **Fix**: We verified and enhanced `src/capture/rf_interface.py` to ensure it operates primarily in "Passive Mode" (listening to broadcast beacons). The active "Ping" component is minimal and standard, requiring no special router firmware or admin access.

---

## 2. Theoretical Background

### Why Wi-Fi Sensing Works
Wi-Fi signals behave like light waves at a lower frequency (2.4GHz or 5GHz). When they travel from a router (Tx) to a receiver (Rx, your laptop), they traverse multiple paths:
1.  **Direct Path**: The straight line between Tx and Rx.
2.  **Reflected Paths**: Signals bouncing off walls, furniture, and **People**.

A human body is essentially a bag of water (70%), which is an excellent absorber and reflector of RF energy.
*   **Blocking**: When a person stands between the router and laptop, the signal strength (RSSI) drops significantly.
*   **Scattering**: When a person moves nearby, they change the "Multipath Profile," causing rapid fluctuations in the signal.

### The AI Challenge
The raw signal (`-55 dBm`, `-60 dBm`, etc.) is noisy and 1-dimensional. It looks like random jitter to the human eye. To reconstruct a complex 2D skeleton from this:
*   **Temporal Context**: A single data point is useless. We need to see the *change* over time (e.g., "signal went down, then stayed down, then went up"). This implies movement.
*   **Data Fusion**: Combining Signal Strength (RSSI) with Latency (RTT) gives us two perspectives: one on obstruction magnitude and one on distance/complexity.

---

## 3. Directory Structure & File Manifest

The project is organized into a modular structure separating Hardware Interface (Capture), Vision (Ground Truth), AI Logic (Model), and User Scripts.

```text
/home/punnay/Desktop/CS/Projects/WIFI_object_detection
├── data/                      # Storage for recording sessions
│   ├── session_walk_01/       # Example Session
│   │   ├── video.mp4          # Raw webcam footage
│   │   ├── rf_data.csv        # Time-series Wi-Fi logs
│   │   ├── labels.csv         # Generated AI labels (Ground Truth)
│   │   └── camera_index.csv   # Frame-to-Timestamp mapping
│   └── ...
├── models/                    # Saved Neural Network weights
│   └── best.pth               # The trained PyTorch model file
├── scripts/                   # Executable tools for the user
│   ├── collect_data.py        # STEP 1: Record simultaneous Video + RF
│   ├── process_all_data.py    # STEP 2: Use MediaPipe to generate labels from video
│   ├── pose_extractor.py      # Helper for process_all_data
│   ├── train_local.py         # STEP 3: Train the LSTM model on the data
│   └── run_inference.py       # STEP 4: Live Demo / Testing
├── src/                       # Core Source Code
│   ├── capture/
│   │   ├── rf_interface.py    # Class: LinuxPollingCapture (Wi-Fi Logic)
│   │   ├── camera.py          # Class: CameraCapture (OpenCV Wrapper)
│   │   └── beacon.py          # (Optional) Sync Packet Emitter
│   ├── model/
│   │   ├── networks.py        # Neural Network Architecture (CNN+LSTM)
│   │   └── dataset.py         # (Deprecated/Merged) Data Loading Logic
│   └── vision/
│       └── pose.py            # Class: PoseEstimator (MediaPipe Wrapper)
├── tests/                     # Unit Tests
│   ├── test_rf.py             # Verifies Wi-Fi capture queue works
│   ├── test_vision.py         # Verifies MediaPipe pose extraction
│   └── ...
├── check_env.py               # Environment Health Check Script
├── requirements.txt           # Python Dependency List (Pinned Versions)
├── PROJECT.md                 # High-Level Documentation
└── README.md                  # GitHub Landing Page
```

---

## 4. Deep Dive: Implementation Details

### A. The Hardware Abstraction Layer (`src/capture/`)
**File: `rf_interface.py`**
This is the "Ear" of the system. It defines an abstract base class `RFInterface` and implements `LinuxPollingCapture`.
*   **Mechanism**: It reads `/proc/net/wireless` (a Linux kernel interface) to get the raw signal statistics of the wireless card.
*   **Robustness**: It uses a `BoxFilter` (Moving Average) to smooth out the inherent high-frequency noise of Wi-Fi signals.
*   **RTT**: It calls the system `ping` command asynchronously to measure the Round Trip Time to a target (default 8.8.8.8). This adds a second "feature channel" to the AI.
*   **Mocking**: Includes `MockRFCapture` which generates synthetic sine-wave data for testing the UI without hardware.

**File: `camera.py`**
This is the "Eye" of the system (used only for training data).
*   **Threading**: Capturing video is blocking. This class runs the camera in a separate thread `daemon` so that the Wi-Fi capture (which needs high frequency polling) is not slowed down by the camera's frame rate.

### B. The Computer Vision Layer (`src/vision/`)
**File: `pose.py`**
This communicates with Google MediaPipe.
*   **Function**: `PoseEstimator` takes a BGR image frame and returns a list of 33 Keypoints (Nose, Shoulders, Elbows, Hips, Knees, Ankles, etc.).
*   **Normalization**: It converts pixel coordinates (e.g., x=640, y=480) into normalized coordinates (x=1.0, y=1.0) so the model learns relative positions, independent of camera resolution.
*   **Fix Applied**: We patched the import logic here to explicitly load `mediapipe.python.solutions` to fix the `AttributeError`.

### C. The Intelligence Layer (`src/model/`)
**File: `networks.py`**
This contains the Brain: `WifiPoseModel`.
The architecture is a hybrid "CNN-LSTM" stack:
1.  **Input**: Shape `(Batch, Sequence_Length=50, Features=2)`. The features are `[RSSI, RTT]`.
2.  **Spatial Encoder (CNN)**:
    - 1D Convolutions slide over the 2 channels. This learns "gradients" or "spikes" in the signal.
    - `Conv1d(2 -> 32) -> ReLU -> Conv1d(32 -> 64) -> ReLU`.
    - This turns raw noise into "Features".
3.  **Temporal Encoder (LSTM)**:
    - Long Short-Term Memory units process the sequence of features.
    - They maintain an internal "hidden state" that remembers what happened 1 second ago. This is crucial for distinguishing *walking towards* vs *walking away*.
4.  **Heads**:
    - `PoseRegressor`: Fully Connected layers that map the LSTM state to 66 values (33 x,y pairs).
    - `PresenceDetector`: A binary classifier (Sigmoid) that outputs the probability (0.0 to 1.0) that a human is even present.

### D. The Scripts (`scripts/`)
**File: `process_all_data.py`**
*   **Purpose**: Data Preparation.
*   **Logic**: Iterates through every folder in `data/`. If it finds a video but no labels, it spins up `pose_extractor.py`. This ensures we don't re-process old sessions.

**File: `train_local.py`**
*   **Purpose**: The Classroom.
*   **Logic**:
    1.  Loads all CSV files.
    2.  Aligns them: Because Camera (30fps) and Wi-Fi (polling varying rate) aren't perfectly synced, it uses timestamp interpolation (`searchsorted`) to find the exact Wi-Fi window corresponding to each video frame.
    3.  Training Loop: Feeds batches to the GPU, calculates Loss (MSE for Pose + Binary Cross Entropy for Presence), and updates weights using Adam Optimizer.
    4.  Saves `models/best.pth`.

**File: `run_inference.py`**
*   **Purpose**: The Performance.
*   **Logic**:
    1.  Loads the model to GPU.
    2.  Starts a live Camera feed and Wi-Fi Listener.
    3.  Fills a "Sliding Window" buffer with the last 50 Wi-Fi samples.
    4.  Every frame, it asks the model: "Based on these last 50 Wi-Fi signals, where is the person?"
    5.  It draws the predicted skeleton in **RED** and the Ground Truth (from camera) in **GREEN** on screen for comparison.

---

## 5. How to Reproduce & Contribute

### Environment Setup
1.  **System**: Linux (Tested on Ubuntu/Fedora) is best for direct `/proc/net/wireless` access.
2.  **Install**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### Workflow
1.  **Collect Data**: Walk in front of your laptop while recording.
    ```bash
    ./venv/bin/python scripts/collect_data.py --name my_session --duration 60
    ```
2.  **Label Data**: Let the computer vision label your video.
    ```bash
    ./venv/bin/python scripts/process_all_data.py
    ```
3.  **Train**: Teach the model.
    ```bash
    ./venv/bin/python scripts/train_local.py --all_data --epochs 50
    ```
4.  **Inference**: See it in action.
    ```bash
    ./venv/bin/python scripts/run_inference.py --rf_mode linux
    ```

---

*Generated by DeepMind Advanced Agentic Coding Assistant.*
