# Wi-Fi Human Detection & Pose Estimation - Project Report

## 1. Project Overview & Purpose
This project implements a **Local AI System** designed to "see" human bodies through Wi-Fi signals. By analyzing the subtle disruptions a human body causes to Radio Frequency (RF) waves—specifically Received Signal Strength Indicator (RSSI) and Round-Trip Time (RTT)—the system can detect human presence and estimate their 2D skeletal pose, even when they are behind walls or out of sight of a camera.

### Core Objectives
1.  **Non-Invasive Detection**: Use existing Wi-Fi hardware (standard Linux laptop) without specialized sensors (like LiDAR or RGB cameras) for the final detection.
2.  **Through-Wall Capability**: Leverage physical properties of RF signals (attenuation and multipath sensing) to detect subjects obscured by obstacles.
3.  **Privacy-First**: Run entirely offline (Edge AI). No video or data is sent to the cloud.
4.  **Robustness**: The system is designed to be resilient to missing hardware, using auto-detection for network interfaces and signal smoothing to handle environmental noise.

---

## 2. Technical Architecture: How It Works

The system operates in three distinct phases: **Collection**, **Training**, and **Inference**.

### Phase 1: Data Collection & Synchronization
To teach the AI how Wi-Fi signals relate to human movement, we simultaneously record two data streams:
*   **RF Data (The Input)**: The system polls standard Linux kernel interfaces (`/proc/net/wireless` and `ping`) at high frequency (~20-50Hz). It captures:
    *   **RSSI**: Signal strength drops when a body blocks the direct path (Line-of-Sight blocking).
    *   **RTT**: Latency changes due to signal scattering/reflection (Multipath effects).
    *   **Smoothing**: A sliding window mechanism (Moving Average) reduces random RF noise to highlight biological movement trends.
*   **Vision Data (The Teacher)**: A webcam records video. **MediaPipe Pose** (Google's ML solution) extracts 33 precise skeletal landmarks (shoulders, elbows, knees, etc.) from each frame. This acts as the "Ground Truth" or label.

### Phase 2: Deep Learning Model
A specialized Neural Network architecture (`WifiPoseModel`) learns the mapping: `f(RF Sequence) -> Pose Vector`.
*   **Input**: A sequence of 50 timestamped RSSI/RTT readings (representing ~1-2 seconds of history).
*   **1D CNN (Convolutional Neural Network)**: Extracts local temporal features (e.g., a sudden drop in signal).
*   **LSTM (Long Short-Term Memory)**: Understands the *sequence* of changes, allowing the model to distinguish between a person walking vs. random interference.
*   **Output Heads**:
    1.  **Pose Regressor**: Predicts the (x, y) coordinates of the human skeleton.
    2.  **Presence Detector**: Predicts the probability (0-1) of a human being present.

### Phase 3: Real-Time Inference
During deployment, the camera can be turned off (or covered).
1.  The system listens to the Wi-Fi adapter in real-time.
2.  It feeds the latest 50 RF samples into the trained model.
3.  The model outputs a skeletal structure, which is visualized on screen.
*   **Result**: You see a "Stick Figure" moving in sync with the actual person, driven entirely by Wi-Fi signal perturbations.

---

## 3. Project File Structure & Purpose

### Root Directory
*   `PROJECT.md`: This file. Comprehensive documentation of the project.
*   `check_env.py`: Diagnostic script. Checks OS compatibility (Linux/Fedora), required system tools (ping, iwconfig), and Python dependencies.
*   `requirements.txt`: List of Python packages (numpy, torch, mediapipe, etc.) pinned to verified working versions (Python 3.11).

### Source Code (`src/`)
The core logic library.
*   **`src/capture/`**: Hardware Interface Layer.
    *   `rf_interface.py`: **CRITICAL**. Handles Wi-Fi scanning. Implements `LinuxPollingCapture` which reads `/proc/net/wireless`, robustly auto-detects the active Wi-Fi interface (e.g., `wlp2s0`), and applies signal smoothing.
    *   `camera.py`: Wraps OpenCV video capture. Handles threading to ensure high-FPS recording without blocking the RF scanner.
    *   `beacon.py`: (Optional) Emits sync packets to align data timestamps across multiple devices.
*   **`src/vision/`**: Computer Vision Layer.
    *   `pose.py`: Wrapper around **MediaPipe**. Extracts human keypoints from video frames. Contains logic to handle image preprocessing and coordinate normalization.
*   **`src/model/`**: AI/ML Layer.
    *   `networks.py`: PyTorch Deep Learning model definition. Contains the `RFEncoder` (CNN+LSTM) and the detection heads.

### Scripts (`scripts/`)
Executable entry points for the user.
*   `scripts/collect_data.py`: **Recorder**. Runs Camera and RF Capture simultaneously. Saves `video.mp4` and `rf_data.csv` to a new session folder in `data/`.
*   `scripts/process_all_data.py`: **Label Generator**. Scans all session folders. If labels are missing, it runs the Vision system on the recorded video to generate `labels.csv` (the Ground Truth).
*   `scripts/train_local.py`: **Trainer**. Loads all labeled data, trains the `WifiPoseModel`, and saves the best weights to `models/best.pth`.
*   `scripts/run_inference.py`: **Demo**. Runs the live system. Loads the trained model and visualizes the "Wi-Fi Skeleton" in real-time.
*   `scripts/pose_extractor.py`: Helper used by `process_all_data.py` to process a single video.

### Data & Output
*   `data/`: Stores recorded sessions.
    *   Subfolders (e.g., `session_walk_01/`) contain raw CSV logs and Videos.
*   `models/`: Stores trained model weights (`best.pth`).
*   `tests/`: Unit tests ensuring system stability (`test_rf.py`, `test_vision.py`, etc.).

---

## 4. Usage Guide

1.  **Setup**: `pip install -r requirements.txt` (Python 3.11+).
2.  **Verify**: `python check_env.py`.
3.  **Collect**: `python scripts/collect_data.py --name walk_test --duration 60 --rf_mode linux`
4.  **Label**: `python scripts/process_all_data.py`
5.  **Train**: `python scripts/train_local.py --all_data --epochs 50`
6.  **Run**: `python scripts/run_inference.py --model models/best.pth --rf_mode linux`
