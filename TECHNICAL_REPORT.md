# Wi-Fi Human Detection & Pose Estimation: Comprehensive Technical Report

**Project**: Wi-Fi Human Detection  
**Version**: 3.0 (Stable / Robust Edition)  
**Date**: January 2026  
**Environment**: Python 3.11.8 (via pyenv)  
**OS**: Linux (tested on Ubuntu/Fedora)

---

## Table of Contents

1.  [Executive Summary](#1-executive-summary)
2.  [System Requirements & Environment](#2-system-requirements--environment)
3.  [Theoretical Foundation](#3-theoretical-foundation)
    *   3.1 RF Signal Propagation & Human Interaction
    *   3.2 The Physics of "Seeing" Without Light
    *   3.3 RSSI vs. CSI
4.  [System Architecture Deep Dive](#4-system-architecture-deep-dive)
    *   4.1 High-Level Data Flow
    *   4.2 Directory Structure Analysis
5.  [Module Analysis: RF Signal Acquisition (`src/capture`)](#5-module-analysis-rf-signal-acquisition-srccapture)
    *   5.1 The `RFInterface` Abstract Base Class
    *   5.2 `ScapyRFCapture`: The Core Engine
    *   5.3 **Automated Monitor Mode Switching** (New Feature)
    *   5.4 **Channel Hopping Mechanism** (New Feature)
6.  [Module Analysis: Visual Ground Truth (`src/vision`)](#6-module-analysis-visual-ground-truth-srcvision)
    *   6.1 MediaPipe Pose Integration
    *   6.2 Camera Robustness & Fallbacks
7.  [Module Analysis: The Neural Network (`src/model`)](#7-module-analysis-the-neural-network-srcmodel)
    *   7.1 Hybrid CNN-LSTM Architecture
    *   7.2 Loss Functions & Optimization
8.  [Data Pipeline & Orchestration Scripts](#8-data-pipeline--orchestration-scripts)
    *   8.1 `collect_data.py`: The Recorder
    *   8.2 `process_all_data.py`: The Labeler
    *   8.3 `train_local.py`: The Teacher
9.  [Operational Workflow (The "Golden Path")](#9-operational-workflow-the-golden-path)
10. [Troubleshooting & Known Issues](#10-troubleshooting--known-issues)

---

## 1. Executive Summary

This project implements a cutting-edge **Non-Line-of-Sight (NLOS) Human Sensing System** that leverages standard commercial Wi-Fi hardware to detect human presence and estimate 2D skeletal pose. Unlike traditional Computer Vision which fails behind obstacles, or specialized Radar systems which require expensive hardware, this system transforms a standard Linux laptop into a passive biological sensor.

The system utilizes a custom **Hybrid CNN-LSTM Deep Learning Architecture** to decode the subtle perturbations in Wi-Fi signal strength (RSSI) and round-trip time (RTT) caused by human movement. By training on a synchronized dataset of RF signals and video-derived ground truth (using MediaPipe), the model learns the complex mapping between electromagnetic wave distortions and human kinematics.

**Key Capabilities:**
*   **Passive Sensing**: No wearable devices required on the subject.
*   **Through-Wall Detection**: Capable of detecting movement behind obstructions (drywall, wood, etc.).
*   **Privacy-Desirable**: Does not capture identifiable optical images of the subject during inference.
*   **Hardware Agnostic**: Runs on standard Linux networking stacks using `scapy` for raw packet capture.

---

## 2. System Requirements & Environment

To ensure reproducibility, strictly adhere to the following environment configuration.

### 2.1 Software Stack
*   **Operating System**: Linux (Kernel 5.x+ recommended for modern Wi-Fi drivers).
*   **Python Version**: **3.11.8** (Managed via `pyenv`).
    *   *Note*: Older versions may lack necessary asyncio features; newer versions (3.12+) may have compatibility issues with `scapy` or `torch` at the time of writing.
*   **Core Dependencies**:
    *   `scapy` (2.5.0+): For raw 802.11 frame capture and injection.
    *   `torch` (2.1.2+): Deep Learning framework (CUDA support recommended).
    *   `mediapipe` (0.10.9): For extracting ground truth skeletal poses from video.
    *   `opencv-python`: For video acquisition and processing.
    *   `pandas`, `numpy`: For data manipulation.

### 2.2 Hardware Requirements
*   **Wi-Fi Interface**: Must support **Monitor Mode**.
    *   Recommended Chipsets: Atheros (e.g., AR9271), Intel (e.g., AX200/210), Ralink.
    *   Interface Name: Typically `wlan0`, `wlpXsY` (e.g., `wlp8s0`).
*   **Camera**: Standard USB Webcam (for training data collection only).
    *   *Note*: The system includes a robust fallback to a "Mock Camera" if no physical camera is detected, allowing RF development to continue even without visual ground truth.

---

## 3. Theoretical Foundation

### 3.1 RF Signal Propagation & Human Interaction
Wi-Fi signals (2.4 GHz and 5 GHz) are electromagnetic waves. When they propagate through space, they interact with objects in their path via:
1.  **Reflection**: Bouncing off walls, floors, and metal.
2.  **Diffraction**: Bending around corners.
3.  **Scattering**: Breaking into multiple rays when hitting complex surfaces (like the human body).
4.  **Absorption**: Energy loss as the wave passes through materials (water in the human body is a strong absorber).

### 3.2 The Physics of "Seeing" Without Light
In a static room, the multipath profile (the sum of all signal paths reaching the receiver) is constant. When a human moves, they act as a **Dynamic Scatterer**.
*   **Shadowing**: Blocking the Line-of-Sight (LoS) causes a drop in Signal Strength (RSSI).
*   **Multipath Variation**: Moving limbs change the path lengths of reflected signals, creating constructive and destructive interference patterns at the receiver.
*   **Doppler Shift**: The velocity of movement induces minute frequency shifts (Micro-Doppler), which correlates with the speed of walking or limb movement.

### 3.3 RSSI vs. CSI
*   **RSSI (Received Signal Strength Indicator)**: A coarse-grained metric representing the total power received. Available on ALL Wi-Fi cards. The current implementation relies primarily on RSSI extracted from Beacon Frames.
*   **CSI (Channel State Information)**: Fine-grained amplitude and phase information for each sub-carrier (OFDM). While richer, it requires modified firmware/drivers (e.g., Nexmon). This project is designed to be **Universal**, utilizing RSSI to ensure it runs on un-modified hardware, while employing Deep Learning to compensate for RSSI's lower resolution.

---

## 4. System Architecture Deep Dive

### 4.1 Directory Structure
The project follows a modular "Clean Architecture" pattern.

```text
WIFI_object_detection/
├── data/                   # Stores recorded sessions (RF + Video)
├── models/                 # Stores trained PyTorch models (.pth)
├── scripts/                # Executable entry points
│   ├── collect_data.py     # Main recorder
│   ├── process_all_data.py # Data pre-processor
│   ├── train_local.py      # Training script
│   └── run_inference.py    # Live demo
├── src/                    # Core Library Code
│   ├── capture/            # RF and Camera interfaces
│   │   ├── rf_interface.py # The heavy lifter for Wi-Fi
│   │   └── camera.py       # Camera wrapper
│   ├── vision/             # Computer Vision logic
│   │   └── pose.py         # MediaPipe wrapper
│   └── model/              # Neural Networks
│       └── networks.py     # PyTorch definitions
└── venv/                   # Python Virtual Environment
```

---

## 5. Module Analysis: RF Signal Acquisition (`src/capture`)

This module is the "Ear" of the system. It is responsible for listening to invisible Wi-Fi signals.

### 5.1 The `RFInterface` Abstract Base Class
Located in `src/capture/rf_interface.py`, this class defines the contract for any capture backend (Mock, Linux-Native, Scapy). It manages a thread-safe `queue.Queue` to stream packet dictionaries to the main application loop.

### 5.2 `ScapyRFCapture`: The Core Engine
This is the production-grade implementation.
*   **Library**: Uses `scapy.all.sniff`.
*   **Target**: 802.11 Beacon Frames.
*   **Extraction**:
    *   Decodes the `RadioTap` header to extract `dBm_AntSignal` (RSSI).
    *   Decodes `Dot11Beacon` to extract SSID.
    *   Extracts Source MAC address.

### 5.3 Automated Monitor Mode Switching (CRITICAL)
A major challenge in Wi-Fi sensing is that cards default to "Managed Mode" (Client). In this mode, they discard packets not addressed to them. To capture raw signals, the card must be in "Monitor Mode".

The code now implements **Auto-Recovery Logic**:
1.  It attempts to start sniffing.
2.  If it detects 0 packets or an error, it checks for `iw` availability.
3.  It executes a rigorous sequence to force the mode switch:
    ```python
    subprocess.run(["ip", "link", "set", iface, "down"])
    subprocess.run(["iw", "dev", iface, "set", "type", "monitor"])
    subprocess.run(["ip", "link", "set", iface, "up"])
    ```
4.  This ensures the system works "out of the box" without manual `airmon-ng` configuration.

### 5.4 Channel Hopping Mechanism
Wi-Fi routers broadcast on specific channels (1-13 in 2.4GHz). If the sniffer stays on Channel 1 but the router is on Channel 6, it will hear nothing (or very weak crosstalk).

**Implementation**:
*   A dedicated background thread `_channel_hopper` runs inside `ScapyRFCapture`.
*   It cycles through the list `[1, 6, 11, 2, 7, ...]` every **0.5 seconds**.
*   It uses `iw dev <iface> set channel <ch>` to retune the radio.
*   **Benefit**: This guarantees that within a few seconds, the sensor will align with the target router's frequency and capture the strongest signal samples.

---

## 6. Module Analysis: Visual Ground Truth (`src/vision`)

This module acts as the "Teacher". Since Wi-Fi signals are abstract, we need a reference to know where the human actually is.

### 6.1 MediaPipe Pose Integration
*   File: `src/vision/pose.py`
*   Uses Google's MediaPipe BlazePose model.
*   Extracts **33 3D Landmarks** (x, y, z, visibility).
*   These 33 points serve as the "Labels" ($Y$) for our supervised learning problem.

### 6.2 Camera Robustness
The `collect_data.py` script includes advanced error handling for the camera:
*   **Auto-Scan**: Iterates indices 0 through 4 to find a working `/dev/videoX`.
*   **Fail-Over**: If NO physical camera is found, it instantiates a `MockCameraCapture`.
    *   *Why?* This allows developers to test/debug the critical RF capture pipeline even if they don't have a webcam attached. The video will be black, but the RF data (csv) will be valid.

---

## 7. Module Analysis: The Neural Network (`src/model`)

### 7.1 Hybrid CNN-LSTM Architecture (`WifiPoseModel`)
The model is designed to handle Time-Series data.

1.  **Input Layer**:
    *   Shape: `(Batch, Sequence_Length, Features)`
    *   Features: RSSI, RTT (normalized).
2.  **Spatial Encoder (1D CNN)**:
    *   A stack of 3 `Conv1d` layers (64 $\to$ 128 $\to$ 256 filters).
    *   Acts as a learnable feature extractor, identifying sudden changes (peaks/troughs) in signal strength that correspond to steps or arm swings.
    *   Effectively performs a learnable Short-Time Fourier Transform (STFT).
3.  **Temporal Integrator (LSTM)**:
    *   A 3-layer LSTM with 256 hidden units.
    *   Crucial for capturing **State**. A static RSSI of -60dBm could mean "Empty Room" or "Person Standing Still". The LSTM looks at the *history* to differentiate (e.g., "Signal was varying 2s ago, so person is present").
4.  **Decoders (Heads)**:
    *   **Pose Head**: Fully Connected layers mapping LSTM state to 66 outputs (33 joints * 2 coordinates).
    *   **Presence Head**: Binary classifier (Sigmoid) determining if a human is in the sensing range.

### 7.2 Loss Functions
*   **Pose**: Mean Squared Error (MSE). $L_{pose} = ||\hat{Y} - Y||^2$
*   **Presence**: Binary Cross Entropy (BCE).
*   **Total Loss**: $L = L_{pose} \cdot Mask_{presence} + L_{presence}$
    *   *Note*: We masks pose loss when no person is present to prevent the model from successfully predicting "0,0,0" for everyone.

---

## 8. Data Pipeline & Orchestration Scripts

### 8.1 `scripts/collect_data.py`
The entry point for dataset creation.
*   **Roles**:
    1.  Starts RF Sniffer (Thread).
    2.  Starts Camera (Thread).
    3.  Synchronizes timestamps using `time.monotonic()`.
    4.  Writes `rf_data.csv`, `camera_index.csv`, and `video.mp4`.
*   **Critical Detail**: Runs a "Safety Check" at T+5 seconds. If packet count is 0, it aborts loudly. This prevents "silent failure" recordings.

### 8.2 `scripts/process_all_data.py`
The "Glue" of the system.
*   Iterates all sessions in `data/`.
*   Reads `video.mp4`.
*   Runs MediaPipe on every frame.
*   Writes `labels.csv` containing the flattened 33 keypoints.
*   **Performance**: Uses TensorFlow Lite XNNPACK delegate for CPU acceleration on Linux.

### 8.3 `scripts/train_local.py`
The learning loop.
*   **Dataset Alignment**: This is the most complex logic.
    *   Video is ~30 FPS (33ms gap).
    *   RF is Variable Rate (5-20ms gap).
    *   The `WiFiPoseDataset` class aligns them by finding the RF window $[T_{video} - 2s, T_{video}]$ for each labeled frame.
*   **Training**: Runs standard PyTorch training loop with `Adam` optimizer and `ReduceLROnPlateau` scheduler.

---

## 9. Operational Workflow (The "Golden Path")

Follow this EXACT sequence to reproduce the system functionality.

### Prerequisite: Sudo and Python
Ensure you are in the project root with the virtual environment active.
```bash
# Check python version (Must be 3.11.8)
python --version
# Activate venv
source venv/bin/activate
```

### Step 1: Collect Data
**CRITICAL**: You must use `sudo` because accessing the Wi-Fi card in promiscuous/monitor mode requires root privileges.
```bash
# --rf_mode scapy: Uses the production Scapy backend
# --duration 60: Records for 60 seconds
sudo ./venv/bin/python scripts/collect_data.py --name session_01 --rf_mode scapy --duration 60
```
*   *Observation*: Ensure the log says "Monitor mode enabled successfully" and "Verified: Received X packets".

### Step 2: Fix File Permissions
Because Step 1 ran as `root`, the created files in `data/` are owned by root. You cannot process them as a normal user unless you fix this.
```bash
# Replace $USER with your username (e.g., 'punnay')
sudo chown -R $USER:$USER data/
```

### Step 3: Process Data (Generate Labels)
This runs the Computer Vision extraction.
```bash
./venv/bin/python scripts/process_all_data.py
```
*   *Output*: You will see `labels.csv` created in each session folder.

### Step 4: Train the Model
```bash
# --all_data: Uses all sessions found in data/
# --epochs 100: Sufficient for initial convergence
./venv/bin/python scripts/train_local.py --all_data --epochs 100
```
*   *Result*: A trained model saved to `models/best.pth`.

### Step 5: Live Inference
See the "Ghost" in the machine.
```bash
sudo ./venv/bin/python scripts/run_inference.py --rf_mode scapy
```

---

## 10. Troubleshooting & Known Issues

### 10.1 `[Errno 100] Network is down`
**Cause**: The interface was administratively down or Scapy attempted to use it while it was in a transition state.
**Fix**: The code now includes an auto-healer that runs `ip link set <iface> up`. If persistent, manually run:
```bash
sudo ip link set wlp8s0 up
```

### 10.2 Empty `rf_data.csv` (0 Packets)
**Cause**: Channel Mismatch or Managed Mode.
**Fix**: The new Channel Hopper fixes the mismatch. The Monitor Mode switch fixes the filtering.
**Validation**: Run `collect_data.py` and watch for the "Verified" log message at the 5-second mark.

### 10.3 Camera Open Errors (`/dev/video0`)
**Cause**: Another process using the camera, or no camera attached.
**Fix**: The script now falls back to Mock Camera. To use a real camera, ensure no other apps (Zoom, Cheese) are open.

---

**End of Technical Report**
