# Wi-Fi & Vision Sensor Fusion for Human Detection

> **Note**: This project implements a Multi-Modal AI system. It does **not** rely solely on basic RSSI triangulation but uses high-dimensional RF feature extraction fused with Computer Vision.

## 📌 Project Overview
This project explores the intersection of **Wireless Sensing** and **Computer Vision**. It uses a standard Wi-Fi Network Interface Card (NIC) to capture raw Radio Frequency (RF) signals (RSSI, RTT, CSI) and trains a Deep Neural Network to estimate human pose and presence. 

Crucially, this system uses **MediaPipe Pose** (Vision) as the "Teacher" to train the RF "Student". By synchronizing video and Wi-Fi data, we create a labelled dataset where RF signal patterns are mapped to specific human movements.

### Key Features
*   **Sensor Fusion (Kalman Filter)**: Combines the precision of Vision with the occlusion-resistance of RF.
*   **Deep Learning on RF**: Uses 1D Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (GRUs/LSTMs) to decode complex signal distortions.
*   **Line-of-Sight Training**: Requires a camera to generate initial Ground Truth labels.
*   **Passive & Active Modes**: Supports both Passive Sniffing (Scapy) and Active Polling (Linux/Ping) for RF data collection.

---

## 🏗 System Architecture

The system operates in two main phases: **Training** and **Inference**.

### 1. Training Phase (Teacher-Student)
In this phase, we capture data to teach the AI model what "Human Presence" looks like in the RF spectrum.

*   **Teacher (Vision)**: A Camera records the subject. `MediaPipe` extracts a 33-point skeletal wireframe. This is the **Ground Truth**.
*   **Student (RF)**: The Wi-Fi card captures packet signal strength (RSSI) and timing (RTT) at high frequency.
*   **Learning**: The Model (`WifiPoseModel`) is trained to predict the Skeleton coordinates solely from the RF data.

### 2. Inference Phase (Fusion)
In the live demo, we run both sensors.
*   If the Camera sees the person, the system uses Vision (High Confidence).
*   If the Person walks behind an object (visual occlusion), the system falls back to the trained RF Model to estimate position.
*   A **Kalman Filter** smooths the transition between these two states to prevent jitter.

---

## 🚀 Getting Started

### Prerequisites
*   **Hardware**: 
    *   Linux Laptop/Pi with Wi-Fi Card (Monitor mode preferred but not required).
    *   USB Webcam.
*   **Software**:
    *   Python 3.8+
    *   PyTorch (CPU or CUDA)
    *   OpenCV, MediaPipe, Scapy, Pandas

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/wifi-object-detection.git
cd wifi-object-detection

# Install dependencies
pip install -r requirements.txt

# (Optional) Enable Monitor Mode for better RF data
# sudo iw dev wlan0 set type monitor
```

---

## 📖 Usage Guide

### 1. Check Hardware
Verify your camera and Wi-Fi interface are detected.
```bash
python scripts/check_hardware.py
```

### 2. Collect Data
Record a session. Walk in front of the camera while the RF captures data.
```bash
# --rf_mode options: mock (test), linux (active ping), scapy (passive sniff)
python scripts/collect_data.py --name session_01 --duration 60 --rf_mode linux
```

### 3. Generate Labels
Process the video to generate the "Answer Key" for the AI.
```bash
# Extracts skeletal pose from video.mp4 -> labels.csv
python scripts/process_all_data.py
```

### 4. Train Model
Train the Neural Network to predict Pose from RF.
```bash
python scripts/train_local.py --all_data --epochs 50
```

### 5. Run Live Inference
See the Fusion Engine in action.
```bash
python scripts/run_inference.py --rf_mode linux
```

---

## 📂 File Structure
*   `src/capture`: Drivers for Camera and Wi-Fi cards.
*   `src/vision`: MediaPipe wrappers for ground truth generation.
*   `src/model`: PyTorch definitions (CNN/LSTM) for the RF model.
*   `src/engine`: Kalman Filter logic for sensor fusion.

## ⚠️ Limitations
*   **Environment Specific**: RF multipath effects are heavily dependent on the room layout. A model trained in Room A may specific retraining for Room B.
*   **Line-of-Sight Training**: You must be visible to the camera during the *training* phase to generate labels.
*   **Single Subject**: The current logic is optimized for tracking one primary subject.

## 📄 Technical Report
For a deep dive into the code, algorithms, and function-level documentation, please refer to [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
