# Wi-Fi Human Sensing: Through-Wall Detection with ESP32 & AI

![Wi-Fi Sensing](https://img.shields.io/badge/Wi--Fi-Sensing-blue)
![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-green)

A deep learning project attempting to track human presence, location, and pose using only Wi-Fi signals from two ESP32 nodes. The system fuses Computer Vision for line-of-sight detection with Channel State Information (CSI) analysis for non-line-of-sight sensing.

**Current Scope (Feb 2026):** The system is optimized for **Room A** (Wi-Fi Sensing) and **Hallway** (Visual Sensing).

---

## 🚀 Concept
1.  **Traffic Generator**: A script floods the network with 100Hz UDP packets.
2.  **CSI Capture**: Two ESP32s (Node A & Node B) intercept these packets and measure the "Channel State Information" (CSI)—how Wi-Fi signals bounce around the room.
3.  **AI Processing**: A **1D CNN + LSTM** neural network analyzes these signal distortions to infer human presence and location.

---

## 🛠️ Hardware Setup

*   **Receiver A (Node A)**: ESP32 in **Room A** (Target IP: `10.42.0.149`).
*   **Receiver B (Node B)**: ESP32 secondary node (Target IP: `10.42.0.173`).
*   **Transmitter**: Laptop (Linux) running a Wi-Fi Hotspot (`10.42.0.1`).
*   **Camera**: Webcam for "Hallway" detection and ground-truth labeling.

---

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/wifi-csi-sensing.git
    cd wifi-csi-sensing
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Flash Firmware**
    *   Open `firmware/esp32_csi_rx/esp32_csi_rx.ino`.
    *   Set `SSID`, `PASSWORD`, and `TARGET_IP` (Laptop IP).
    *   Flash to both ESP32s.

---

## 🏃 Usage

### 1. Collect Data
Use the master collector to record labeled sessions.
```bash
python3 scripts/master_collector.py
```
*   Follow prompts to record **Room A**, **Hallway**, and **Empty** scenarios.
*   **Crucial**: You must collect "Empty" room data to teach the model the baseline noise.

### 2. Process Labels
Extracts pose labels from video and cleans the CSI data.
```bash
python3 scripts/process_all_data.py --force_relabel
```

### 3. Train the Model
Trains the CNN-LSTM model on all collected data.
```bash
python3 scripts/train_local.py --all_data --epochs 30
```
*   **Note**: This uses **Global Scaling** to preserve relative signal strength differences between nodes.

### 4. Run Live Inference
Starts the real-time detection engine.
```bash
python3 scripts/run_inference.py
```

---

## 🔍 Technical Details

### Hybrid Sensing Logic
The system employs a rigorous decision hierarchy:
1.  **Visual Confirmation**: If the camera detects a person, the location is confirmed as **HALLWAY**.
2.  **RF Inference**: If the subject leaves the visual field, the system utilizes the Deep Learning model to monitor **ROOM A**.

See `CSI_VISION.md` for a detailed breakdown of the Signal Processing Pipeline and Neural Network Architecture.
