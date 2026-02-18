# Wi-Fi Human Detection & Pose Estimation (v2.0)

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Pytorch](https://img.shields.io/badge/pytorch-2.1-red)
![Platform](https://img.shields.io/badge/platform-ESP32%20%7C%20Linux-lightgrey)

A robust, non-invasive human sensing system that uses **Channel State Information (CSI)** from two commodity ESP32 microcontrollers to detect human presence, classify location (Room A vs Room B), and estimate skeletal pose—even through walls.

---

## 🚀 Key Features

*   **Dual-Node Sensing**: Synchronized capture from two ESP32 receivers (Room A & Room B) to eliminate blind spots.
*   **Hybrid Inference Engine**:
    *   **Visual Priority**: Uses a Webcam (MediaPipe) for perfect detection when the user is in the Hallway.
    *   **RF Fallback**: Uses a **CNN-LSTM** Model to detect user location (Room A/B) and pose when out of camera view.
*   **Robust Signal Processing**:
    *   **Hampel Filter** for amplitude outlier removal.
    *   **Phase Sanitization** (Unwrapping + Linear Detrending) to remove clock jitter.
    *   **Log-Difference Features** to solve the "Near-Far" signal bias problem.
*   **Rich UI**: Real-time terminal dashboard showing network health, probability distributions, and system status.

---

## 🛠️ Hardware Setup

### Topology
*   **Transmitter (TX)**: Linux Laptop (`10.42.0.1`) sending 100Hz UDP packets.
*   **Receiver A (RX1)**: ESP32 in **Room A** (Center) - IP `10.42.0.149`.
*   **Receiver B (RX2)**: ESP32 in **Room B** (Corner) - IP `10.42.0.173`.
*   **Camera**: Webcam connected to Laptop (Hallway view).

### Circuit/Firmware
*   ESP32s must be flashed with `firmware/esp32_csi_rx/esp32_csi_rx.ino`.
*   **Important**: Hardcode the `TARGET_IP` in firmware to your laptop's IP.

---

## 📦 Installation

1.  **Clone & Environment**:
    ```bash
    git clone https://github.com/yourusername/wifi-csi-sensing.git
    cd wifi-csi-sensing
    pip install -r requirements.txt
    ```

2.  **Verify Hardware**:
    Ensure your ESP32s are powered on and pingable:
    ```bash
    ping 10.42.0.149
    ping 10.42.0.173
    ```

---

## 🏃 Usage

### 1. Real-Time Inference (Main Demo)
This script runs the full pipeline: Camera Capture + Dual ESP32 Sniffing + Neural Network Inference.
```bash
python3 scripts/run_inference.py
```
*   **Flags**:
    *   `--debug_stats`: Show detailed signal variance/energy in console.
    *   `--swap_nodes`: If you physically swapped Room A/B devices, use this to invert logic software-side.

### 2. Data Collection (For Retraining)
To collect new data for your specific room layout:
```bash
python3 scripts/master_collector.py
```
*   Follow the prompts to record labeled sessions for **Room A**, **Room B**, **Hallway**, and **Empty**.
*   **Note**: Collect at least 2 minutes of "Empty" room data to calibrate the noise floor.

### 3. Model Training
```bash
python3 scripts/train_local.py --all_data --epochs 50
```
*   Trained models are saved to `models/best.pth`.
*   Training automatically updates the `models/scaler.json` calibration file.

---

## 🧠 Technical Details

For a deep dive into the Math, Signal Processing, and Neural Network Architecture, please read the **[Technical Documentation](CSI_VISION.md)**.

### Performance Notes
*   **Latency**: System runs at ~15-20 FPS on an RTX 3060/4060/5060.
*   **Synchronization**: The system waits for *both* ESP32 buffers to fill (50 frames ~ 0.5s) before inference. Network drops will pause inference (shown as "Buffering...").
*   **Calibration**: If Room A is consistently detected even when in Room B, ensure Node B is not obstructed by metal objects.

---

## 🐛 Troubleshooting

| Symptom | Probable Cause | Fix |
| :--- | :--- | :--- |
| **"Waiting for Data..."** | ESP32s not sending UDP. | Check Laptop Firewall (`sudo ufw allow 8888/udp`) or ESP32 Power. |
| **"Buffering..." forever** | One node is offline. | Run `ping` to find the dead node. Both must be active. |
| **Room B detected as Room A** | Signal Bias. | Ensure you are using the latest `models/best.pth` trained with Log-Diff features. |
| **Camera not working** | `cv2` device index. | Change `device_id=0` to `1` in `run_inference.py`. |

---

> **Author**: Punnay
> **Date**: Feb 2026
