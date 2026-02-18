# Wi-Fi Human Sensing: Through-Wall Detection with ESP32 & AI

![Wi-Fi Sensing](https://img.shields.io/badge/Wi--Fi-Sensing-blue)
![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-green)
![Status: Debugging](https://img.shields.io/badge/Status-Debugging_Signal_Bias-critical)

A deep learning project attempting to track human presence, location (Room 1 vs Room 2), and pose using only Wi-Fi signals from two ESP32 nodes.

**Current Status (Feb 2026):** The system successfully captures and visualizes CSI data, but currently suffers from a critical **Signal Strength Bias** where "Room B" is frequently detected even when the user is in "Room A". Hardware and software investigations are ongoing.

---

## 🚀 Concept
1.  **Traffic Generator**: A script floods the network with 100Hz UDP packets.
2.  **CSI Capture**: Two ESP32s (Node A & Node B) intercept these packets and measure the "Channel State Information" (CSI)—how Wi-Fi signals bounce around the room.
3.  **AI Processing**: A **1D CNN + LSTM** neural network analyzes these signal distortions to infer human presence and location.

---

## 🛠️ Hardware Setup

*   **Receiver A (Node A)**: ESP32 in **Room 1** (Target IP: `10.42.0.149`).
*   **Receiver B (Node B)**: ESP32 in **Room 2** (Target IP: `10.42.0.173`).
*   **Transmitter**: Laptop (Linux) running a Wi-Fi Hotspot (`10.42.0.1`).
*   **Camera**: Webcam for ground-truth labeling during training.

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
*   Follow prompts to record "Room A", "Room B", and "Empty" scenarios.
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
*   **Note**: This now uses **Global Scaling** (Median/IQR calculated across the entire dataset) to preserve relative signal strength differences between nodes.

### 4. Run Live Inference
Starts the real-time detection engine.
```bash
python3 scripts/run_inference.py --debug_stats
```
*   `--debug_stats`: Prints the live Scaled Energy of Node A vs Node B to the console.

---

## 🐛 Troubleshooting & Known Issues

### **Critical Issue: Room B Bias**
**Symptom**: The model correctly identifies "Room B" when the user is in Room B, but **also** detects "Room B" (or fluctuates heavily) when the user is in Room A. "Room A" detection is rare or unstable.

**Attempted Solutions (None Worked So Far):**
1.  **Global Scaling**: We switched from "Instance Normalization" (which made both nodes look identical) to "Global Scaling" to preserve Node A's naturally higher amplitude.
    *   *Result*: Training data shows separation (Node A variance > Node B variance in Room A), but live inference still output incorrect predictions.
2.  **Physical Verification**: Confirmed Node A (`10.42.0.149`) is physically louder/closer than Node B (`10.42.0.173`).
    *   *Result*: Verified. Node A Raw Mean ~15, Node B Raw Mean ~13.
3.  **Strict IP Mapping**: Hardcoded IP-to-Slot assignments in `dataset.py` and `run_inference.py` to prevent channel swapping.
    *   *Result*: Verified correct slots, but bias persists.
4.  **Retraining**: Retrained model from scratch on Global Scaled data.
    *   *Result*: No significant improvement in live accuracy.

**Current Hypothesis**:
The model might be over-fitting to the specific *shape* of the Room B reflections rather than the *magnitude* of the signal. Alternatively, the "Global Scaler" might be skewed by the "Empty" room data if it dominates the dataset, compressing the dynamic range for the "Human" class.

See `CSI_VISION.md` for detailed technical breakdown.
