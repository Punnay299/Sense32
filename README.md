# Wi-Fi Human Sensing: See Through Walls with ESP32 & AI

![Wi-Fi Sensing Banner](https://img.shields.io/badge/Wi--Fi-Sensing-blue)
![Python 3.11](https://img.shields.io/badge/Python-3.11-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-green)

A cutting-edge deep learning project that tracks human presence, location (Room 1 vs Room 2), and pose (stick figure) using only Wi-Fi signals. No cameras required for the final deployment!

**Key Features:**
*   **Through-Wall Detection**: Detects human presence in adjacent rooms (NLOS).
*   **Real-Time Pose Estimation**: Visualizes a stick figure purely from CSI (Channel State Information) data.
*   **Zone Classification**: Accurately identifies if a person is in Room 1, Room 2, or the Hallway.
*   **Autopilot Data Collection**: Automated tools to generate high-quality training datasets.

---

## 🚀 How It Works

1.  **Traffic Generator**: A script floods the network with high-rate (100Hz) UDP packets.
2.  **CSI Capture (ESP32)**: An ESP32 microcontroller intercepts these packets and extracts the "Channel State Information" (CSI)—a complex matrix describing how the Wi-Fi waves bounced around the room.
3.  **AI Processing**: A **1D CNN + LSTM** neural network analyzes the CSI distortions caused by the human body.
4.  **Result**: The AI outputs the person's location and 33-point skeletal pose in real-time.

---

## 🛠️ Hardware Setup

*   **Receiver**: ESP32 DevKit V1 (running custom firmware).
*   **Transmitter**: Standard Wi-Fi Router or a second ESP32.
*   **Server**: Laptop/PC with Python 3.11 & GPU (optional but recommended).

**Placement Strategy:**
*   **Router**: Corner of Room 1.
*   **ESP32**: Center of Room 1.
*   This creates a "sensing web" effective for both Line-of-Sight (Room 1) and Non-Line-of-Sight (Room 2).

*See [DATA_COLLECTION.md](DATA_COLLECTION.md) for detailed setup diagrams.*

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
    *   Open `firmware/esp32_csi_rx/esp32_csi_rx.ino` in Arduino IDE.
    *   Set your Wi-Fi credentials (`SSID`, `PASSWORD`) and Laptop IP (`TARGET_IP`).
    *   Flash to your ESP32.

---

## 🏃 Usage

### 1. Collect Training Data
We use a **Master Script** to guide you through recording Room 1, Room 2, and Hallway scenarios.
```bash
python3 scripts/master_collector.py
```
*(Follow the on-screen prompts. Records valid data for ~10 minutes).*

### 2. Process Labels
Extracts ground-truth pose labels from the webcam (using MediaPipe) and applies **Blind Labeling** for hidden zones.
```bash
python3 scripts/process_all_data.py --force_relabel
```

### 3. Train the AI
Trains the custom CNN-LSTM model.
```bash
python3 scripts/train_local.py --all_data --epochs 50
```

### 4. Run Live Demo
Start the traffic generator (Terminal 1) and the Inference Engine (Terminal 2).

**Terminal 1:**
```bash
python3 scripts/traffic_generator.py --ip <ESP32_IP> --rate 100
```

**Terminal 2:**
```bash
python3 scripts/run_inference.py --rf_mode esp32
```

---

## 📂 Project Structure

*   `src/`: Core libraries (CSI capture, Model architecture, Dataset loader).
*   `scripts/`: Executable tools for training, inference, and data collection.
*   `firmware/`: ESP32 Arduino C++ code.
*   `data/`: Stores recorded sessions (gitignored).
*   `models/`: Stores trained `.pth` model checkpoints.

---

## 🧠 Technical Details

*   **Model**: A hybrid architecture using 1D Convolutional layers (for spatial feature extraction from subcarriers) followed by LSTM layers (for temporal dynamics).
*   **Inputs**: 64 subcarriers of CSI amplitude/phase.
*   **Outputs**:
    *   **Pose**: 66 values (33 keypoints x 2 coordinates).
    *   **Presence**: Probability (0-1).
    *   **Location**: Classification (Room 1, Room 2, Hallway, Empty).

*See [CSI_VISION.md](CSI_VISION.md) for a deep dive into the theory.*

---

## 🤝 Contributors
*   **Punnay** - Lead Developer & Research

LICENSE: MIT
