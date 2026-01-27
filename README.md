# Wi-Fi Human Detection & Pose Estimation 📡🚶

> **"Seeing Through Walls" with standard Wi-Fi hardware.**

![Project Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-With_CUDA-orange)
![Scapy](https://img.shields.io/badge/Scapy-2.5+-yellow)

**Wi-Fi Human Detection** is an AI-powered sensing system that transforms a standard Linux laptop into a biological sensor. By capturing and analyzing raw Wi-Fi signal perturbations (RSSI/RTT), it uses a custom Deep Neural Network (CNN-LSTM) to estimate human pose, even through walls.

This project was developed and tested using **Python 3.11.8** via `pyenv`.

---

## 🚀 Features
*   **Non-Invasive**: No cameras required for the end-user (inference runs on RF signals only).
*   **Through-Wall Capability**: Detects subjects hidden behind obstacles using high-frequency packet sniffing.
*   **Privacy-First**: Edge AI execution. No video or data is uploaded to the cloud.
*   **Hardware-Agnostic**: Works with most standard Wi-Fi cards (Intel, Atheros) via standard Linux kernel interfaces.
*   **Auto-Tuning**: Automatically handles Monitor Mode switching and Channel Hopping.

---

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/wifi-human-detection.git
    cd wifi-human-detection
    ```

2.  **Set Up Environment (Use Python 3.11.8)**
    ```bash
    # Ensure you are using Python 3.11.8
    pyenv local 3.11.8
    
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Verify Setup**
    ```bash
    python check_env.py
    ```

---

## 📖 Operational Workflow

Follow these exact commands to run the project.

### 1. Collect Training Data
**Important**: You must run with `sudo` to access the Wi-Fi card in monitor mode.
```bash
# This records RF signals and Video (for ground truth)
sudo ./venv/bin/python scripts/collect_data.py --name session_01 --rf_mode scapy --duration 60
```

### 2. Fix Permissions
Since collection ran as root, you must claim ownership of the data files before processing.
```bash
sudo chown -R $USER:$USER data/
```

### 3. Process Data (Generate Labels)
Extracts skeletons from the video to create the training dataset.
```bash
./venv/bin/python scripts/process_all_data.py
```

### 4. Train the Model
Trains the CNN-LSTM network to map RF signals to Pose.
```bash
./venv/bin/python scripts/train_local.py --all_data --epochs 100
```

### 5. Run Live Inference
Turn off the lights, or walk behind a door. The system will visualize your skeleton based *only* on the Wi-Fi signals.
```bash
sudo ./venv/bin/python scripts/run_inference.py --rf_mode scapy
```

---

## ❓ Troubleshooting

### "Permission Denied" Errors
If you see permission errors during processing, you likely skipped Step 2. Run `sudo chown -R $USER:$USER data/`.

### Empty RF Data
The system includes robust verification. If `collect_data.py` exits saying "WARNING: NO RF DATA RECEIVED", check your Wi-Fi interface name or ensure no other monitor processes (like airmon-ng) are conflicting.

---

## 📄 License
[MIT](https://choosealicense.com/licenses/mit/)
