# Wi-Fi Human Detection & Pose Estimation 📡🚶

> **"Seeing Through Walls" with standard Wi-Fi hardware.**

![Project Status](https://img.shields.io/badge/Status-Operational-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.9-yellow)

## 📖 Overview

This project is an **AI-powered Sensing System** that detects human presence and estimates 2D skeletal poses using only **Wi-Fi signals** (RSSI & RTT). It transforms a standard Linux laptop into a biological sensor, capable of detecting motion and posture even through obstacles.

✅ **Privacy-First**: No cameras required for the end-user (inference runs on RF signals only).
✅ **Hardware-Agnostic**: Works with standard Wi-Fi cards (Intel, Atheros) via standard Linux kernel interfaces.
✅ **Robust AI**: Uses a custom **CNN-LSTM** Grid Neural Network trained on fusion data.

---

## 🚀 Quick Start

### 1. Installation
Clone the repo and set up the environment:
```bash
# Create Virtual Environment
python -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Verify Setup
Ensure you have a supported Wi-Fi interface:
```bash
./venv/bin/python check_env.py
```

### 3. Run the Demo (Inference)
Launch the real-time visualizer. This will show the camera feed (for verification) overlaid with the stick-figure predicted purely from Wi-Fi data:
```bash
./venv/bin/python scripts/run_inference.py --rf_mode linux --model models/best.pth
```
*(Note: Press `q` to quit)*

---

## 🛠️ Development Workflow

For researchers and developers wanting to retrain the model:

1.  **Collect Data**: Record yourself walking while the system captures Wi-Fi + Video.
    ```bash
    ./venv/bin/python scripts/collect_data.py --name my_session --duration 60
    ```
2.  **Label Data**: Use Computer Vision (MediaPipe) to generate "Ground Truth" labels.
    ```bash
    ./venv/bin/python scripts/process_all_data.py
    ```
3.  **Train Model**: Train the Neural Network on your new data (GPU Accelerated).
    ```bash
    ./venv/bin/python scripts/train_local.py --all_data --epochs 50
    ```

---

## 📚 Documentation

For a deep dive into the architecture, signal processing logic, and directory structure, please read the **[Comprehensive Project Report](PROJECT_REPORT.md)**.

## 🤝 Contribution
*   **Issues**: Please check `PROJECT_REPORT.md` for known limitations before filing issues.
*   **Pull Requests**: Ensure all tests pass (`python -m pytest tests/`) before submitting.
