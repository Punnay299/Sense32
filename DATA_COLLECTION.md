# Wi-Fi CSI Data Collection Guide

This guide explains how to set up the hardware and collect high-quality data for training the Wi-Fi Sensing AI.

## 1. Hardware Setup (The "Perfect" Layout)

To achieve robust detection across multiple zones (Room 1, Room 2, Hallway), specific placement of devices is crucial.

### Components
*   **1x ESP32 (Receiver)**: Captures CSI data.
*   **1x Wi-Fi Router/AP (Transmitter)**: Sends packets.
*   **1x Laptop**: Runs the AI and Traffic Generator.
*   **1x Webcam**: Connected to laptop (for ground truth labeling).

### Placement
1.  **Wi-Fi Router**: Place in the **Corner of Room 1** (near the wall shared with Room 2).
2.  **ESP32 Receiver**: Place in the **Center of Room 1**.
3.  **Laptop + Camera**: Place in **Room 1**, facing the user. ensure the camera has a clear view of Room 1.
4.  **Target Zones**:
    *   **Room 1**: The area between the Router and ESP32. (Line-of-Sight).
    *   **Room 2**: The room adjacent to Room 1. (Non-Line-of-Sight / Through-Wall).
    *   **Hallway**: The path connecting Room 1 and Room 2.

> **Why this setup?**
> Placing the router in the corner and ESP32 in the center creates a "sensing web" that covers both the room it's in (LOS) and interacts with reflections from the adjacent room (NLOS).

---

## 2. Software Preparation

Ensure your environment is ready.

1.  **Flash ESP32**: Upload `firmware/esp32_csi_rx/esp32_csi_rx.ino` to your ESP32.
2.  **Connect**: Connect your Laptop to the **same Wi-Fi network** as the ESP32 (defined in the firmware).
3.  **Verify IP**: Find the IP address of your ESP32 (e.g., `10.42.0.173`).

---

## 3. Data Collection Process (Automated)

We use a **Master Script** to automate the entire process. This ensures consistent data quality by running a specific "Traffic Generator" that floods the network with high-rate UDP packets (100Hz), ensuring the AI has enough data to "see" movement.

### Step 1: Run the Master Collector

Open a terminal in the project root:

```bash
python3 scripts/master_collector.py
```

### Step 2: Follow the Prompts

The script will guide you through 7 specific recording sessions. Follow the instructions on screen:

| Session | Zone | Action | Duration |
| :--- | :--- | :--- | :--- |
| **1-3** | **Room 1** | Walk around randomly for 30s, then Sit for 30s. | 60s each |
| **4** | **Room 2** | Walk to Room 2, walk around, and sit. (Camera will NOT see you - this is intentional). | 60s |
| **5-6** | **Hallway** | Walk back and forth between Room 1 and Room 2. | 60s each |
| **7** | **Empty** | Leave the area completely. Record an empty room. | 60s |

> **Note on "Blind Labeling"**:
> When you are in **Room 2**, the camera cannot see you. The system is designed to handle this! It uses "Blind Labeling" to automatically label these samples as "Room 2" based on the session name, even if the computer vision system detects nothing.

---

## 4. Processing the Data

After collection, you must process the raw video and CSI data to generate training labels.

```bash
python3 scripts/process_all_data.py --force_relabel
```

*   **What this does**:
    *   Reads `video.mp4` for each session.
    *   Uses MediaPipe to detect your pose (for Room 1).
    *   Uses **Blind Labeling** logic (for Room 2) to force "Presence" labels even when hidden.
    *   Saves synchronization data to `labels.csv`.

---

## 5. Training the AI

Train the deep learning model (CNN + LSTM) on your new dataset.

```bash
python3 scripts/train_local.py --all_data --epochs 50
```

*   **`--all_data`**: Uses every session in the `data/` folder.
*   **`--epochs 50`**: Trains for 50 iterations (usually enough for convergence).

---

## 6. Live Inference

To see the AI in action:

**Terminal 1 (Traffic Generator)**:
*Sends high-speed packets to the ESP32 to "illuminate" the environment.*
```bash
python3 scripts/traffic_generator.py --ip <ESP32_IP> --rate 100
```

**Terminal 2 (AI Brain)**:
*Reads CSI from ESP32, runs the Neural Network, and visualizes the result.*
```bash
python3 scripts/run_inference.py --rf_mode esp32
```

You should now see a stick figure that tracks your movement, even through walls!
