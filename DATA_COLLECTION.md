# Wi-Fi CSI Data Collection Guide (Robust Protocol)

This guide explains how to collect **High-Quality, Robust Data** for training the Wi-Fi Sensing AI. Following this protocol strictly is critical for the model to differentiate between **Room 1 (Line-of-Sight)** and **Room 2 (Through-Wall)**.

## 1. Hardware Setup (The "Golden" Layout)

### Components
*   **1x ESP32 (Receiver)**: Captures CSI data.
*   **1x Wi-Fi Router/AP (Transmitter)**: Sends packets.
*   **1x Laptop**: Runs the AI and Traffic Generator.
*   **1x Webcam**: Connected to laptop (for ground truth labeling).

### Placement
1.  **Wi-Fi Router**: Place in the **Corner of Room 1** (near the wall shared with Room 2).
2.  **ESP32 Receiver**: Place in the **Center of Room 1**.
    *   **Height**: Chest level (~1.2m). Do not place on the floor!
3.  **Laptop + Camera**: Place in **Room 1**, facing the user. Ensure the camera has a clear view of Room 1.

---

## 2. Software Preparation

1.  **Flash ESP32**: Ensure `firmware/esp32_csi_rx` is uploaded.
2.  **Connect**: Laptop must be on the same Wi-Fi network.
3.  **Start Traffic Generator (NEW)**:
    **Terminal 1 (Traffic Generator)**:
    *Sends high-speed packets to the ESP32 to "illuminate" the environment.*
    ```bash
    # Replace 192.168.1.50 with your actual ESP32 IP address
    python3 scripts/traffic_generator.py --ip 10.42.0.173 --rate 100 --size 10
    ```

    **Terminal 2 (AI Brain)**:
    *Reads CSI from ESP32, runs the Neural Network, and visualizes the result.*
    ```bash
    python3 scripts/run_inference.py --rf_mode esp32
    ```

---

## 3. Data Collection Protocol (The "7-Session" Standard)

We use `scripts/master_collector.py` to automate this. Each session focuses on a specific zone and activity mix.

**Total Time**: ~12 Minutes.

### 🛑 IMPORTANT RULES
*   **Do NOT just sit still.** The AI learns from *motion*. Even when "Sitting", you should shift your weight, move your arms, or turn your head occasionally.
*   **Do NOT block the LOS completely for too long.** In Room 1, try to move *around* the ESP32, not just stand directly between it and the router for the whole minute.

### Session Breakdown

| Session | Zone | Description & Action | Duration |
| :--- | :--- | :--- | :--- |
| **1** | **Room 1 (Walk)** | **Normal Walking**. Walk randomly around Room 1. Cross the Line-of-Sight path multiple times. vary your speed (slow/fast). | 60s |
| **2** | **Room 1 (Sit/Stand)** | **Micro-Motion**. Place a chair in Room 1. Sit down for 10s, Stand up for 10s. Repeat. Move your arms while sitting (e.g., typing, drinking). | 60s |
| **3** | **Room 1 (Complex)** | **Random Actions**. Do squats, wave hands, pick up objects from the floor. This teaches the AI "Human Body" signature. | 60s |
| **4** | **Room 2 (Through-Wall)** | **Walk & Sit**. Go to Room 2. Walk along the shared wall. Sit on the bed/chair in Room 2. **CRITICAL**: The camera will NOT see you. This is expected! The system uses "Blind Labeling" to learn this zone. | 90s |
| **5** | **Room 2 (Deep)** | **Deep Penetration**. Go to the far end of Room 2. Walk back and forth. This teaches the AI weak signal handling. | 60s |
| **6** | **Hallway** | **Transition**. Walk from Room 1 -> Hallway -> Room 2 and back. Pause in the hallway for 5 seconds. | 60s |
| **7** | **Empty** | **Baseline**. Leave the area completely. Ensure NO ONE is in Room 1, Room 2, or Hallway. This is crucial for the "Presence=0" class. | 60s |

### How to Run

1.  Start the **Traffic Generator** (Terminal 1).
2.  Start the **Collector** (Terminal 2):
    ```bash
    python3 scripts/master_collector.py
    ```
3.  Follow the on-screen prompts. Enter the label when asked (e.g., `room1`, `room2`, `empty`).

---

## 4. Processing & Training

After collection, you must process the data with the new **Adaptive Normalization**.

1.  **Process Data**:
    ```bash
    python3 scripts/process_all_data.py --force_relabel
    ```
    *   This extracts skeletons (MediaPipe) and aligns timestamps.

2.  **Train Model**:
    ```bash
    python3 scripts/train_local.py --all_data --epochs 50
    ```
    *   This will automatically fit the `AdaptiveScaler` to your specific environment and save it to `models/scaler.json`.
    *   It uses **Data Augmentation** (Noise, Shift, Scale) to make the model robust.

3.  **Run Inference**:
    ```bash
    python3 scripts/run_inference.py --rf_mode esp32
    ```
    *   This now uses **Probability Smoothing** to prevent the stick figure from flickering.

---

## 5. Troubleshooting Bad Data

*   **"Stick Figure Flickers"**: You likely captured too much "Empty" data labeled as "Room 1", or you stood too still. Move more during recording!
*   **"Room 2 not detected"**: The wall might be too thick for the default TX power. Move the Router closer to the shared wall.
*   **"FPS is Low"**: Check the Traffic Generator. Ensure `--size 10` is used. Large packets cause congestion.
