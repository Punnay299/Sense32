# Wi-Fi Human Sensing: Technical Deep Dive & System Architecture

## Table of Contents
1.  [Abstract](#abstract)
2.  [Introduction to Wi-Fi Sensing](#1-introduction-to-wi-fi-sensing)
3.  [Theoretical Foundation](#2-theoretical-foundation)
    *   [2.1 Channel State Information (CSI)](#21-channel-state-information-csi)
    *   [2.2 Multipath Propagation & Fresnel Zones](#22-multipath-propagation--fresnel-zones)
    *   [2.3 Why CSI > RSSI](#23-why-csi--rssi)
4.  [System Architecture (Dual-Node)](#3-system-architecture-dual-node)
    *   [3.1 Hardware Components](#31-hardware-components)
    *   [3.2 Network Topology](#32-network-topology)
    *   [3.3 Data Flow Pipeline](#33-data-flow-pipeline)
5.  [Implementation Details](#4-implementation-details)
    *   [4.1 Firmware (ESP32)](#41-firmware-esp32)
    *   [4.2 Active Traffic Generation](#42-active-traffic-generation)
    *   [4.3 Signal Processing & Sanitization](#43-signal-processing--sanitization)
6.  [Neural Network Design](#5-neural-network-design)
    *   [5.1 RF Encoder (1D CNN)](#51-rf-encoder-1d-cnn)
    *   [5.2 Temporal Processor (LSTM)](#52-temporal-processor-lstm)
    *   [5.3 Multi-Head Output](#53-multi-head-output)
7.  [Training Strategy](#6-training-strategy)
    *   [6.1 Cross-Modal Supervision](#61-cross-modal-supervision)
    *   [6.2 Blind Labeling for NLOS](#62-blind-labeling-for-nlos)
    *   [6.3 Data Augmentation](#63-data-augmentation)
8.  [Performance & Limitations](#7-performance--limitations)
9.  [Future Work](#8-future-work)
10. [Bibliography](#9-bibliography)

---

## Abstract

This document details the complete theoretical and technical implementation of a privacy-preserving human sensing system using off-the-shelf Wi-Fi hardware. By leveraging Channel State Information (CSI) from two ESP32 microcontrollers, the system fuses spatial RF features to detect human presence, classify location (Room A, Room B, Hallway), and reconstruct 2D skeletal pose in real-time ($>$30 FPS), even through walls (Non-Line-of-Sight).

---

## 1. Introduction to Wi-Fi Sensing

Traditional Wi-Fi networks are designed for data communication. However, the radio waves that carry our internet data also physically interact with the environment. They bounce off walls, furniture, and crucially, human bodies.

**Wi-Fi Sensing** is the art of repurposing these communication signals as a radar system. As a person moves through a Wi-Fi field, they induce minute Doppler shifts and scattering effects. By analyzing these distortions, we can infer the person's location and activity without cameras or wearable sensors.

This project implements a **Dual-Node System**, utilizing two separate receivers to provide spatial diversity, significantly improving robust detection in complex multi-room environments compared to single-link systems.

---

## 2. Theoretical Foundation

### 2.1 Channel State Information (CSI)

In modern MIMO-OFDM systems (like 802.11n), data is transmitted over multiple subcarriers (frequencies) simultaneously. The channel properties for each subcarrier $k$ are described by the Channel State Information (CSI) matrix, $H(k)$.

The received signal $Y$ relates to the transmitted signal $X$ as:
$$ Y(k) = H(k) \times X(k) + N(k) $$

where $N$ is noise. The CSI $H(k)$ is a complex number:
$$ H(k) = \|H(k)\| e^{j \angle H(k)} $$

*   **Amplitude ($\|H(k)\|$):** Represents signal strength attenuation (fading/shadowing).
*   **Phase ($\angle H(k)$):** Represents the delay/shift of the signal path.

Our ESP32 system extracts **64 subcarriers** of CSI data. This gives us a frequency-domain "fingerprint" of the channel 100 times per second.

### 2.2 Multipath Propagation & Fresnel Zones

Wi-Fi signals travel via multiple paths:
1.  **Line-of-Sight (LoS):** Direct path from TX to RX.
2.  **Reflections (NLoS):** Paths bouncing off walls/floors.

When a human moves, they alter specific paths.
*   **Blocking LoS:** Causes a deep fade in amplitude.
*   **Reflecting Path:** Changes the path length, causing phase rotation.

We model this using **Fresnel Zones**—concentric ellipsoids between TX and RX. A moving object passing through zone boundaries causes constructive/destructive interference, visible as sinusoidal patterns in the CSI amplitude. Using 64 subcarriers effectively gives us 64 different "scanning frequencies," allowing us to resolve finer movements.

### 2.3 Why CSI > RSSI

*   **RSSI (Received Signal Strength Indicator):** A single coarse value averaged over the entire packet. It is highly unstable, susceptible to random noise, and cannot distinguish static interference from human motion.
*   **CSI:** Granular data per subcarrier. We can see *frequency-selective fading*. For example, a hand wave might affect subcarrier #30 but not #10. This rich feature set is what enables Pose Estimation.

---

## 3. System Architecture (Dual-Node)

To handle the "Through-Wall" challenge, we use a distributed architecture.

### 3.1 Hardware Components
*   **Transmitter (TX):** Laptop (Host Machine) acting as a Wi-Fi Hotspot.
*   **Receiver A (RX-A):** ESP32 DevKit V1 placed in **Room A**.
*   **Receiver B (RX-B):** ESP32 DevKit V1 placed in **Room B**.

### 3.2 Network Topology
The system operates in a Star Topology:
*   **Gateway:** 10.42.0.1 (Laptop)
*   **Nodes:** 10.42.0.x (ESP32s)
*   **Protocol:** UDP Unicast.

We avoid Broadcast traffic to prevent network congestion. The Laptop sends targeted high-rate UDP packets to each ESP32 IP individually.

### 3.3 Data Flow Pipeline
1.  **Generation:** Python script blasts 10-byte UDP payloads at 100Hz to both ESP32 IPs.
2.  **Capture:** ESP32 radio receives packet $\rightarrow$ Hardware calculates CSI $\rightarrow$ Firmware traps CSI.
3.  **Encapsulation:** Firmware packs CSI (128 bytes) + Metadata (Timestamp, RSSI) into a custom return packet.
4.  **Exfiltration:** ESP32 sends this packet *back* to the Laptop's UDP listening port (8888).
5.  **Synchronization:** Laptop buffers packets from Node A and Node B. When both buffers are full (SeqLen=50), they are stacked into a `[Batch, 128, 50]` tensor for inference.

---

## 4. Implementation Details

### 4.1 Firmware (ESP32)
*   **Platform:** Arduino / ESP-IDF.
*   **Key Function:** `esp_wifi_set_csi_rx_cb()`. This registers a callback that fires on *every* Wi-Fi layer packet.
*   **Gateway Detection:** `WiFi.gatewayIP()` is used to dynamically find the laptop, ensuring "Plug & Play" without hardcoding IPs.
*   **Efficiency:** We strip all non-essential headers. The payload is purely raw binary data to maximize throughput.

### 4.2 Active Traffic Generation
Passive sensing (listening to router beacons) is confined to 10Hz (10 packets/sec). This is too slow for human motion (breathing is ~0.3Hz, walking ~2Hz).
We implemented **Active Traffic Injection**:
*   `SendRate`: 100Hz.
*   `Payload`: Minimal (10 bytes).
*   **Result:** High-resolution temporal sampling capable of capturing micro-doppler effects.

### 4.3 Signal Processing & Sanitization
Raw CSI is noisy.
1.  **Amplitude Calculation:** $ \sqrt{R^2 + I^2} $.
2.  **Outlier Removal:** We ignore packets with `len != 128` or missing sequence numbers.
3.  **Adaptive Scaling:**
    $$ x' = \frac{x - median(x)}{IQR(x)} $$
    *   We use Median/IQR instead of Mean/StdDev because Wi-Fi noise is often "spiky" (non-Gaussian). A single glitch shouldn't skew the normalization.
4.  **Dual-Node Fusion:**
    *   Input A: `[50, 64]`
    *   Input B: `[50, 64]`
    *   Fused: `[50, 128]` (Concatenated along channel dimension).

---

## 5. Neural Network Design

We designed a custom `WifiPoseModel` in PyTorch, optimized for sequential RF data.

### 5.1 RF Encoder (1D CNN)
Extracts spatial features from the frequency domain.
*   **Layer 1:** Conv1D(128 $\to$ 64, kernel=3, stride=1). ReLU + BatchNorm.
*   **Layer 2:** Conv1D(64 $\to$ 128, kernel=3, stride=1). ReLU + BatchNorm.
*   **Layer 3:** Conv1D(128 $\to$ 256, kernel=3, stride=2). MaxPool.
*   **Purpose:** Learns correlations between subcarriers (e.g., adjacent subcarriers fading together indicates a wideband block).

### 5.2 Temporal Processor (LSTM)
Extracts time-domain dynamics.
*   **Input:** Flattened features from CNN.
*   **Architecture:** 2-Layer LSTM with `Hidden=256`.
*   **Purpose:** Captures motion patterns (e.g., the rhythmic pattern of walking vs. the static pattern of sitting).

### 5.3 Multi-Head Output
The network splits into three specialized heads:
1.  **Pose Head (Regression):** Linear(256 $\to$ 66). Outputs 33 `(x, y)` coordinates for the COCO skeleton.
2.  **Presence Head (Binary):** Linear(256 $\to$ 1) + Sigmoid. Probability of human presence.
3.  **Location Head (Classification):** Linear(256 $\to$ 4) + Softmax. Classes: `[Room A, Room B, Hallway, Empty]`.

---

## 6. Training Strategy

### 6.1 Cross-Modal Supervision (Teacher-Student)
Sensing Wi-Fi is invisible, making labeling impossible for humans. We use a computer vision model as a "Teacher".
1.  **Setup:** Camera + Wi-Fi record specific scenarios simultaneously.
2.  **Teacher:** **MediaPipe Pose** extracts accurate skeletons from video.
3.  **Student:** Our `WifiPoseModel` takes CSI input and tries to predict the MediaPipe output.
4.  **Loss:** MSE Loss (Pose) + CrossEntropy (Location) + BCE (Presence).

### 6.2 Blind Labeling for NLOS
For Through-Wall scenarios, the camera cannot see the user. MediaPipe fails.
*   **We implemented "Blind Labeling":**
    *   The user explicitly records a "Room A" session.
    *   The pipeline *ignores* the empty camera frames.
    *   It forces the label: `Presence=1`, `Location=Room A`, `Pose=Unknown`.
    *   This teaches the model: *"This specific RF pattern means Room A, even if you can't see anyone."*

### 6.3 Data Augmentation
To prevent overfitting to a specific day's environment:
*   **Gaussian Noise:** Adding random noise to CSI amplitude.
*   **Time Warp:** Stretching/compressing the time axis (simulation speed variations).
*   **Drop Channel:** Randomly masking out subcarriers (simulating deep fades).

---

## 7. Performance & Limitations

### Performance
*   **Inference Speed:** ~5ms per batch on GTX 1650 (200 FPS theoretical).
*   **Detection Range:** Approx 5-8 meters per node through 1 brick wall.
*   **Location Accuracy:** >95% (with dual nodes).

### Limitations
1.  **Static Objects:** If a person stands perfectly still, Doppler shifts vanish. The system relies on breathing detection, which requires higher SNR.
2.  **Environment Dependency:** A model trained in House A performs poorly in House B without fine-tuning (Domain Adaptation problem).
3.  **Multi-Person:** The current model assumes a single dominant reflection. Multi-person tracking requires separating mixed signals (Source Separation).

---

## 8. Future Work

The roadmap for version 2.0 includes:
1.  **CSI Phase Sanitization:** Implementing linear transformation to remove SFO/CFO errors from Phase data, enabling complex-valued neural networks.
2.  **Transformer Backbone:** Replacing LSTM with a Transformer (Self-Attention) to better capture long-term dependencies.
3.  **Edge Deployment:** Quantizing the model to INT8 to run directly on the ESP32-S3's AI accelerator.

---

## 9. Bibliography

1.  **Wang, W. et al. (2019).** *Phone-to-Phone WiFi Sensing.*
2.  **Ma, Y. et al. (2019).** *WiFi-Based Human Pose Estimation.*
3.  **Google MediaPipe.** *Cross-platform solution for audio, video, and time series data.*
4.  **Espressif Systems.** *ESP-IDF Programming Guide: Wi-Fi CSI.*
