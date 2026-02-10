# Wi-Fi Human Sensing: Technical Deep Dive

## Abstract

This document details the theoretical foundation, system architecture, and implementation details of a Wi-Fi Channel State Information (CSI) based Human Sensing System. We leverage the fine-grained physical layer information from commodity Wi-Fi hardware (ESP32) to detect human presence, classify zones (Line-of-Sight vs. Non-Line-of-Sight), and estimate 2D human pose.

Unlike traditional RSSI-based systems which only measure signal strength, CSI captures the amplitude and phase of multiple subcarriers (frequencies) within a Wi-Fi channel. This allows us to observe detailed multipath propagation effects caused by human body movements.

---

## 1. Theoretical Foundation

### 1.1 Channel State Information (CSI)

In modern MIMO-OFDM (Multiple Input Multiple Output - Orthogonal Frequency Division Multiplexing) systems like 802.11n/ac, the wireless channel is modeled as:

$$ Y = H \times X + N $$

Where:
*   **Y**: Received Signal
*   **X**: Transmitted Signal
*   **H**: Channel Matrix (CSI)
*   **N**: Noise

The Channel Matrix **H** contains the complex channel frequency response (CFR) for each subcarrier *k*:

$$ H(k) = \|H(k)\| e^{j \angle H(k)} $$

*   **Amplitude ||H(k)||**: Represents signal attenuation due to path loss, shadowing, and fading.
*   **Phase ∠H(k)**: Represents signal delay and phase shifts due to multipath propagation.

When a human moves in the environment, they reflect and scatter Wi-Fi signals. This changes the multipath profile, causing dynamic variations in both amplitude and phase across different subcarriers. These variations are the "fingerprint" of human motion.

### 1.2 The "Fresnel Zone" Model

We can model human sensing using Fresnel Zones—ellipsoidal regions between the transmitter (TX) and receiver (RX).

1.  **Line-of-Sight (LOS)**: The direct path between TX and RX. If a person blocks this, amplitude drops significantly.
2.  **Reflected Paths (NLOS)**: Signals bounce off walls and objects. A person moving in a reflected path changes the phase length of that path, causing constructive or destructive interference at the receiver.

Our ESP32 system captures 64 subcarriers. Each subcarrier has a slightly different wavelength, meaning they sample the space with different spatial resolutions. By combining all 64, we get a high-resolution "image" of the RF environment.

---

## 2. System Architecture

The project consists of three main pipelines:

### 2.1 Data Acquisition Pipeline

*   **Hardware**: ESP32 DevKit V1 (Tensilica Xtensa LX6).
*   **Firmware**: Custom C++ firmware using `esp_wifi_set_csi_rx_cb`.
*   **Protocol**:
    *   **Traffic Generator**: A Python script sends high-rate (100Hz) UDP packets from the Laptop (AP) to the ESP32 (Station).
    *   **CSI Capture**: The ESP32 extracts the CSI matrix from each received packet.
    *   **UDP Streaming**: The ESP32 encapsulates the CSI (64 bytes imaginary, 64 bytes real) + RSSI + Timestamp into a custom UDP packet and streams it back to the laptop.
    *   **Source Locking**: The Python receiver (`run_inference.py`) automatically locks onto the dominant MAC/IP source to prevent "jitter" from rogue APs or secondary ESP32s.

### 2.2 Neural Network Architecture (`WifiPoseModel`)

We use a hybrid **CNN-LSTM** architecture designed for spatiotemporal feature extraction.

*   **Input**: Tensor of shape `[Batch, 64, SeqLen]`.
    *   64 Channels (Subcarriers).
    *   SeqLen=50 (0.5 seconds of history at 100Hz).
*   **Encoder (1D CNN)**:
    *   3 layers of 1D Convolution (`kernel_size=3`, `padding=1`).
    *   Extracts spatial features (correlations between adjacent subcarriers).
    *   **Batch Normalization** and **ReLU** activation.
*   **Temporal Processor (LSTM)**:
    *   Takes the flattened features from the CNN.
    *   Input Dimension: 64 * FeatureMapSize.
    *   Hidden Dimension: 256.
    *   Captures the time-evolution of the signal (e.g., the doppler shift pattern of a walking person).
*   **Heads (Multi-Task Learning)**:
    *   **Pose Head**: Fully Connected (256 -> 66). Outputs 33 (x,y) keypoint coordinates.
    *   **Presence Head**: Fully Connected (256 -> 1). Sigmoid activation. Probability of human presence.
    *   **Location Head**: Fully Connected (256 -> 4). Softmax activation. Classifies Zone (Room 1, Room 2, Hallway, Empty).

### 2.3 Training Pipeline (Self-Supervised / Weakly Supervised)

Generating ground truth for Wi-Fi sensing is the hardest challenge. We solved it using a **Cross-Modal Supervision** approach.

1.  **Vision-Based "Teacher"**: A webcam records the user. We use **MediaPipe Pose** to extract 33 skeletal keypoints. These become the "Ground Truth" labels.
2.  **RF-Based "Student"**: The ESP32 captures CSI simultaneously.
3.  **Synchronization**: We align Visual and RF streams using monotonic timestamps.

**The "Blind Labeling" Innovation:**
For Through-Wall (NLOS) scenarios (Room 2), the camera cannot see the user.
*   **Problem**: MediaPipe returns "No Person", which confuses the model (RF shows strong activity, Label says "Empty").
*   **Solution**: We implement "Blind Labeling".
    *   During data collection for Room 2, we *know* the user is present.
    *   We override the camera's "Empty" label with "Presence=1" and "Location=Room 2".
    *   This forces the model to learn the specific RF signature of Room 2, even without visual confirmation.

---

## 3. Implementation Details

### 3.1 Data Sanitization & Normalization

Raw CSI data is noisy and hardware-dependent.
1.  **Amplitude Extraction**: $ \sqrt{Real^2 + Imag^2} $.
2.  **Outlier Removal**: We clip extremely high amplitudes caused by hardware glitches.
3.  **Normalization**:
    *   **Initial**: We tried Z-Score (Standardization).
    *   **Final (Robust)**: We use **Min-Max Scaling** (dividing by 127.0). This proved superior because signal strength (amplitude) carries vital information about distance (Room 1 vs Room 2). Z-Score was removing this distance information.

### 3.2 Dual-Stream Processing

The `run_inference.py` script runs two parallel threads:
1.  **RF Thread**: High-priority. Reads UDP socket, parses CSI, updates a circular buffer (Deque).
2.  **Inference Thread**: Runs the PyTorch model on the buffer content.
    *   We check `len(buffer) == 50`.
    *   If buffer is full, we infer.
    *   We use `torch.no_grad()` for speed.
    *   On GPU (CUDA), inference takes <5ms, allowing for 100+ FPS real-time performance.

---

## 4. Advanced Techniques

### 4.1 Traffic Generation Strategy
Passive sniffing is unreliable (packet rates fluctuate with network load). We switched to **Active Injection**.
*   We use a Python script (`traffic_generator.py`) to blast UDP packets.
*   Rate: 100Hz (1 packet every 10ms).
*   Result: A consistent, high-density "illumination" of the RF environment, critical for capturing subtle motions like breathing or sitting.

### 4.2 Handling Domain Shift
We discovered a massive domain shift between "Ping" traffic and "UDP Flood" traffic.
*   Ping packets are small and infrequent -> Low CSI Amplitudes.
*   UDP Flood packets are large and frequent -> High CSI Amplitudes.
*   **Fix**: We mandated that **Training Data MUST be collected using the Traffic Generator**. This ensures the data distribution (amplitude/phase variance) matches exactly between Training and Inference.

---

## 5. Future Work

1.  **Phase Sanitization**: Currently, we use raw phase, which is corrupted by Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO). Implementing linear phase sanitization could improve accuracy.
2.  **Multi-Person Tracking**: The current model assumes a single subject. Upgrading to a Transformer-based architecture could allow for distinguishing multiple reflections.
3.  **Edge Deployment**: Quantizing the model (int8) to run directly on a Raspberry Pi or even the ESP32-S3 itself using TensorFlow Lite for Microcontrollers.

---

## 6. Detailed Bibliography & References

1.  **Wang et al. (2019)**: "Phone-to-Phone WiFi Sensing".
2.  **Ma et al. (2019)**: "WiFi-based Human Pose Estimation".
3.  **DensePose**: "Dense Human Pose Estimation In The Wild".
4.  **MediaPipe**: Google's framework for multimodal learning.

---

*> "The walls have ears... and now they have eyes."*
