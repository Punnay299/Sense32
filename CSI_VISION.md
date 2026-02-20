# Wi-Fi Human Sensing: Technical Deep Dive

## 1. System Architecture Overview

The system is a hybrid **Visual-RF Sensing Platform** designed to track human presence and pose in a multi-room environment. It leverages the complementary strengths of two sensing modalities:
1.  **Computer Vision (Camera)**: Provides high-precision ground truth and line-of-sight detection in the "Hallway".
2.  **Wi-Fi Channel State Information (CSI)**: Provides non-line-of-sight (NLOS) sensing for "Room A" using a dual-node ESP32 array.

### Hardware Topology
*   **Transmitter (TX)**: Linux Host (Laptop) acting as the central AP (SoftAP mode). It broadcasts high-frequency UDP ping packets (100 Hz).
*   **Receiver Network (RX)**:
    *   **Node A (ESP32)**: Positioned in **Room A**. Captures CSI from the TX packets.
    *   **Node B (ESP32)**: Positioned in a secondary location to provide diversity, capturing CSI from the same TX packets.
*   **Synchronization**: The Linux Host acts as the master clock. It batches received CSI packets from both nodes into 500ms time windows (50 frames at 100Hz), ensuring temporal alignment between the two distinct physical views.

---

## 2. Signal Processing Pipeline

The raw RF data (Channel State Information) is noisy and phase-shifted. A rigorous sanitization pipeline is applied before the data reaches the neural network.

### 2.1 Raw Data Structure
Each UDP packet received from an ESP32 contains a payload of 64 subcarriers (OFDM). Use of 20MHz bandwidth on 2.4GHz Wi-Fi (802.11n) provides:
*   **Complex Values**: $H(f) = A(f) e^{j\theta(f)}$
*   **Dimensions**: 2 Nodes $\times$ 64 Subcarriers = 128 raw complex features per time step.

### 2.2 Sanitization Steps
1.  **Amplitude Cleaning (Hampel Filter)**:
    *   A median absolute deviation (MAD) filter is applied to the amplitude stream to remove outliers caused by packet corruption or hardware glitches.
    *   $$ |x_i - \text{median}(W)| > K \cdot \text{MAD} \rightarrow \text{Replace with median} $$

2.  **Phase Sanitization (Linear Detrending)**:
    *   Raw phase is unusable due to Sampling Frequency Offset (SFO) and Packet Detection Delay (PDD).
    *   We apply **Phase Unwrapping** followed by a **Linear Fit Subtraction** to remove the slope caused by the time lag, recovering the true phase variation caused by physical reflections.

3.  **Log-Scaling**:
    *   Wi-Fi signal attenuation follows a log-distance path loss model.
    *   We apply $Log1p(x) = \log(1 + x)$ to compress the dynamic range of the amplitude, allowing the model to learn from both strong line-of-sight signals and weak multi-path reflections.

4.  **Feature Engineering**:
    *   **Difference Features**: To enhance spatial resolution, we calculate the differential log-amplitude between Node A and Node B: $\Delta Amp = \log(Amp_A) - \log(Amp_B)$.
    *   **Final Input Tensor**:
        *   Shape: `[Batch, Seq_Len=50, Features=320]`
        *   Channels: `[Amp_A(64), Phase_A(64), Amp_B(64), Phase_B(64), Diff_Amp(64)]`.

---

## 3. Neural Network Architecture (Hybrid CNN-LSTM)

The model is designed to process spatiotemporal data—learning the "shape" of the RF environment (Space) and how it changes (Time).

### 3.1 RF Encoder (Spatial Feature Extraction)
A **1D Convolutional Neural Network (CNN)** acts as the feature extractor, treating the 320 input subcarriers like pixels in a 1D image.
*   **Layer 1**: `Conv1d(320 -> 64)` + BatchNorm + ReLU + Dropout
*   **Layer 2**: `Conv1d(64 -> 128)`
*   **Layer 3**: `Conv1d(128 -> 256)`
*   **Layer 4**: `Conv1d(256 -> 512)`
*   **Purpose**: Compresses the comprehensive spectral view into a dense 512-dimensional latent vector representing the instantaneous state of the room.

### 3.2 Temporal Processor (Sequence Modeling)
A **Long Short-Term Memory (LSTM)** network processes the sequence of latent vectors.
*   **Input**: Sequence of 50 latent vectors (0.5 seconds).
*   **Structure**: 3-Layer Stacked LSTM with Hidden Dimension 256.
*   **Purpose**: Distinguishes between static interference (furniture) and dynamic interference (human motion). It learns the "gait" and movement patterns imprinted on the RF signal.

### 3.3 Multi-Task Heads
The network splits into three specialized heads:
1.  **Pose Regressor**:
    *   Fully Connected layers mapping LSTM output -> 33 $(x, y)$ keypoints (COCO topology).
    *   Loss: MSE (Mean Squared Error).
2.  **Presence Detector**:
    *   Binary Classifier (Sigmoid) determining if a human is in the RF field.
    *   Loss: BCE (Binary Cross Entropy).
3.  **Location Classifier**:
    *   Multi-class Classifier (Softmax) outputting probabilities for: `[Room A, Hallway, Empty]`.
    *   Loss: Cross Entropy.

---

## 4. Federated Inference Logic

The runtime system uses a hierarchical decision logic to ensure robustness and high accuracy.

### 4.1 Visual-Dominant Check
The system first polls the camera feed using MediaPipe Pose Estimation.
*   **Condition**: If the computer vision model detects a person with high confidence ($> 0.8$).
*   **State**: **HALLWAY**.
*   **Rationale**: Visual data is ground truth. If the camera sees the person, they are undeniably in the camera's field of view (Hallway).

### 4.2 RF-Based Sensing
When the subject exits the camera's view, the system relies on the Neural Network's analysis of the Wi-Fi environment.
*   **Condition**: Camera = Null.
*   **State**: **ROOM A**.
*   **Mechanism**: The recurrent neural network continues to track the subject's impact on the RF multipath environment in Room A. Even without line-of-sight, the disturbances in the CSI amplitude and phase allow the system to maintain presence awareness.

---

## 5. Training Strategy

### 5.1 Data Collection
Data is collected in "sessions" to capture diverse conditions:
*   **Room A Sessions**: Subject performs randomized activities (walking, sitting, standing) in Room A.
*   **Hallway Sessions**: Subject transitions through the hallway.
*   **Empty Sessions**: Recorded with **zero human presence**. This is critical for the model to learn the "noise floor" (baseline RF environment).

### 5.2 Global Scaling
To preserve the relative signal strength differences between Node A (Room A) and Node B, we rely on **Global Scaling**:
*   Instead of normalizing per-sample (which would erase relative magnitude), we calculate the **Median** and **IQR (Inter-Quartile Range)** of the *entire training dataset*.
*   These global statistics are frozen and applied to live inference data, ensuring that a "loud" signal remains loud relative to the learned baseline.
