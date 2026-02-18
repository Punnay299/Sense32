# Wi-Fi Human Sensing: Technical Deep Dive

## 1. System Architecture

### Hardware Logic
*   **Dual-Node Sensing**: Uses two ESP32 receivers to create a "stereo" view of the RF environment.
*   **Active Ping**: Laptop sends 100Hz UDP pings. ESP32s capture the CSI of these pings and return them.
*   **Synchronized Batching**: The laptop acts as the clock, creating 50-frame windows (0.5s) containing synchronized data from Node A and Node B.

### Data Pipeline
1.  **Raw CSI**: Complex numbers (Real + Imaginary) for 64 subcarriers.
2.  **Sanitization**:
    *   **Phase**: Unwrapped and Linear Detrended to remove clock drift (SFO).
    *   **Amplitude**: Converted to magnitude.
3.  **Feature Stacking**:
    *   Input Tensor: `[Batch, Seq_Len=50, Features=256]`
    *   Features = `[Amp_A(64), Phase_A(64), Amp_B(64), Phase_B(64)]`.

---

## 2. Neural Network (CNN-LSTM)

To process this time-series spatial data, we use a hybrid architecture:

1.  **RF Encoder (1D CNN)**:
    *   Extracts spatial features from the 256 subcarriers/channels.
    *   Learns "shapes" of reflections (e.g., a specific interference pattern caused by a body).
    *   4 Layers of `Conv1d -> BatchNorm -> ReLU -> Dropout`.

2.  **Temporal Processor (LSTM)**:
    *   Takes the CNN features seqeunce and analyzes movement over time.
    *   Crucial for distinguishing "Moving Human" vs "Static Furniture".

3.  **Heads**:
    *   **Pose**: Regresses 33 (x,y) coordinates.
    *   **Presence**: Binary classification (0/1).
    *   **Location**: Multi-class (Room A, Room B, Hallway, Empty).

---

## 3. Current Challenge: Signal Strength Bias

### The Problem
The system exhibits a persistent bias towards identifying signals as **"Room B"**.
*   **Scenario 1 (User in Room B)**: Correctly detected as Room B.
*   **Scenario 2 (User in Room A)**: Incorrectly detected as Room B (or Hallway/Empty). "Room A" detection is extremely weak.

### Diagnostic Journey & Failed Solutions

#### Attempt 1: Physical Node Swapping
*   *Hypothesis*: Maybe we physically swapped the ESP32s or the code swapped the IPs.
*   *Action*: We verified MAC addresses and stuck strict IP mapping (`149`=Node A, `173`=Node B).
*   *Result*: **Disproved**. Hardware setup is correct.

#### Attempt 2: Instance Normalization (The "Flattening" Bug)
*   *Hypothesis*: The robust scaler was normalizing *per sample*. If Node A is loud and Node B is quiet, `InstanceNorm` makes them both Mean=0, Std=1. The model loses the "loudness" information.
*   *Action*: We switched to **Global Scaling**. We calculate the Median/IQR of the *entire dataset* and apply that fixed transform to every sample.
    *   *Theory*: This guarantees that if Node A is physically 2x louder, it enters the Neural Network as 2x larger values.
*   *Result*: **Failed**. Even with Global Scaling, the live accuracy did not improve.

#### Attempt 3: Feature Engineering (Variance)
*   *Hypothesis*: Maybe Mean Amplitude isn't the key. Maybe it's Variance (Motion).
*   *Investigation*: We ran `verify_dataset_mapping.py`.
    *   **Room A Data**: Node A Variance (0.498) > Node B Variance (0.495). (Difference: 0.003)
    *   **Room B Data**: Node B Variance (0.489) > Node A Variance (0.485).
*   *Finding*: The difference in physical signal variance is **extremely small** after scaling. The Signal-to-Noise Ratio (SNR) might be too low for the model to reliably distinguish the rooms based on variance alone.

### Conclusion (Feb 2026)
The software pipeline is theoretically correct (Global Scaling, Strict IPs, Synchronized Frames). However, the **physical signal contrast** between Room A and Room B events is vanishingly small in the processed data.

**Next Steps (Recommended but not yet implemented)**:
1.  **Hardware Relocation**: Move Node A closer to the center of Room A to increase its dominance.
2.  **Power Boost**: Increase TX power on the ESP32 (if possible) or the Laptop.
3.  **Model Architecture**: Switch to a Transformer or Attention-based model that might attend to the subtle variance differences better than the LSTM.
