# The Definitive Technical Guide: Wi-Fi Object Detection

**Version**: 4.0 (Exhaustive Analysis)
**Date**: 2026-01-28
**Scope**: Complete Codebase Dissection (Line-by-Line)

---

# Table of Contents

1.  [Preface](#preface)
2.  [System Architecture Overview](#system-architecture-overview)
3.  [Detailed File Analysis](#detailed-file-analysis)
    *   3.1 `src/capture/rf_interface.py`
    *   3.2 `src/capture/camera.py`
    *   3.3 `src/capture/beacon.py`
    *   3.4 `src/vision/pose.py`
    *   3.5 `src/engine/fusion.py`
    *   3.6 `src/model/dataset.py`
    *   3.7 `src/model/network.py`
    *   3.8 `src/model/networks.py`
4.  [Script Logic Analysis](#script-logic-analysis)
    *   4.1 `scripts/collect_data.py`
    *   4.2 `scripts/process_all_data.py`
    *   4.3 `scripts/pose_extractor.py`
    *   4.4 `scripts/train_local.py`
    *   4.5 `scripts/run_inference.py`
    *   4.6 `scripts/finetune_local.py`
    *   4.7 `scripts/label_assist.py`
    *   4.8 `scripts/check_hardware.py`
    *   4.9 `scripts/debug_capture.py`
    *   4.10 `scripts/pretrain_rf.py`
    *   4.11 `scripts/test_scapy_layers.py`

---

# Preface

This document is intended to be the single source of truth for the Wi-Fi Object Detection codebase. It assumes the reader has access to the source code but requires a deep, almost "commentary-track" style explanation of why every line exists.

Every class, function, and significant variable is documented below.

---

# Detailed File Analysis

## 3.1 `src/capture/rf_interface.py`

**Location**: `src/capture/rf_interface.py`
**Type**: Driver Layer

This file contains the logic for interacting with the Network Interface Card (NIC). It uses the **Factory Pattern** (via the storage of different classes) and **Template Method Pattern** (via the `RFInterface` abstract base class).

### Imports
*   `threading`, `queue`: Essential for the asynchronous architecture. Network packets come in at random times; we must handle them without blocking the main application thread.
*   `scapy.all`: The primary library for packet manipulation in Python. Used for the `ScapyRFCapture` mode.

### Class `RFInterface(ABC)`
This is an **Abstract Base Class**. It cannot be instantiated directly. It enforces a specific structure on all subclasses.

*   **`__init__(self, callback=None)`**:
    *   `self.callback`: A function pointer. If provided, every packet is sent here *immediately*. Useful for debugging print statements.
    *   `self.packet_queue = queue.Queue()`: The core data structure. This is a thread-safe implementation of a First-In-First-Out (FIFO) buffer. Features like `collect_data.py` will poll this queue.
    *   `self.thread`: Holds the handle to the worker thread (so we can join/kill it later).

*   **`start(self)`**:
    *   Sets `self.running = True`.
    *   `threading.Thread(target=self._run)`: Spawns the thread.
    *   `daemon=True`: **CRITICAL**. This ensures that if the main python program crashes or exits, this thread dies automatically. If this were False, the script would hang forever waiting for the thread to close.

*   **`_emit(self, data)`**:
    *   **Normalization Logic**:
        ```python
        if 'timestamp_monotonic_ms' not in data:
            data['timestamp_monotonic_ms'] = time.monotonic() * 1000
        ```
        This ensures data integrity. Time is the single most important variable in sensor fusion. We define "Time 0" as the system boot time (Monotonic) because Wall Clock time (`time.time()`) can jump if the user changes the timezone or NTP updates.

### Class `MockRFCapture`
*   **Logic**:
    *   `raw_rssi = -60 + 20 * math.sin(...)`:
        *   -60 is the baseline dBm (average signal).
        *   20 is the amplitude. Signal swings from -40 (very close) to -80 (far/blocked).
        *   `elapsed * 2 * math.pi / 10.0`: The `10.0` is the period in seconds. The signal loops every 10 seconds.
    *   `time.sleep(0.05)`: Simulates a 20Hz refresh rate.

### Class `UDPRFCapture`
*   **Use Case**: Remote sensing.
*   **`_run`**:
    *   `recvfrom(4096)`: Standard UDP buffer size.
    *   `json.loads`: The "Protocol". We expect text-based JSON.
    *   **Error Handling**: The `try...except json.JSONDecodeError` block is vital. UDP is unreliable; we might receive half a packet, or a corrupted packet. We must just log warning and continue, not crash.

### Class `LinuxPollingCapture`
*   **`_detect_interface`**:
    *   Tries to read `/proc/net/wireless`.
    *   Reasoning: We don't want the user to have to hardcode "wlan0" or "wlp2s0". We parse the system file to find the first active interface.
*   **`_get_rssi`**:
    *   The heart of the "Standard" mode.
    *   Parsing logic: `lines.split()[3]` is standard for the `/proc/net/wireless` column layout.
    *   **Smoothing**:
        ```python
        self.rssi_history.append(val)
        if len > 5: pop(0)
        return sum / len
        ```
        *   **Why?**: RF noise is Gaussian. A single sample of -80 might actually be -60 with momentary interference. Averaging N samples reduces the variance by $\sqrt{N}$.
    *   **Fallbacks**: The code is defensive. If `/proc` fails (e.g., WSL2 or specialized driver), it tries `iwconfig` then `nmcli`.

### Class `ScapyRFCapture`
*   **`_packet_handler`**:
    *   `pkt.haslayer(RadioTap)`: We check for physical layer info.
    *   `mac = pkt[Dot11].addr2`: In 802.11 frames, `addr2` is the Transmitter Address (Source). We want to know *who* sent the packet.
*   **`_channel_hopper`**:
    *   **The Problem**: A Wi-Fi card is a radio tuner. It can only listen to 2.412 GHz (Ch1) OR 2.417 GHz (Ch2), etc.
    *   **The Solution**: We simulate a "Wideband" receiver by jumping between channels quickly.
    *   `iw dev ... set channel X`: A kernel command that retunes the radio.
    *   **Side Effect**: While tuned to Ch1, we miss everything on Ch6. This means our data is sparse.

---

## 3.2 `src/capture/camera.py`

**Location**: `src/capture/camera.py`
**Type**: Driver Layer

### Class `CameraCapture`
*   **`__init__`**:
    *   `self.lock = threading.Lock()`: **CRITICAL**. Python's Global Interpreter Lock (GIL) handles some safety, but when writing to a complex object like a numpy array (`self.frame`), we need explicit locking to prevent a race condition where the reader reads a half-updated frame.

*   **`update` (The Worker)**:
    *   Loop: `while not self.stopped`.
    *   `grabbed, frame = self.cap.read()`: This line blocks the CPU for ~30ms (at 30FPS).
    *   `ts_mono = time.monotonic()`: We timestamp the frame *immediately* after capture. This minimizes "jitter" (variable latency) from downstream processing.

*   **`read` (The Interface)**:
    *   `return self.frame.copy()`: **CRITICAL**. We return a *copy*, not a reference. If we returned a reference, the worker thread might overwrite the image *while the Model is processing it*, causing visual artifacts or crashes.

---

## 3.3 `src/capture/beacon.py`

**Location**: `src/capture/beacon.py`
**Type**: Utility / Driver

### Class `LabelSyncBeacon`
*   **`_run`**:
    *   `socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)`:
        *   This flag tells the OS kernel "Yes, I really want to send this packet to EVERYONE on the network." without it, `255.255.255.255` will fail.
    *   **Payload**: Uses JSON for simplicity.
        *   `seq`: Sequence number. Helps detect packet loss.
        *   `ts_sent`: The sender's time.

---

## 3.4 `src/vision/pose.py`

**Location**: `src/vision/pose.py`
**Type**: Vision Logic

### Class `PoseEstimator`
This wraps `mediapipe.solutions.pose`.

*   **`process_frame`**:
    *   `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`: OpenCV uses BGR (Blue-Green-Red) byte order by default. MediaPipe (and almost all other ML libraries) use RGB. If you forget this, the person will look blue, and the model detection rate drops significantly.

*   **`get_keypoints_flat`**:
    *   This function serializes the complex `NormalizedLandmarkList` object into a simple python list `[float]`.
    *   It drops the helper attributes (presence/smoothness) and keeps the raw `x, y, z, visibility`.
    *   **Output Size**: 33 points * 4 values = 132 floats.

---

## 3.5 `src/engine/fusion.py`

**Location**: `src/engine/fusion.py`
**Type**: Math / Logic

### Class `FusionEngine`
The sensor fusion logic.

*   **`__init__`**:
    *   `self.F` (Transition Matrix):
        ```python
        [1, 0, dt, 0],  # x = x + vx*dt
        [0, 1, 0, dt],  # y = y + vy*dt
        ```
        This encodes the "Law of Inertia". Objects in motion stay in motion.
    *   `self.H` (Measurement Matrix):
        ```python
        [1, 0, 0, 0],
        [0, 1, 0, 0]
        ```
        This creates the mapping: "I measured (x,y), which corresponds to the first two elements of the state vector (x,y,vx,vy)."

*   **`update`**:
    *   **Gating Logic**:
        *   The `if/elif` block determines *which* sensor is valid.
        *   This is a "Selection Fusion" approach rather than a weighted average fusion. We rely entirely on Vision if available, and entirely on RF if Vision fails. We do not try to average them because Vision is orders of magnitude more accurate when available.
    *   **Kalman Update**:
        *   `y = z - Hx`: The "Innovation" or "Residual". (Measurement - Prediction).
        *   `S = HPH' + R`: The "Innovation Covariance". Total uncertainty.
        *   `K = PH'S^-1`: The "Optimal Kalman Gain".
        *   `x = x + Ky`: The "Posteriori State Estimate".

---

## 3.6 `src/model/dataset.py`

**Location**: `src/model/dataset.py`
**Type**: Data Loading

### Class `RFDataset`
*   **`__getitem__`**:
    *   **Goal**: Return `(rf_tensor, pose_target, presence_flag)`.
    *   **The Windowing Problem**:
        *   A single RF packet tells us nothing.
        *   We need a *sequence* of packets (the "Feature").
        *   We chose `100` packets as the sequence length.
    *   **Logic**:
        *   Finds current timestamp $T$.
        *   Finds index of packets in range $[T-1s, T]$.
        *   **Padding**: If the list is short (e.g., 20 packets), we prepend zeros.
        *   **Truncation**: If the list is long (e.g., 200 packets), we take the last 100.
        *   This ensures the input tensor is always shape `(Batch, Channels, 100)`.

---

## 3.7 `src/model/network.py`

**Location**: `src/model/network.py`
**Type**: Deep Learning Model (Legacy)

### Class `RFModel`
A simpler model used for early experiments.
*   **Layers**:
    *   `Conv1d`: Extracts features.
    *   `Pool`: Reduces dimensionality.
    *   `GRU`: Gated Recurrent Unit. Simpler/Faster than LSTM.
*   **Heads**:
    *   `regressor`: Outputs `(x, y)` (only 2 values). This model assumed we only wanted center-of-mass, not full skeleton.

---

## 3.8 `src/model/networks.py`

**Location**: `src/model/networks.py`
**Type**: Deep Learning Model (Production)

### Class `WifiPoseModel`
The main model architecture. It composes three sub-modules.

1.  **`RFEncoder`**:
    *   **Structure**: 3-layer CNN followed by a 3-layer LSTM.
    *   **Dropout (`0.1`, `0.3`)**: Randomly zeros out neurons during training. This prevents overfitting, forcing the model to learn robust features rather than memorizing noise.
2.  **`PoseRegressor`**:
    *   **Structure**: Dense (Linear) layers. `256 -> 128 -> 66`.
    *   **Output**: 66 values corresponding to the flattened $(x, y)$ of 33 keypoints.
3.  **`PresenceDetector`**:
    *   **Structure**: `256 -> 64 -> 1`.
    *   **Output**: Sigmoid. This is a binary classification head.

---

# 4. Script Logic Analysis

## 4.1 `scripts/collect_data.py`

**Location**: `scripts/collect_data.py`
**Type**: Application Entry Point (Recording)

**Flow Analysis**:
1.  **Arguments**: Parses `--name`, `--duration`, `--rf_mode`.
2.  **Setup**:
    *   Calls `os.makedirs` generated with `datetime.now`. This prevents overwriting old data.
    *   Initializes `CameraCapture` and `RFInterface`.
3.  **Synchronization Beacon**:
    *   `if args.beacon`: Starts the UDP broadcaster.
    *   Logic: Defines `beacon_log` callback to save sent beacons to CSV.
4.  **Main Loop**:
    *   `time.sleep(1/30.0)`: Caps the loop rate.
    *   **RF**: Drains the queue. `pkt = rf.get_queue().get()`. Writes to CSV.
    *   **Camera**: `cam.read()`.
        *   **Check**: `if ts_mono > last_frame_ts`. This is crucial. Since the loop might run faster than the camera (or vice versa), we ensure we only save *unique* frames to the disk.
5.  **Clean Exit**:
    *   `finally`: Ensures `cam.stop()` and `rf.stop()` are called even if the user hits `Ctrl+C`. This releases the resources (camera handle) properly.

---

## 4.2 `scripts/process_all_data.py`

**Location**: `scripts/process_all_data.py`
**Type**: Batch Processing

**Flow Analysis**:
1.  **Globbing**: `glob.glob("data/session_*")`. Finds all folders.
2.  **Subprocess Call**:
    *   Instead of importing `pose_extractor.py`, it runs it as a subprocess: `subprocess.check_call([...])`.
    *   **Why?**: MediaPipe has a known memory leak in some versions when processing massive amounts of video in a single python process. By spawning a fresh process for each video, we ensure the OS reclaims memory after each session.

---

## 4.3 `scripts/pose_extractor.py`

**Location**: `scripts/pose_extractor.py`
**Type**: Preprocessing (Labels)

**Flow Analysis**:
1.  **Input**: Takes a session path.
2.  **Video Loop**:
    *   Opens `video.mp4`.
    *   Iterates through *every* frame.
3.  **Inference**:
    *   Calls `pose_estimator.process_frame(frame)`.
4.  **Extraction**:
    *   If `results.pose_landmarks` exists:
        *   Sets `visible = True`.
        *   Extracts JSON landmarks.
        *   computes Center of Mass.
    *   Else:
        *   Sets `visible = False`.
        *   Sets Center to `-1, -1`.
5.  **Output**: Writes `labels.csv`.

---

## 4.4 `scripts/train_local.py`

**Location**: `scripts/train_local.py`
**Type**: Training

**Flow Analysis**:
1.  **Datasets**:
    *   Loads multiple sessions.
    *   Concatenates them: `torch.utils.data.ConcatDataset`.
    *   Splits: `random_split` (80% Train, 20% Val).
2.  **Model Setup**:
    *   Checks for CUDA. `device = 'cuda' if ...`.
    *   Defines `MSELoss` and `BCELoss`.
3.  **Training Loop**:
    *   **Forward Pass**: `pred = model(rf)`.
    *   **Loss Calculation**:
        *   `loss_pres = criterion(pres, gt)`.
        *   `loss_pose = criterion(pose, gt) * mask`.
        *   **Mask Logic**: `mask` is derived from `presence_gt`. If the person is not there, `mask=0`, so `loss_pose` becomes 0. This teaches the model: "I don't care what coordinates you guess when nobody is there, but you better guess 'Presence=0'".
    *   **Backward Pass**: `loss.backward()`, `optimizer.step()`.
4.  **Scheduler**: `ReduceLROnPlateau`. If validation loss stops improving, lower the learning rate.

---

## 4.5 `scripts/run_inference.py`

**Location**: `scripts/run_inference.py`
**Type**: Application (Runtime)

**Flow Analysis**:
1.  **Initialization**:
    *   Loads model weights: `model.load_state_dict(...)`.
    *   Starts hardware.
2.  **Buffer Management**:
    *   `rf_buffer = collections.deque(maxlen=50)`.
    *   This is a sliding window. As new packets arrive, old ones pop off.
3.  **Coordination API**:
    *   Unlike `collect_data.py` (record everything), this script must *wait* until it has enough data.
    *   `if len(rf_buffer) == 50`: Only run inference if buffer is full (system warmed up).
4.  **Visualization**:
    *   Draws the "Ground Truth" (MediaPipe) in Green.
    *   Draws the "Prediction" (RF) in Red.
    *   Calculates presence probability. If `< 0.5`, shows "RF: No Person".

---

## 4.6 `scripts/finetune_local.py`

**Location**: `scripts/finetune_local.py`
**Type**: Training (Alternative)

**Analysis**:
This script is similar to `train_local.py` but simplified for "Fine-Tuning".
*   It assumes you have a pre-trained model.
*   It focuses on adapting a generic model to a specific room.
*   **Key Difference**: Often uses a smaller Learning Rate (`lr`) to gently nudge the weights without destroying previous knowledge.

---

## 4.7 `scripts/label_assist.py`

**Location**: `scripts/label_assist.py`
**Type**: Utility (GUI)

**Analysis**:
A `cv2` based GUI for correcting dataset errors.
*   **Scenario**: The user walked in the dark. MediaPipe failed. The `labels.csv` says "Not Visible". But the user *knows* they were there.
*   **Logic**:
    *   Reads `video.mp4` and `labels.csv`.
    *   `cv2.setMouseCallback`: Listens for clicks.
    *   `update`:
        *   Mouse Click $\rightarrow$ Updates `labels[current_idx]['center_x']`.
        *   Key 'V' $\rightarrow$ Toggles Visibility.
        *   Key 'S' $\rightarrow$ Overwrites `labels.csv`.

---

## 4.8 `scripts/check_hardware.py`

**Location**: `scripts/check_hardware.py`
**Type**: Diagnostic

**Analysis**:
*   `check_camera()`:
    *   Brute forces indices 0-4. `cv2.VideoCapture(i)`.
    *   Checks `cap.isOpened()` AND `cap.read()`. Some cameras open but return black frames; this checks for actual data.
*   `check_rf()`:
    *   Lists `/sys/class/net`.
    *   Checks the `operstate` file to see if the interface is `up`, `down`, or `dormant`.

---

## 4.9 `scripts/debug_capture.py`

**Location**: `scripts/debug_capture.py`
**Type**: Diagnostic

**Analysis**:
*   Specific check for **Root Privileges** (`os.getuid()`). Scapy generally requires root to open raw sockets.
*   Runs a short 5-second sniff.
*   Prints packet summaries. This confirms that the driver is actually passing data to Python, ruling out kernel driver issues.

---

## 4.10 `scripts/pretrain_rf.py`

**Location**: `scripts/pretrain_rf.py`
**Type**: Placeholder / Template

**Analysis**:
*   Intended for use with public datasets (e.g., Widar3.0, CSI-Tool).
*   Defines the structure for a `PublicRFDataset` that would load `.mat` files (Matlab format common in research) instead of our `.csv` format.
*   Code:
    ```python
    for param in model.regressor.parameters():
        param.requires_grad = False
    ```
    This shows how to **Freeze** layers. In pretraining, we might only want to train the Encoder (Feature Extractor) and not the specific coordinate regressor.

---

## 4.11 `scripts/test_scapy_layers.py`

**Location**: `scripts/test_scapy_layers.py`
**Type**: Diagnostic

**Analysis**:
*   Validates the `RadioTap` header parsing.
*   Different drivers (Atheros vs Intel vs Realtek) provide different Radiotap formats. This script simply dumps what it sees so the developer can adjust `rf_interface.py` regex/parsing logic accordingly.

---

**(End of Detailed Analysis. Total coverage of all provided files.)**
