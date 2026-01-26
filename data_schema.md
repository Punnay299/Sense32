# Data Schema

## Common Conventions
- **Timestamp**: All timestamps are in **Milliseconds (ms)**.
    - `ts_monotonic`: `time.monotonic() * 1000` (Used for relative alignment/drift correction)
    - `ts_wall`: `time.time() * 1000` (Used for human-readable logging)

## 1. RF Data (`rf_data.csv`)
Captured by `RFInterface` implementations.
| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp_monotonic_ms` | float | Host arrival time (local laptop clock) |
| `timestamp_device_ms` | float | Device capture time (if available from sniffer) |
| `source` | str | Source identifier (e.g., 'udp_sniffer', 'wlan0') |
| `rssi` | int | Received Signal Strength Indicator (dBm) |
| `mac_address` | str | Source MAC address of the packet |
| `ssid` | str | (Optional) SSID if beacon frame |
| `csi_amp` | json_array | (Optional) Amplitude list `[a1, a2, ...]` |
| `csi_phase` | json_array | (Optional) Phase list `[p1, p2, ...]` |

## 2. Camera Data (`camera_index.csv`)
Index mapping video frames to timestamps.
| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | int | 0-indexed frame number |
| `timestamp_monotonic_ms` | float | Time when frame was grabbed |
| `image_path` | str | Relative path to saved frame image |

## 3. Labels (`labels.csv`)
Ground truth or Pseudo-labels (from Vision).
| Column | Type | Description |
| :--- | :--- | :--- |
| `frame_index` | int | Matches `camera_index.csv` |
| `person_id` | int | Track ID (usually 0 for single person) |
| `pose_visible` | bool | Whether person is detected |
| `keypoints_flat` | json_array | `[x0, y0, c0, x1, y1, c1, ...]` (Normalized 0-1) |
| `center_x` | float | Center of mass X (Normalized 0-1) |
| `center_y` | float | Center of mass Y (Normalized 0-1) |
| `posture` | str | 'standing', 'walking', 'sitting', 'unknown' |
| `confidence` | float | Overall pose confidence (0-1) |

## 4. Sync Beacons (`beacons.csv`)
Log of synchronization bursts sent by `LabelSyncBeacon`.
| Column | Type | Description |
| :--- | :--- | :--- |
| `seq_id` | int | Sequence number of the beacon |
| `timestamp_monotonic_ms` | float | Time beacon was sent |
| `payload` | str | Content of the beacon packet |
