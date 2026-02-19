# Physical Setup Guide & Best Practices

## ⚠️ Safety Warning
*   **High Voltage**: Do NOT touch the ESP32 pins while plugged into mains via USB.
*   **Heat**: The ESP32 can get warm. Ensure it has airflow.

## 📡 Sensor Placement Strategy (The "Cross-Fire" Setup)

To achieve "Ultra-Robust" detection, we need to maximize the number of signal paths that cut through the area of interest.

### 1. The Central "Sun" (Primary Room / Hall)
*   **Device**: ESP32 #1 or Laptop
*   **Location**: Center of the Hall, elevated (e.g., on a table or shelf).
*   **Role**: Provides strong, direct LOS (Line-of-Sight) coverage of the main activity area.
*   **Height**: Chest level (~1.2m) is ideal for capturing body movement.

### 2. The "Cross-Cut" (Secondary Room)
*   **Device**: ESP32 #2
*   **Location**: Corner of the adjacent room, diagonal to the Main Hall device.
*   **Role**: Forces Wi-Fi signals to penetrate the wall.
*   **Why?**: When you move in the Hall, you disturb the reflections. But when you move *between* the rooms, you disturb the direct wall-penetrating path. This combination is key for location classification.

### 3. Laptop (Data Collector)
*   **Location**: Hall (near the Central ESP32 is fine, or anywhere stable).
*   **Role**: Runs the Neural Network and acts as the Wi-Fi Access Point (`CSI_NET`).
*   **Hybrid Mode**: The laptop *also* collects RSSI data, adding another "viewpoint" to the system.

## 🔌 Wiring & Power
*   **ESP32s**: Use any standard 5V USB charger (phone charger) or Power Bank.
*   **Cables**: Micro-USB data cables (ensure they are good quality).

## 🟢 Connection Verification
Before recording a full session:
1.  Run `sudo ./venv/bin/python scripts/debug_udp.py`.
2.  Plug in ESP32s one by one.
3.  Verify you see packets from valid IPs (e.g., `10.42.0.x`).
