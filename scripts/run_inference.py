import argparse
import os
import sys
import time
import collections
import logging
import cv2
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.capture.rf_interface import LinuxPollingCapture, MockRFCapture, ScapyRFCapture
from src.capture.camera import CameraCapture
from src.model.networks import WifiPoseModel
from src.vision.pose import PoseEstimator
from src.utils.normalization import AdaptiveScaler
from src.utils.csi_sanitizer import CSISanitizer

def main():
    parser = argparse.ArgumentParser()
    # Simplified Interface
    parser.add_argument("--rf_mode", default="esp32", choices=["mock", "linux", "scapy", "esp32"])
    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--port", type=int, default=8888, help="UDP Port for ESP32 CSI")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP IP to bind to")
    parser.add_argument("--node_a_ip", type=str, default="10.42.0.149", help="IP of Node A (Room A)")
    parser.add_argument("--node_b_ip", type=str, default="10.42.0.173", help="IP of Node B (Room B)")
    # Removed swap_nodes and debug_stats as they complicate things
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load Model (Dual Node = 320 Features)
    # We maintain the input shape/model weights to avoid retraining.
    model = WifiPoseModel(input_features=320, output_points=33).to(device)
    if os.path.exists(args.model):
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
            logging.info("Model loaded.")
        except:
             logging.warning("Model mismatch. Using random weights (dummy).")
    else:
        logging.warning("Model not found, using random weights.")
    model.eval()
    
    # 2. Load Scaler
    scaler = AdaptiveScaler()
    scaler_path = "models/scaler.json"
    if os.path.exists(scaler_path):
        scaler.load(scaler_path)
    else:
        logging.warning("Scaler not found! Data will be unscaled.")

    # 3. Init Sensors
    if args.rf_mode == "esp32":
        from src.capture.csi_capture import CSICapture
        rf = CSICapture(port=args.port, ip=args.ip)
    else:
        rf = MockRFCapture()

    cam = CameraCapture(device_id=0)
    pose_estimator = PoseEstimator()

    rf.start()
    cam.start()

    # 4. State Management
    # Keep A & B buffers for compatibility with trained model
    nodes = {} 
    node_buffers = {
        'A': collections.deque(maxlen=50),
        'B': collections.deque(maxlen=50)
    }
    last_node_update = {'A': 0.0, 'B': 0.0}

    # LOGGING
    logging.info("System Initialized. Logic: Camera -> Hallway, Else Wi-Fi Presence -> Room A.")
    
    # --- SIMPLIFIED UI ---
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    from rich import box
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )
    
    def generate_ui(fps, loc_text, pres_prob, node_status):
        # Header
        layout["header"].update(Panel(f"Simple Wi-Fi Sensing | FPS: {fps:.1f}", style="bold white on blue"))
        
        # Main Table
        table = Table(box=box.ROUNDED)
        table.add_column("Sensor", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Inference", style="green")

        # Network Status
        net_status = "Online" if node_status else "Waiting for Data..."
        active_cnt = len(nodes)
        table.add_row("Wi-Fi Network", net_status, f"Nodes Active: {active_cnt}")
        
        # Detection
        style = "bold green" if loc_text != "Empty" else "dim white"
        table.add_row("System Output", f"[{style}]{loc_text}[/{style}]", f"Confidence: {pres_prob:.1%}")
        
        layout["main"].update(Panel(table, title="Real-Time Detection"))
        return layout

    fps = 0
    frame_count = 0
    prev_time = time.time()

    with Live(layout, refresh_per_second=4) as live:
        try:
            while True:
                # --- RF Processing (Keep collecting A & B) ---
                while not rf.get_queue().empty():
                    pkt = rf.get_queue().get()
                    src = pkt['source']
                    
                    # Auto-Assign
                    if src not in nodes:
                        if args.node_a_ip and args.node_a_ip in src: nodes[src] = 'A'
                        elif args.node_b_ip and args.node_b_ip in src: nodes[src] = 'B'
                        else:
                            # Strict or nothing? Let's be permissive since logic is simplified
                            if 'A' not in nodes.values(): nodes[src] = 'A'
                            elif 'B' not in nodes.values(): nodes[src] = 'B'
                    
                    if src in nodes:
                        node_id = nodes[src]
                        last_node_update[node_id] = time.time()
                        
                        if 'csi_amp' in pkt and len(pkt['csi_amp']) >= 64:
                            raw_amp = np.array(pkt['csi_amp'][:64], dtype=np.float32)
                            raw_phase = np.zeros(64, dtype=np.float32)
                            if 'csi_phase' in pkt and len(pkt['csi_phase']) >= 64:
                                raw_phase = np.array(pkt['csi_phase'][:64], dtype=np.float32)
                            
                            combined_raw = np.concatenate([raw_amp, raw_phase])
                            node_buffers[node_id].append(combined_raw)

                # --- Frame Processing ---
                frame, _, _ = cam.read()
                if frame is None: continue
                h, w, _ = frame.shape

                # FPS
                frame_count += 1
                if time.time() - prev_time >= 1.0:
                    fps = frame_count / (time.time() - prev_time)
                    frame_count = 0
                    prev_time = time.time()

                # --- SIMPLIFIED INFERENCE ---
                loc_text = "Init..."
                pres_prob = 0.0
                
                # 1. VISUAL CHECK (Priority)
                pose_results = pose_estimator.process_frame(frame)
                
                if pose_results and pose_results.pose_landmarks:
                    # Camera Detects Person -> HALLWAY
                    loc_text = "Hallway"
                    pres_prob = 1.0
                    pose_estimator.draw_landmarks(frame, pose_results)
                    cv2.putText(frame, "HALLWAY (Visual)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                
                else:
                    # 2. WI-FI CHECK (Secondary)
                    # Check if we have enough data (Need both A and B for model shape)
                    if len(node_buffers['A']) == 50 and len(node_buffers['B']) == 50:
                        # Prepare Data (Same Pipeline)
                        raw_a_full = np.array(node_buffers['A'])
                        raw_b_full = np.array(node_buffers['B'])
                        
                        # Sanitize & Scale
                        clean_a_amp = CSISanitizer.sanitize_amplitude(raw_a_full[:, :64])
                        # Apply Gain Correction to B (proven needed in previous tasks)
                        clean_b_amp = CSISanitizer.sanitize_amplitude(raw_b_full[:, :64]) * 1.10
                        clean_a_ph = CSISanitizer.sanitize_phase(raw_a_full[:, 64:])
                        clean_b_ph = CSISanitizer.sanitize_phase(raw_b_full[:, 64:])
                        
                        # Log1p + Diff
                        log_a = np.log1p(clean_a_amp)
                        log_b = np.log1p(clean_b_amp)
                        diff = log_a - log_b
                        
                        # Stack: [A_Log, A_Ph, B_Log, B_Ph, Diff] = 320
                        flat_a = np.concatenate([log_a, clean_a_ph], axis=1)
                        flat_b = np.concatenate([log_b, clean_b_ph], axis=1)
                        merged = np.concatenate([flat_a, flat_b, diff], axis=1)
                        
                        scaled = scaler.transform(merged)
                        
                        t_merged = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            pred_pose, pred_pres, pred_loc = model(t_merged)
                        
                        pres_prob = pred_pres.item()
                        
                        # LOGIC: If Presence > 0.4 -> ROOM A
                        if pres_prob > 0.4:
                            loc_text = "Room A"
                            
                            # Draw Ghost Pose
                            pose_coords = pred_pose.cpu().numpy()[0]
                            for i in range(0, len(pose_coords), 2):
                                x = int(pose_coords[i] * w)
                                y = int(pose_coords[i+1] * h)
                                if x > 0 and y > 0:
                                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                            cv2.putText(frame, f"ROOM A (Pres: {pres_prob:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        else:
                            loc_text = "Empty"
                            cv2.putText(frame, "EMPTY", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                    else:
                        loc_text = "Buffering..."

                # Update UI
                node_active = (time.time() - last_node_update.get('A', 0) < 2.0)
                live.update(generate_ui(fps, loc_text, pres_prob, node_active))
                
                cv2.imshow("Wi-Fi Human Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        finally:
            rf.stop()
            cam.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
