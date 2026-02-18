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
    parser.add_argument("--rf_mode", default="esp32", choices=["mock", "linux", "scapy", "esp32"])
    parser.add_argument("--model", default="models/best.pth")
    parser.add_argument("--port", type=int, default=8888, help="UDP Port for ESP32 CSI")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="UDP IP to bind to")
    parser.add_argument("--node_a_ip", type=str, default="10.42.0.149", help="IP of Node A (Room A)")
    parser.add_argument("--node_b_ip", type=str, default="10.42.0.173", help="IP of Node B (Room B)")
    parser.add_argument("--swap_nodes", action="store_true", help="Swap Node A and Node B logic")
    parser.add_argument("--debug_stats", action="store_true", help="Print detailed stats to console")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Load Model (Dual Node = 320 Features)
    model = WifiPoseModel(input_features=320, output_points=33).to(device)
    if os.path.exists(args.model):
        try:
            model.load_state_dict(torch.load(args.model, map_location=device))
            logging.info("Model loaded.")
        except:
             logging.warning("Model mismatch. Using random weights.")
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
    nodes = {} 
    node_buffers = {
        'A': collections.deque(maxlen=50),
        'B': collections.deque(maxlen=50)
    }
    last_node_update = {'A': 0.0, 'B': 0.0}
    node_variance = {'A': 0.0, 'B': 0.0}
    
    # Smoothing
    smoothed_probs = np.array([0.25, 0.25, 0.25, 0.25]) # Init outside loop

    # LOGGING
    logging.info("System Initialized. Dual Room Logic Restored.")
    
    # --- RICH UI ---
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    from rich import box
    
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    
    def generate_ui(fps, nodes, node_buffers, loc_text, loc_probs, pres_prob, node_variance):
        # Header
        layout["header"].update(Panel(f"Dual-Node Wi-Fi Sensing | FPS: {fps:.1f}", style="bold white on blue"))
        
        # Left: Network Layout
        table_nodes = Table(title="ESP32 Nodes", box=box.ROUNDED)
        table_nodes.add_column("ID", style="cyan")
        table_nodes.add_column("IP Address", style="magenta")
        table_nodes.add_column("Packets", style="green")
        table_nodes.add_column("Status", style="yellow")
        table_nodes.add_column("Motion (Var)", style="red")
        
        current_time = time.time()
        for ip, nid in nodes.items():
            display_nid = nid
            if args.swap_nodes: display_nid = 'B' if nid == 'A' else 'A'

            buf_len = len(node_buffers.get(nid, []))
            last = current_time - last_node_update.get(nid, 0)
            status_str = f"{last:.1f}s ago" if last < 2.0 else "LOST"
            status_color = "green" if last < 1.0 else "red"
            motion = node_variance.get(nid, 0.0)
            
            table_nodes.add_row(display_nid, ip, str(buf_len), f"[{status_color}]{status_str}[/{status_color}]", f"{motion:.2f}")
            
        layout["left"].update(Panel(table_nodes, title="Network Status"))
        
        # Right: AI Prediction
        table_ai = Table(title="Inference Result", box=box.ROUNDED)
        table_ai.add_column("Class", style="white")
        table_ai.add_column("Probability", style="bold green")
        
        classes = ["Room A", "Room B", "Hallway", "Empty"]
        best_idx = np.argmax(loc_probs) if loc_probs is not None else -1
        
        for i, cls in enumerate(classes):
            prob = loc_probs[i] if loc_probs is not None else 0.0
            style = "bold green" if i == best_idx else "white"
            table_ai.add_row(cls, f"{prob:.1%}", style=style)

        table_ai.add_section()
        table_ai.add_row("Presence Confidence", f"{pres_prob:.2f}", style="cyan")

        layout["right"].update(Panel(table_ai, title=f"Pred: [bold]{loc_text}[/bold]"))
        layout["footer"].update(Panel("Press 'q' to quit.", style="dim"))
        
        return layout

    fps = 0
    frame_count = 0
    prev_time = time.time()

    with Live(layout, refresh_per_second=4) as live:
        try:
            while True:
                # --- RF Processing ---
                while not rf.get_queue().empty():
                    pkt = rf.get_queue().get()
                    src = pkt['source']
                    
                    if src not in nodes:
                        if args.node_a_ip and args.node_a_ip in src:
                             nodes[src] = 'A' if not args.swap_nodes else 'B'
                        elif args.node_b_ip and args.node_b_ip in src:
                             nodes[src] = 'B' if not args.swap_nodes else 'A'
                        else:
                            if not args.node_a_ip and not args.node_b_ip:
                                if len(nodes) == 0: nodes[src] = 'A'
                                elif len(nodes) == 1: nodes[src] = 'B'
                    
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

                # --- DUAl ROOM INFERENCE ---
                loc_text = "Init..."
                pres_prob = 0.0
                loc_probs = smoothed_probs
                
                # 1. VISUAL CHECK (Priority: Hallway)
                pose_results = pose_estimator.process_frame(frame)
                is_visible_cam = False

                if pose_results and pose_results.pose_landmarks:
                    is_visible_cam = True
                    loc_text = "Hallway"
                    pres_prob = 1.0
                    loc_probs = np.array([0.0, 0.0, 1.0, 0.0]) # Force Hallway
                    smoothed_probs = loc_probs # Reset smoothing
                    
                    pose_estimator.draw_landmarks(frame, pose_results)
                    cv2.putText(frame, "Visual Lock (Hallway)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                else:
                    # 2. WI-FI CHECK
                    if len(node_buffers['A']) == 50 and len(node_buffers['B']) == 50:
                        # Prepare Data
                        raw_a_full = np.array(node_buffers['A'])
                        raw_b_full = np.array(node_buffers['B'])
                        
                        raw_a_amp = raw_a_full[:, :64]
                        raw_b_amp = raw_b_full[:, :64]
                        
                        # Variance for UI
                        var_a = np.mean(np.var(raw_a_amp, axis=0))
                        var_b = np.mean(np.var(raw_b_amp, axis=0))
                        node_variance['A'] = var_a
                        node_variance['B'] = var_b

                        # Sanitize & Scale
                        clean_a_amp = CSISanitizer.sanitize_amplitude(raw_a_amp)
                        clean_b_amp = CSISanitizer.sanitize_amplitude(raw_b_amp) * 1.10 # Gain fix
                        
                        clean_a_ph = CSISanitizer.sanitize_phase(raw_a_full[:, 64:])
                        clean_b_ph = CSISanitizer.sanitize_phase(raw_b_full[:, 64:])
                        
                        log_a = np.log1p(clean_a_amp)
                        log_b = np.log1p(clean_b_amp)
                        diff = log_a - log_b
                        
                        flat_a = np.concatenate([log_a, clean_a_ph], axis=1)
                        flat_b = np.concatenate([log_b, clean_b_ph], axis=1)
                        merged = np.concatenate([flat_a, flat_b, diff], axis=1)
                        
                        scaled = scaler.transform(merged)
                        
                        t_merged = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            pred_pose, pred_pres, pred_loc = model(t_merged)
                        
                        pres_prob = pred_pres.item()
                        raw_probs = torch.softmax(pred_loc, dim=1).cpu().numpy()[0]
                        # classes = ["Room A", "Room B", "Hallway", "Empty"]
                        
                        # Suppress Hallway if camera empty
                        if not is_visible_cam:
                            raw_probs[2] = 0.0 
                            raw_probs = raw_probs / (np.sum(raw_probs) + 1e-6)
                        
                        # Smoothing
                        alpha = 0.3 # Smoother transitions
                        smoothed_probs = alpha * raw_probs + (1 - alpha) * smoothed_probs
                        
                        loc_probs = smoothed_probs
                        loc_idx = np.argmax(loc_probs)
                        
                        classes = ["Room A", "Room B", "Hallway", "Empty"]
                        
                        if pres_prob < 0.4:
                            loc_text = "Empty"
                        else:
                            loc_text = classes[loc_idx]
                            
                        # Draw CSI Pose
                        pose_coords = pred_pose.cpu().numpy()[0]
                        if pres_prob > 0.4 and loc_text != "Empty":
                             for i in range(0, len(pose_coords), 2):
                                x = int(pose_coords[i] * w)
                                y = int(pose_coords[i+1] * h)
                                if x > 0 and y > 0:
                                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                             cv2.putText(frame, f"CSI: {loc_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                             
                             if args.debug_stats and frame_count % 10 == 0:
                                 print(f"DEBUG: Probs={loc_probs}")

                    else:
                        loc_text = "Buffering..."

                # Update UI
                live.update(generate_ui(fps, nodes, node_buffers, loc_text, loc_probs, pres_prob, node_variance))
                
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
