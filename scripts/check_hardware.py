
import cv2
import os
import subprocess
import glob

def check_camera():
    print("\n--- Checking Cameras ---")
    available_cams = []
    # Check first 5 indices
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"[OK] Camera {i}: Opened successfully. Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
                available_cams.append(i)
            else:
                 print(f"[WARN] Camera {i}: Opened but failed to read frame.")
            cap.release()
        else:
            print(f"[FAIL] Camera {i}: Could not open.")
            
    if not available_cams:
        print("CRITICAL: No working cameras found!")
    else:
        print(f"Working Camera Indices: {available_cams}")
    return available_cams

def check_rf():
    print("\n--- Checking Wi-Fi Interfaces ---")
    try:
        ifaces = os.listdir('/sys/class/net/')
    except:
        print("Could not list /sys/class/net/")
        return []
        
    wireless = []
    for iface in ifaces:
        path = os.path.join('/sys/class/net/', iface, 'wireless')
        if os.path.exists(path) or iface.startswith('wl'):
            print(f"Found Wireless Interface: {iface}")
            # Check state
            try:
                state = open(f"/sys/class/net/{iface}/operstate").read().strip()
                print(f"  State: {state}")
                if state == "down":
                    print("  [WARN] Interface is DOWN.")
            except: 
                print("  State: Unknown")
            wireless.append(iface)
            
    if not wireless:
        print("CRITICAL: No wireless interfaces found.")
    return wireless

if __name__ == "__main__":
    check_camera()
    check_rf()
