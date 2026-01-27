import sys
import shutil
import platform
import subprocess

def check_command(cmd):
    path = shutil.which(cmd)
    if path:
        print(f"[OK] Found {cmd}: {path}")
        return True
    else:
        print(f"[ERR] Missing {cmd}")
        return False

def check_python_modules():
    modules = ["cv2", "mediapipe", "torch", "numpy", "pandas", "scapy"]
    all_ok = True

    for mod in modules:
        try:
            __import__(mod)
            print(f"[OK] Python module '{mod}' imported.")
        except ImportError:
            print(f"[ERR] Python module '{mod}' NOT found.")
            all_ok = False
        except Exception as e:
             print(f"[WARN] Error importing '{mod}': {e}")
    return all_ok

def main():
    print("=== Wi-Fi Human Detection Environment Check ===")
    
    os_type = platform.system()
    print(f"OS: {os_type}")
    
    if os_type == "Linux":
        # Check system tools for LinuxPollingCapture
        cmds = ["ping", "cat"] # Essentials
        for c in cmds:
             check_command(c)
             
        # Check RF tools (at least one should exist)
        rf_tools = ["iwconfig", "nmcli", "iw"]
        found_rf = False
        for tool in rf_tools:
            if check_command(tool):
                found_rf = True
        
        if not found_rf:
            print("[WARN] No standard Wi-Fi tool found (iwconfig/nmcli/iw). RF Capture might fail.")
            
        # Check for /proc/net/wireless
        try:
            with open("/proc/net/wireless", "r") as f:
                print("[OK] /proc/net/wireless is accessible.")
        except Exception as e:
            print(f"[WARN] /proc/net/wireless not readable: {e} (Scapy mode/iw fallback recommended)")
            
    # Check Python
    if not check_python_modules():

        print("\n[FAIL] Missing Python dependencies. Run 'pip install -r requirements.txt'")
        sys.exit(1)
        
    print("\n[SUCCESS] Environment looks good.")

if __name__ == "__main__":
    main()
