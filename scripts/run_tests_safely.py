import os
import glob
import subprocess
import sys
import time

def main():
    test_files = glob.glob("tests/test_*.py")
    test_files.sort()
    
    print(f"Found {len(test_files)} tests.")
    
    passed = []
    failed = []
    timed_out = []
    
    for f in test_files:
        print(f"Running {f}...", end="", flush=True)
        start = time.time()
        try:
            # Run with coverage or just python
            cmd = [sys.executable, f]
            # Capture output
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            
            if result.returncode == 0:
                print(" \033[92mPASS\033[0m")
                passed.append(f)
            else:
                print(" \033[91mFAIL\033[0m")
                print(f"--- Output of {f} ---")
                print(result.stdout)
                print(result.stderr)
                print("---------------------")
                failed.append(f)
                
        except subprocess.TimeoutExpired:
            print(" \033[93mTIMEOUT\033[0m")
            timed_out.append(f)
        except Exception as e:
            print(f" \033[91mERROR: {e}\033[0m")
            failed.append(f)
            
    print("\n================ Results ================")
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Timed Out: {len(timed_out)}")
    
    if failed:
        print("\nFailed Tests:")
        for t in failed: print(f"- {t}")
        
    if timed_out:
        print("\nTimed Out Tests:")
        for t in timed_out: print(f"- {t}")

if __name__ == "__main__":
    main()
