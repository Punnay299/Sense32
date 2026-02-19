import os
import sys
import importlib.util
import logging

logging.basicConfig(level=logging.INFO)

def check_file(filepath):
    try:
        spec = importlib.util.spec_from_file_location("module.name", filepath)
        module = importlib.util.module_from_spec(spec)
        # Scan source for syntax errors without executing
        with open(filepath, 'r') as f:
            source = f.read()
        compile(source, filepath, 'exec')
        logging.info(f"[PASS] Syntax Check: {filepath}")
        return True
    except Exception as e:
        logging.error(f"[FAIL] Syntax/Import Error in {filepath}: {e}")
        return False

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logging.info(f"Checking integrity for project at {root_dir}")
    
    passed = 0
    failed = 0
    
    # Walk scripts and src
    for subdir in ['scripts', 'src']:
        path = os.path.join(root_dir, subdir)
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    if check_file(full_path):
                        passed += 1
                    else:
                        failed += 1
                        
    print("\n--- Summary ---")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
