import os

base_dir = "data"
if not os.path.exists(base_dir):
    print(f"Data directory {base_dir} does not exist.")
    exit()

print(f"Scanning {base_dir}...")
sessions = []
skipped = []

for name in os.listdir(base_dir):
    path = os.path.join(base_dir, name)
    
    # logic from train_local.py
    if "room_b" in name.lower() or "room2" in name.lower():
        skipped.append(name)
        continue
        
    if os.path.isdir(path) and "session_" in name:
        sessions.append(name)

print("\nIncluded Sessions:")
for s in sessions:
    print(f"  - {s}")

print("\nSkipped Sessions (Room B):")
for s in skipped:
    print(f"  - {s}")
