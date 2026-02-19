import mediapipe as mp
import cv2
import numpy as np

# Robust Mediapipe Import (Manual Injection Strategy)
import sys
import importlib.util
import os

mp_pose = None
mp_drawing = None

# 1. Try Standard
try:
    import mediapipe.solutions.pose
    import mediapipe.solutions.drawing_utils
    mp_pose = mediapipe.solutions.pose
    mp_drawing = mediapipe.solutions.drawing_utils
except (ImportError, AttributeError):
    pass

# 2. Try Python Submodule
if mp_pose is None:
    try:
        # Force import of the python module
        import mediapipe.python
        import mediapipe.python.solutions.pose
        import mediapipe.python.solutions.drawing_utils
        mp_pose = mediapipe.python.solutions.pose
        mp_drawing = mediapipe.python.solutions.drawing_utils
    except (ImportError, AttributeError):
        pass

# 3. Manual Path Injection (The Nuclear Option)
if mp_pose is None:
    try:
        import mediapipe
        if hasattr(mediapipe, '__file__'):
            # .../site-packages/mediapipe/__init__.py
            mp_path = os.path.dirname(mediapipe.__file__)
            python_path = os.path.join(mp_path, 'python')
            if os.path.exists(python_path):
                # Add mediapipe/python to sys.path temporarily to import solutions
                # But solutions expects 'mediapipe.python' prefix usually.
                # Let's try to find 'solutions' inside python/
                sol_path = os.path.join(python_path, 'solutions')
                if os.path.exists(sol_path):
                    # We found it on disk.
                    # Try to import 'mediapipe.python.solutions.pose' using importlib
                    spec = importlib.util.spec_from_file_location("mediapipe.python.solutions.pose", os.path.join(sol_path, "pose.py"))
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules["mediapipe.python.solutions.pose"] = mod
                        spec.loader.exec_module(mod)
                        mp_pose = mod
                        
                    spec_draw = importlib.util.spec_from_file_location("mediapipe.python.solutions.drawing_utils", os.path.join(sol_path, "drawing_utils.py"))
                    if spec_draw and spec_draw.loader:
                        mod_draw = importlib.util.module_from_spec(spec_draw)
                        sys.modules["mediapipe.python.solutions.drawing_utils"] = mod_draw
                        spec_draw.loader.exec_module(mod_draw)
                        mp_drawing = mod_draw
    except Exception as e:
        # Silently fail manual injection if it doesn't work, we'll raise error below
        pass

if mp_pose is None:
    raise ImportError("Could not import mediapipe solutions")

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
            
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )

    def process_frame(self, frame):
        """
        Processes a BGR frame and returns pose landmarks.
        :param frame: numpy array BGR
        :return: (landmarks_list, frame_with_drawing)
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        return results

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )
        return frame

    def get_keypoints_flat(self, results):
        """
        Returns a flat list of [x, y, z, visibility] for all 33 landmarks.
        """
        if not results.pose_landmarks:
            return None
            
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
        return keypoints

    def get_center_of_mass(self, results):
        """
        Returns (x, y) of the approximate center (Hips).
        """
        if not results.pose_landmarks:
            return None
        
        # Hip landmarks are 23 and 24
        left_hip = results.pose_landmarks.landmark[23]
        right_hip = results.pose_landmarks.landmark[24]
        
        cx = (left_hip.x + right_hip.x) / 2
        cy = (left_hip.y + right_hip.y) / 2
        return (cx, cy)
