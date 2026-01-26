import mediapipe as mp
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5):
        # Standard Import - Expecting working environment
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
            
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
