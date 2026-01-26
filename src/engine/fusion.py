import numpy as np

class FusionEngine:
    def __init__(self):
        # State: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        
        # covariance
        self.P = np.eye(4) * 0.1
        
        # Process Noise
        self.Q = np.eye(4) * 0.01
        
        # Measurement Matrix (We measure x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # State Transition (Simple constant velocity model)
        # dt will be variable, so we construct F dynamically or assume fixed small dt
        self.dt = 0.033 # 30FPS
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement Noise
        # Vision is accurate (low noise)
        self.R_vision = np.eye(2) * 0.001
        # RF is noisy
        self.R_rf = np.eye(2) * 0.05 

    def update(self, vision_xy, vision_conf, rf_xy, rf_conf):
        """
        :param vision_xy: (x, y) normalized
        :param vision_conf: float
        :param rf_xy: (x, y) normalized
        :param rf_conf: float
        :return: (x, y) fused
        """
        # 1. Prediction Step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 2. Measurement Update
        z = None
        R = None
        
        if vision_conf > 0.6:
            z = np.array(vision_xy)
            R = self.R_vision
        elif rf_conf > 0.5:
            # Only use RF if it's confident enough and Vision failed
            z = np.array(rf_xy)
            R = self.R_rf
        
        if z is not None:
            # Kalman Gain
            # K = P H^T (H P H^T + R)^-1
            S = self.H @ self.P @ self.H.T + R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            y = z - (self.H @ self.state) # pre-fit residual
            self.state = self.state + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P
            
        return self.state[0], self.state[1]
