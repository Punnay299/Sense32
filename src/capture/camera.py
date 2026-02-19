import cv2
import time
import threading
import logging
import numpy as np

class CameraCapture:
    def __init__(self, device_id=0, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.timestamp_monotonic = 0
        self.timestamp_wall = 0
        self.cap = None
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Starts the camera capture thread."""
        logging.info(f"Opening camera {self.device_id}...")
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera {self.device_id}")
            raise RuntimeError(f"Could not open camera {self.device_id}")
        
        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Read the first frame
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed:
            self.timestamp_monotonic = time.monotonic() * 1000
            self.timestamp_wall = time.time() * 1000

        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        logging.info("Camera started.")
        return self

    def update(self):
        """Background thread function to continuously read frames."""
        while not self.stopped:
            if not self.cap.isOpened():
                break
                
            grabbed, frame = self.cap.read()
            ts_mono = time.monotonic() * 1000
            ts_wall = time.time() * 1000
            
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
                    self.timestamp_monotonic = ts_mono
                    self.timestamp_wall = ts_wall
            else:
                logging.warning("Camera failed to grab frame.")
                time.sleep(0.1)
                
            # Yield slightly to avoid 100% CPU usage if FPS is low
            time.sleep(0.005)

    def read(self):
        """Returns the latest frame and its timestamp."""
        with self.lock:
            if not self.grabbed:
                return None, 0, 0
            return self.frame.copy(), self.timestamp_monotonic, self.timestamp_wall

    def stop(self):
        """Stops the camera thread and releases resources."""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        logging.info("Camera stopped.")

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    cam = CameraCapture()
    try:
        cam.start()
        for i in range(50):
            frame, ts, _ = cam.read()
            if frame is not None:
                print(f"Frame {i}: {frame.shape} at {ts:.2f}")
            time.sleep(0.1)
    finally:
        cam.stop()

class MockCameraCapture:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.stopped = False
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.timestamp_monotonic = 0
        self.timestamp_wall = 0
        self.thread = None

    def start(self):
        logging.info("Mock Camera Started.")
        self.stopped = False
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            # Generate random noise frame
            self.frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
            self.timestamp_monotonic = time.monotonic() * 1000
            self.timestamp_wall = time.time() * 1000
            time.sleep(1.0 / self.fps)

    def read(self):
        return self.frame.copy(), self.timestamp_monotonic, self.timestamp_wall

    def stop(self):
        self.stopped = True
        if self.thread:
            self.thread.join()
        logging.info("Mock Camera Stopped.")
