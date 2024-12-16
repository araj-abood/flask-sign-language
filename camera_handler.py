import cv2

class CameraHandler:
    def __init__(self):
        self.cap = None
        
    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
            
    def stop(self):
        """Stop camera capture"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            
    def read_frame(self, flip=True):
        """Read a frame from the camera"""
        if not self.cap:
            raise RuntimeError("Camera not started")
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read frame")
            
        return cv2.flip(frame, 1) if flip else frame
