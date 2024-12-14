import cv2
import os
import time
from datetime import datetime
import json
import mediapipe as mp

class SignCollector:
    def __init__(self, base_dir='dataset'):
        self.base_dir = base_dir
        self.metadata = self._load_metadata()
        
        # Initialize MediaPipe Hands with support for two hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,  # Lowered threshold
            min_tracking_confidence=0.3    # Lowered threshold
        )
        
        # Create base directory
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        # Initialize camera
        self.cap = None
            
    def _initialize_camera(self):
        """Initialize camera with fallback options"""
        if self.cap is not None:
            self.cap.release()
            time.sleep(0.5)  # Add delay to ensure proper release
                
        for index in range(3):
            try:
                cap = cv2.VideoCapture(0)  # Try specifically camera 0 first
                if cap.isOpened():
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap = cap
                    
                    # Verify camera is working by reading a test frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        time.sleep(0.5)  # Give camera time to warm up
                        return True
                    else:
                        cap.release()
                
            except Exception as e:
                print(f"Failed to initialize camera at index {index}: {e}")
                
        print("Could not initialize any camera")
        return False

    def _load_metadata(self):
        """Load or create metadata file"""
        metadata_file = os.path.join(self.base_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {'signs': {}}

    def _save_metadata(self):
        """Save metadata to file"""
        metadata_file = os.path.join(self.base_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _release_camera(self):
        """Safely release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        time.sleep(0.5)  # Add delay to ensure proper release

    def process_frame(self, frame):
        """Process frame and detect hand landmarks"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(frame_rgb)
            
            # Initialize hand detection count
            hands_detected = 0
            
            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                hands_detected = len(results.multi_hand_landmarks)
                
                # Draw landmarks for each detected hand
                for hand_idx, (hand_landmarks, hand_handedness) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)):
                    try:
                        # Get handedness (left or right)
                        handedness = hand_handedness.classification[0].label
                        
                        # Choose different colors for left and right hands
                        color = (0, 255, 0) if handedness == "Right" else (255, 0, 0)
                        
                        # Draw the landmarks with default styling
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Add hand label
                        wrist = hand_landmarks.landmark[0]
                        frame_h, frame_w = frame.shape[:2]
                        wrist_x = int(wrist.x * frame_w)
                        wrist_y = int(wrist.y * frame_h)
                        cv2.putText(frame, handedness, (wrist_x, wrist_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
                        continue
                        
                return hands_detected
        except Exception as e:
            print(f"Error in process_frame: {e}")
            return 0

    def capture_sign(self, sign_name, num_frames=30):
        """Capture images for a sign"""
        # Initialize camera with multiple attempts
        attempts = 3
        while attempts > 0:
            if self._initialize_camera():
                break
            print(f"Retrying camera initialization, {attempts-1} attempts remaining")
            attempts -= 1
            time.sleep(1)
                
        if attempts == 0:
            print("Failed to initialize camera after multiple attempts")
            return 0

        sign_dir = os.path.join(self.base_dir, sign_name)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        print(f"\nCollecting images for sign: {sign_name}")
        print("Press SPACE to start capturing")
        print("Press Q to stop")

        frames_captured = 0
        capturing = False
        consecutive_failures = 0

        try:
            while True:
                try:
                    # Read frame
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        print(f"Failed to read frame (attempt {consecutive_failures})")
                        
                        if consecutive_failures >= 5:
                            print("Too many consecutive frame reading failures, reinitializing camera...")
                            if not self._initialize_camera():
                                print("Failed to reinitialize camera")
                                break
                            consecutive_failures = 0
                        continue

                    consecutive_failures = 0  # Reset failure counter on successful frame read
                    frame = cv2.flip(frame, 1)  # Mirror image
                    
                    # Process frame and detect hands
                    hands_detected = self.process_frame(frame)

                    # Draw guide rectangle
                    height, width = frame.shape[:2]
                    rect_size = min(width, height) // 2
                    x = (width - rect_size) // 2
                    y = (height - rect_size) // 2
                    cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), 
                                (0, 255, 0) if hands_detected > 0 else (0, 0, 255), 2)

                    # Add overlays
                    cv2.putText(frame, f"Sign: {sign_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Images: {frames_captured}/{num_frames}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Add hand detection status
                    status_text = f"Hands Detected: {hands_detected}"
                    status_color = (0, 255, 0) if hands_detected > 0 else (0, 0, 255)
                    cv2.putText(frame, status_text, (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

                    if capturing and hands_detected > 0:
                        # Crop and save frame
                        cropped_frame = frame[y:y + rect_size, x:x + rect_size]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(sign_dir, f"{sign_name}_{hands_detected}hands_{timestamp}.jpg")
                        cv2.imwrite(filename, cropped_frame)
                        frames_captured += 1

                        if frames_captured >= num_frames:
                            break

                    cv2.imshow('Sign Collection', frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        capturing = True

                except Exception as e:
                    print(f"Error in capture loop: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        break
                    continue

        except Exception as e:
            print(f"Error during capture: {e}")
        finally:
            # Clean up camera resources
            self._release_camera()

        # Update metadata
        self.metadata['signs'][sign_name] = {
            'captured_at': datetime.now().isoformat(),
            'num_images': frames_captured
        }
        self._save_metadata()

        return frames_captured

    def collect_dataset(self):
        """Main collection loop"""
        try:
            while True:
                print("\nSign Language Collection")
                sign = input("Enter the sign to capture (or 'quit' to exit): ").strip()
                
                if sign.lower() == 'quit':
                    break

                try:
                    num_frames = int(input("Number of frames to capture (default 30): "))
                except ValueError:
                    num_frames = 30

                frames_captured = self.capture_sign(sign, num_frames)
                print(f"Captured {frames_captured} frames for sign '{sign}'")
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        finally:
            self._release_camera()
            self.hands.close()

    def __del__(self):
        """Cleanup"""
        self._release_camera()
        if hasattr(self, 'hands'):
            self.hands.close()

if __name__ == "__main__":
    collector = SignCollector()
    collector.collect_dataset()