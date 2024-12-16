# camera_utils.py

import cv2
import time
from text_renderer import ArabicTextRenderer

class DataCollector:
    def __init__(self):
        self.text_renderer = ArabicTextRenderer()
        
    def countdown_timer(self, cap, countdown_from=3):
        """Display countdown timer while keeping camera feed live"""
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)  # Mirror
            
            elapsed_time = time.time() - start_time
            remaining = countdown_from - int(elapsed_time)
            
            if remaining <= 0:
                return frame
                
            cv2.putText(frame, str(remaining), 
                        (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 4)
            
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

    def collect_data_with_timer(self, detector, signs, samples_needed, countdown_seconds=3):
        """Collect data with countdown timer before each capture"""
        cap = cv2.VideoCapture(0)
        
        print("\n=== Starting Data Collection (With Timer) ===")
        print("Instructions:")
        print("- Show your hand sign clearly to the camera")
        print("- Press 'c' to start the countdown and capture")
        print("- Press 'n' to move to next sign")
        print("- Press 'q' to quit")
        print(f"- You'll have {countdown_seconds} seconds to prepare after pressing 'c'")
        
        current_sign_idx = 0
        samples_collected = 0
        
        try:
            while current_sign_idx < len(signs):
                current_sign = signs[current_sign_idx]
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                
                # Display sign in Arabic with English label
                frame = self.text_renderer.render_text(
                    frame, 
                    f"Sign: {current_sign}", 
                    (10, 10)
                )
                cv2.putText(frame, f"Samples: {samples_collected}/{samples_needed}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    capture_frame = self.countdown_timer(cap, countdown_seconds)
                    if capture_frame is None:
                        break
                        
                    try:
                        detector.add_sign(current_sign, capture_frame)
                        samples_collected += 1
                        print(f"Collected sample {samples_collected}/{samples_needed} for {current_sign}")
                        
                        if samples_collected >= samples_needed:
                            current_sign_idx += 1
                            samples_collected = 0
                            if current_sign_idx < len(signs):
                                print(f"\nMoving to next sign: {signs[current_sign_idx]}")
                                
                    except ValueError as e:
                        print(f"Error: {e}")
                elif key == ord('n'):
                    current_sign_idx += 1
                    samples_collected = 0
                    if current_sign_idx < len(signs):
                        print(f"\nMoving to next sign: {signs[current_sign_idx]}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nData collection completed!")

class PredictionTester:
    def __init__(self):
        self.text_renderer = ArabicTextRenderer()
        self.confidence_threshold = 0.60
        
    def draw_helper_rectangle(self, frame):
        """Draw helper rectangle for hand positioning"""
        height, width = frame.shape[:2]
        rect_size = min(width, height) // 2
        x = (width - rect_size) // 2
        y = (height - rect_size) // 2
        cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), 
                     (255, 255, 255), 2)

    def test_predictions(self, detector):
        """Test the trained model with live predictions"""
        cap = cv2.VideoCapture(0)
        
        print("\n=== Starting Real-time Prediction ===")
        print("Press 'q' to quit")
        print("Show your hand to the camera to begin detection")
        
        debug_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    continue
                    
                frame = cv2.flip(frame, 1)
                
                try:
                    sign, confidence = detector.predict_sign(frame)
                    
                    if debug_counter % 30 == 0:
                        print(f"Predicted sign: {sign}, Confidence: {confidence:.2f}")
                    debug_counter += 1
                    
                    if sign != "no_hand" and confidence > self.confidence_threshold:
                        frame = self.text_renderer.render_text(
                            frame, 
                            f"Sign: {sign}", 
                            (10, 30),
                            color=(0, 255, 0)
                        )
                        
                        frame = self.text_renderer.render_text(
                            frame, 
                            f"Confidence: {confidence:.2f}", 
                            (10, 80),
                            color=(0, 255, 0)
                        )
                    else:
                        frame = self.text_renderer.render_text(
                            frame,
                            "Waiting for hand sign...",
                            (10, 30),
                            color=(0, 255, 255)
                        )
                                   
                except Exception as e:
                    print(f"Prediction error: {e}")
                    frame = self.text_renderer.render_text(
                        frame,
                        "Prediction Error",
                        (10, 30),
                        color=(0, 0, 255)
                    )
                    import traceback
                    print("Full error details:")
                    print(traceback.format_exc())
                
                self.draw_helper_rectangle(frame)
                cv2.imshow('Prediction', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()