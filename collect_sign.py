from hand_detector import HandSignDetector
import cv2
import time

def main():
    # Initialize detector
    detector = HandSignDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    # List of signs you want to collect
    signs = ["hello", "thank_you", "goodbye"]  # Add your own signs here
    samples_needed = 5  # Number of samples per sign
    
    print("Starting data collection...")
    print("Instructions:")
    print("- Press 'c' to capture a sample")
    print("- Press 'n' to move to next sign")
    print("- Press 'q' to quit")
    
    current_sign_idx = 0
    samples_collected = 0
    
    while current_sign_idx < len(signs):
        current_sign = signs[current_sign_idx]
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Add text to frame
        cv2.putText(frame, 
                    f"Collecting: {current_sign} ({samples_collected}/{samples_needed})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Hand Sign Collection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                detector.add_sign(current_sign, frame)
                samples_collected += 1
                print(f"Collected sample {samples_collected} for {current_sign}")
                
                if samples_collected >= samples_needed:
                    current_sign_idx += 1
                    samples_collected = 0
                    if current_sign_idx < len(signs):
                        print(f"\nMoving to next sign: {signs[current_sign_idx]}")
                        time.sleep(1)
            except ValueError as e:
                print(f"Error: {e}")
        elif key == ord('n'):
            current_sign_idx += 1
            samples_collected = 0
            if current_sign_idx < len(signs):
                print(f"\nMoving to next sign: {signs[current_sign_idx]}")
    
    # Save the collected data
    detector.save_model('hand_signs.pkl')
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection complete!")

if __name__ == "__main__":
    main()