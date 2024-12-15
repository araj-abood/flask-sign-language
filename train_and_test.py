from neural_hand_detector import NeuralHandSignDetector
import cv2
import time

def countdown_timer(cap, countdown_from=3):
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

def collect_data_with_timer(detector, signs, samples_needed, countdown_seconds=3):
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
    
    while current_sign_idx < len(signs):
        current_sign = signs[current_sign_idx]
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Sign: {current_sign}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{samples_needed}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            capture_frame = countdown_timer(cap, countdown_seconds)
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
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection completed!")

def collect_data_quick(detector, signs, samples_needed):
    """Collect data without countdown timer for faster collection"""
    cap = cv2.VideoCapture(0)
    
    print("\n=== Starting Quick Data Collection ===")
    print("Instructions:")
    print("- Press 'c' to capture immediately")
    print("- Press 'n' to move to next sign")
    print("- Press 'q' to quit")
    
    current_sign_idx = 0
    samples_collected = 0
    
    while current_sign_idx < len(signs):
        current_sign = signs[current_sign_idx]
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        
        cv2.putText(frame, f"Sign: {current_sign}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {samples_collected}/{samples_needed}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            try:
                detector.add_sign(current_sign, frame)
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
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection completed!")

def test_predictions(detector):
    """Test the trained model with live predictions"""
    cap = cv2.VideoCapture(0)
    
    print("\n=== Starting Real-time Prediction ===")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        
        try:
            sign, confidence = detector.predict_sign(frame)
            cv2.putText(frame, f"Sign: {sign}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Prediction error: {e}")
        
        cv2.imshow('Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    detector = NeuralHandSignDetector()
    signs = ["وقف"]  # Add your own signs
    samples_needed = 30  # Number of samples per sign
    
    while True:
        print("\n=== Hand Sign Recognition System ===")
        print("1. Collect training data (with timer)")
        print("2. Quick collect training data (no timer)")
        print("3. Train model")
        print("4. Test predictions")
        print("5. Save model")
        print("6. Load model")
        print("7. Quit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            collect_data_with_timer(detector, signs, samples_needed)
        elif choice == '2':
            collect_data_quick(detector, signs, samples_needed)
        elif choice == '3':
            print("\nTraining model...")
            detector.train_model(epochs=100)
            print("Training completed!")
        elif choice == '4':
            test_predictions(detector)
        elif choice == '5':
            detector.save_model('neural_hand_signs.pth')
            print("\nModel saved to 'neural_hand_signs.pth'")
        elif choice == '6':
            try:
                detector.load_model('neural_hand_signs.pth')
                print("\nModel loaded successfully!")
            except FileNotFoundError:
                print("\nError: No saved model found!")
        elif choice == '7':
            break
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()