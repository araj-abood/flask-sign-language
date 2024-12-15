from neural_hand_detector import NeuralHandSignDetector
import cv2
import time

def collect_data(detector):
    cap = cv2.VideoCapture(0)
    
    # Define the signs you want to collect
    signs = ["hello", "thank_you", "goodbye"]  # Add your own signs
    samples_needed = 30  # Number of samples per sign
    
    print("\n=== Starting Data Collection ===")
    print("Instructions:")
    print("- Show your hand sign clearly to the camera")
    print("- Press 'c' to capture a sample")
    print("- Press 'n' to move to next sign")
    print("- Press 'q' to quit")
    
    current_sign_idx = 0
    samples_collected = 0
    
    while current_sign_idx < len(signs):
        current_sign = signs[current_sign_idx]
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Show instructions
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
                        time.sleep(1)
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
    cap = cv2.VideoCapture(0)
    
    print("\n=== Starting Real-time Prediction ===")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)  # Mirror
        
        try:
            sign, confidence = detector.predict_sign(frame)
            # Display prediction
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
    
    while True:
        print("\n=== Hand Sign Recognition System ===")
        print("1. Collect training data")
        print("2. Train model")
        print("3. Test predictions")
        print("4. Save model")
        print("5. Load model")
        print("6. Quit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            collect_data(detector)
        elif choice == '2':
            print("\nTraining model...")
            detector.train_model(epochs=100)
            print("Training completed!")
        elif choice == '3':
            test_predictions(detector)
        elif choice == '4':
            detector.save_model('neural_hand_signs.pth')
            print("\nModel saved to 'neural_hand_signs.pth'")
        elif choice == '5':
            try:
                detector.load_model('neural_hand_signs.pth')
                print("\nModel loaded successfully!")
            except FileNotFoundError:
                print("\nError: No saved model found!")
        elif choice == '6':
            break
        else:
            print("\nInvalid choice! Please try again.")

if __name__ == "__main__":
    main()