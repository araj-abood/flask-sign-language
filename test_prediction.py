from hand_detector import HandSignDetector
import cv2

def main():
    # Initialize detector and load saved model
    detector = HandSignDetector()
    detector.load_model('hand_signs.pkl')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting prediction...")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Get prediction
        sign, confidence = detector.predict_sign(frame)
        
        # Add prediction to frame
        cv2.putText(frame, 
                    f"Sign: {sign} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Hand Sign Prediction', frame)
        
        # Handle key presses
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()