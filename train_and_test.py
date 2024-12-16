from neural_hand_detector import NeuralHandSignDetector
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import numpy as np
import os
import arabic_reshaper
from bidi.algorithm import get_display


def get_font_path():
    """Get the path to the IBM Plex Sans Arabic font file"""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the primary font to use
    primary_font = os.path.join(current_dir, 'fonts', 'IBMPlexSansArabic-Medium.ttf')
    
    # Fallback fonts in order of preference
    font_paths = [
        primary_font,
        os.path.join(current_dir, 'fonts', 'IBMPlexSansArabic-Regular.ttf'),
        os.path.join(current_dir, 'fonts', 'IBMPlexSansArabic-Light.ttf')
    ]
    
    # Try each font path
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
            
    # If no font is found, return None
    return None
def put_arabic_text_with_background(frame, text, position, font_size=1, color=(0, 255, 0)):
    """Add Arabic text with a dark background for better readability"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load font
    font_path = get_font_path()
    try:
        if font_path:
            font = ImageFont.truetype(font_path, int(font_size * 32))
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()
    
    # Handle bidirectional text properly
    text_parts = text.split(": ")
    if len(text_parts) == 2:
        label, arabic_text = text_parts
        # Reshape and apply bidirectional algorithm only to Arabic part
        reshaped_arabic = arabic_reshaper.reshape(arabic_text)
        bidi_arabic = get_display(reshaped_arabic)
        text = f"{label}: {bidi_arabic}"
    else:
        # If it's only Arabic text
        reshaped_text = arabic_reshaper.reshape(text)
        text = get_display(reshaped_text)
    
    # Get text size
    bbox = draw.textbbox(position, text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate background rectangle dimensions
    padding = 10
    bg_x1 = position[0] - padding
    bg_y1 = position[1] - padding
    bg_x2 = position[0] + text_width + padding
    bg_y2 = position[1] + text_height + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), cv2.FILLED)
    
    # Draw text
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def put_arabic_text(frame, text, position, font_size=1, color=(0, 255, 0)):
    """Helper function to display Arabic text correctly"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load font
    font_path = get_font_path()
    try:
        if font_path:
            font = ImageFont.truetype(font_path, int(font_size * 32))
        else:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()
    
    # Handle bidirectional text properly
    text_parts = text.split(": ")
    if len(text_parts) == 2:
        label, arabic_text = text_parts
        # Reshape and apply bidirectional algorithm only to Arabic part
        reshaped_arabic = arabic_reshaper.reshape(arabic_text)
        bidi_arabic = get_display(reshaped_arabic)
        text = f"{label}: {bidi_arabic}"
    else:
        # If it's only Arabic text
        reshaped_text = arabic_reshaper.reshape(text)
        text = get_display(reshaped_text)
    
    # Draw the text
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
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
        
        # Display sign in Arabic with English label
        frame = put_arabic_text(frame, f"Sign: {current_sign}", (10, 30))
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
    print("Show your hand to the camera to begin detection")
    
    # Check if model exists
    if not hasattr(detector, 'model') or detector.model is None:
        print("Error: Model is not ready. Please train the model or load a pre-trained model first.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Counter to reduce debug spam
    debug_counter = 0
    CONFIDENCE_THRESHOLD = 0.30
    
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
            
            if sign != "no_hand" and confidence > CONFIDENCE_THRESHOLD:
                # Display sign text (top line)
                frame = put_arabic_text_with_background(
                    frame, 
                    f"Sign: {sign}", 
                    (10, 30),
                    font_size=1,
                    color=(0, 255, 0)
                )
                
                # Display confidence (bottom line)
                frame = put_arabic_text_with_background(
                    frame, 
                    f"Confidence: {confidence:.2f}", 
                    (10, 80),
                    color=(0, 255, 0)
                )
            else:
                frame = put_arabic_text_with_background(
                    frame,
                    "Waiting for hand sign...",
                    (10, 30),
                    color=(0, 255, 255)
                )
                           
        except Exception as e:
            print(f"Prediction error: {e}")
            frame = put_arabic_text_with_background(
                frame,
                "Prediction Error",
                (10, 30),
                color=(0, 0, 255)
            )
            import traceback
            print("Full error details:")
            print(traceback.format_exc())
        
        # Add helper rectangle
        frame_height, frame_width = frame.shape[:2]
        rect_size = min(frame_width, frame_height) // 2
        x = (frame_width - rect_size) // 2
        y = (frame_height - rect_size) // 2
        cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), 
                     (255, 255, 255), 2)
        
        cv2.imshow('Prediction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    detector = NeuralHandSignDetector()
    # Define your Arabic signs
    signs = ["وقف", "مرحبا", "شكرا"]  # Add more signs as needed
    samples_needed = 30
    
    while True:
        print("\n=== Hand Sign Recognition System ===")
        print("1. Collect training data (with timer)")
        print("2. Train model")
        print("3. Test predictions")
        print("4. Save model")
        print("5. Load model")
        print("6. Quit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '1':
            collect_data_with_timer(detector, signs, samples_needed)
        elif choice == '2':
            print("\nTraining model...")
            try:
                detector.train_model(epochs=100)
                print("Training completed!")
            except Exception as e:
                print(f"Training error: {e}")
        elif choice == '3':
            # Add model status check
            if not hasattr(detector, 'model') or detector.model is None:
                print("\nError: No trained model found. Please train the model first or load a pre-trained model.")
                continue
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
            except Exception as e:
                print(f"\nError loading model: {e}")
        elif choice == '6':
            break
        else:
            print("\nInvalid choice! Please try again.")

            

if __name__ == "__main__":
    main()