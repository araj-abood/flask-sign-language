import mediapipe as mp
import cv2
import numpy as np
from typing import Dict, List, Tuple
import pickle

class HandSignDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        # Dictionary to store sign data
        self.sign_data: Dict[str, List[List[float]]] = {}
        
    def extract_landmarks(self, frame) -> List[float]:
        """
        Extract hand landmarks from a single frame
        Returns flattened list of landmarks
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return []
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Convert landmarks to flat list
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
            
        return landmarks
    
    def add_sign(self, sign_name: str, frame):
        """
        Add a new training example for a sign
        """
        landmarks = self.extract_landmarks(frame)
        if not landmarks:
            raise ValueError("No hand detected in frame")
            
        if sign_name not in self.sign_data:
            self.sign_data[sign_name] = []
            
        self.sign_data[sign_name].append(landmarks)
        
    def save_model(self, filename: str):
        """
        Save the collected sign data
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.sign_data, f)
            
    def load_model(self, filename: str):
        """
        Load previously saved sign data
        """
        with open(filename, 'rb') as f:
            self.sign_data = pickle.load(f)
            
    def predict_sign(self, frame) -> Tuple[str, float]:
        """
        Predict the sign in a given frame
        Returns (predicted_sign, confidence)
        """
        current_landmarks = self.extract_landmarks(frame)
        if not current_landmarks:
            return ("no_hand", 0.0)
            
        best_match = None
        best_score = float('inf')
        
        # Compare with all known signs
        for sign_name, examples in self.sign_data.items():
            for example in examples:
                # Calculate Euclidean distance
                distance = np.linalg.norm(np.array(current_landmarks) - np.array(example))
                if distance < best_score:
                    best_score = distance
                    best_match = sign_name
                    
        # Convert distance to confidence (inverse relationship)
        confidence = 1 / (1 + best_score)
        return (best_match, confidence)

# Example usage for data collection:
def collect_training_data():
    detector = HandSignDetector()
    cap = cv2.VideoCapture(0)
    
    signs_to_collect = ["hello", "thank_you", "goodbye"]  # Add your signs here
    samples_per_sign = 5
    
    for sign in signs_to_collect:
        samples_collected = 0
        print(f"Collecting samples for sign: {sign}")
        
        while samples_collected < samples_per_sign:
            ret, frame = cap.read()
            if not ret:
                continue
                
            cv2.imshow('Collection', frame)
            key = cv2.waitKey(1)
            
            if key == ord('c'):  # Press 'c' to capture
                try:
                    detector.add_sign(sign, frame)
                    samples_collected += 1
                    print(f"Collected sample {samples_collected}/{samples_per_sign}")
                except ValueError as e:
                    print(f"Error: {e}")
            elif key == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    detector.save_model('hand_signs.pkl')