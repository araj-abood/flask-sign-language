import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

class HandSignNet(nn.Module):
    def __init__(self, num_classes):
        super(HandSignNet, self).__init__()
        # 21 landmarks x 3 coordinates (x, y, z)
        input_size = 21 * 3
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class HandSignDataset(Dataset):
    def __init__(self, landmarks, labels):
        self.landmarks = landmarks
        self.labels = labels
    
    def __len__(self):
        return len(self.landmarks)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.landmarks[idx]), self.labels[idx]

class NeuralHandSignDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        self.sign_data: Dict[str, List[List[float]]] = {}
        self.model = None
        self.sign_to_idx = {}
        self.idx_to_sign = {}
        
    def extract_landmarks(self, frame) -> List[float]:
        """
        Extract hand landmarks from a single frame
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return []
        
        hand_landmarks = results.multi_hand_landmarks[0]
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
    
    def train_model(self, epochs=100):
        """
        Train the neural network on collected data
        """
        # Prepare data
        X = []
        y = []
        self.sign_to_idx = {sign: idx for idx, sign in enumerate(self.sign_data.keys())}
        self.idx_to_sign = {idx: sign for sign, idx in self.sign_to_idx.items()}
        
        for sign, landmarks_list in self.sign_data.items():
            X.extend(landmarks_list)
            y.extend([self.sign_to_idx[sign]] * len(landmarks_list))
        
        # Create dataset and dataloader
        dataset = HandSignDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        self.model = HandSignNet(len(self.sign_data))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def predict_sign(self, frame) -> Tuple[str, float]:
        """
        Predict the sign in a given frame using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        landmarks = self.extract_landmarks(frame)
        if not landmarks:
            return ("no_hand", 0.0)
        
        # Prepare input
        input_tensor = torch.FloatTensor(landmarks).unsqueeze(0)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs).item()
            confidence = probabilities[0][predicted_idx].item()
        
        return (self.idx_to_sign[predicted_idx], confidence)
    
    def save_model(self, filename: str):
        """
        Save the trained model and sign data
        """
        save_dict = {
            'model_state': self.model.state_dict() if self.model else None,
            'sign_data': self.sign_data,
            'sign_to_idx': self.sign_to_idx,
            'idx_to_sign': self.idx_to_sign
        }
        torch.save(save_dict, filename)
    
    def load_model(self, filename: str):
        """
        Load a trained model and sign data
        """
        save_dict = torch.load(filename)
        self.sign_data = save_dict['sign_data']
        self.sign_to_idx = save_dict['sign_to_idx']
        self.idx_to_sign = save_dict['idx_to_sign']
        
        if save_dict['model_state']:
            self.model = HandSignNet(len(self.sign_data))
            self.model.load_state_dict(save_dict['model_state'])

    def draw_hand_landmarks(self, frame):
        """Draw hand landmarks on frame and return processed frame"""
        if frame is None or frame.size == 0:
            return frame
            
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
            
            return frame
        except Exception as e:
            print(f"Error in draw_hand_landmarks: {e}")
            return frame


            