from text_renderer import ArabicTextRenderer
from config import Config
import cv2

class PredictionDisplay:
    def __init__(self):
        self.text_renderer = ArabicTextRenderer()
        self.debug_counter = 0
        
    def draw_helper_rectangle(self, frame):
        """Draw helper rectangle for hand positioning"""
        height, width = frame.shape[:2]
        rect_size = min(width, height) // 2
        x = (width - rect_size) // 2
        y = (height - rect_size) // 2
        cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), 
                     Config.RECT_COLOR, 2)
                     
    def display_prediction(self, frame, sign, confidence):
        """Display prediction results on frame"""
        if sign != "no_hand" and confidence > Config.CONFIDENCE_THRESHOLD:
            frame = self.text_renderer.render_text(
                frame, f"Sign: {sign}", (10, 30), Config.TEXT_COLOR)
            frame = self.text_renderer.render_text(
                frame, f"Confidence: {confidence:.2f}", (10, 80), Config.TEXT_COLOR)
        else:
            frame = self.text_renderer.render_text(
                frame, "Waiting for hand sign...", (10, 30), Config.WARNING_COLOR)
            
        self.draw_helper_rectangle(frame)
        return frame
