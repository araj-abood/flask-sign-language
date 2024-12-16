import arabic_reshaper
from bidi.algorithm import get_display
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from config import Config

class ArabicTextRenderer:
    def __init__(self):
        self.font = self._load_font()

    def _load_font(self):
        """Load the appropriate font file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        for font_path in Config.FONT_PATHS:
            full_path = os.path.join(current_dir, font_path)
            if os.path.exists(full_path):
                try:
                    return ImageFont.truetype(full_path, Config.FONT_SIZE)
                except Exception as e:
                    print(f"Font loading error: {e}")
        
        return ImageFont.load_default()

    def _process_text(self, text):
        """Process text for proper Arabic display"""
        text_parts = text.split(": ")
        if len(text_parts) == 2:
            label, arabic_text = text_parts
            reshaped_arabic = arabic_reshaper.reshape(arabic_text)
            bidi_arabic = get_display(reshaped_arabic)
            return f"{label}: {bidi_arabic}"
        else:
            reshaped_text = arabic_reshaper.reshape(text)
            return get_display(reshaped_text)

    def render_text(self, frame, text, position, color=Config.TEXT_COLOR, with_background=True):
        """Render text on frame with optional background"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        processed_text = self._process_text(text)
        
        if with_background:
            # Calculate text dimensions
            bbox = draw.textbbox(position, processed_text, font=self.font)
            padding = 10
            bg_dims = (
                position[0] - padding,
                position[1] - padding,
                bbox[2] + padding,
                bbox[3] + padding
            )
            
            # Draw background
            cv2.rectangle(frame, (bg_dims[0], bg_dims[1]), (bg_dims[2], bg_dims[3]), 
                         (0, 0, 0), cv2.FILLED)
            
            # Recreate PIL image after drawing rectangle
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
        
        # Draw text
        draw.text(position, processed_text, font=self.font, fill=(color[2], color[1], color[0]))
        
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
