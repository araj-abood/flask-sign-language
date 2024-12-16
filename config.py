class Config:
    """Central configuration for the application"""
    CONFIDENCE_THRESHOLD = 0.30
    FONT_SIZE = 32
    TEXT_COLOR = (0, 255, 0)
    WARNING_COLOR = (0, 255, 255)
    ERROR_COLOR = (0, 0, 255)
    RECT_COLOR = (255, 255, 255)
    SAMPLES_NEEDED = 30
    COUNTDOWN_SECONDS = 3
    DEBUG_INTERVAL = 30

    # Font configuration
    FONT_PATHS = [
        'fonts/IBMPlexSansArabic-Medium.ttf',
        'fonts/IBMPlexSansArabic-Regular.ttf',
        'fonts/IBMPlexSansArabic-Light.ttf'
    ]

    # Predefined signs
    SIGNS = ["وقف", "مرحبا", "شكرا"]