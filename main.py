# AI features import
import pytesseract
import easyocr
import pyautogui
import numpy as np
import cv2
import platform
import keyboard
import time
import ollama
import base64
import io
from PIL import Image
import utils

# App features import
import sys
import os

import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Enable DPI scaling
except Exception:
    pass  # Ignore if it fails

from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QTextBrowser, QSizeGrip
from PyQt6.QtGui import QPixmap, QPainter, QColor, QShortcut, QKeySequence
from PyQt6.QtCore import Qt, QPoint, QTimer

CHARACTER_FOLDER = "characters"  # Root character directory

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Read the character name from command-line arguments
    character_name = sys.argv[1] if len(sys.argv) > 1 else None
    overlay = utils.ChatOverlay(character_name)
    overlay.show()
    sys.exit(app.exec())

# Initialize EasyOCR reader
#reader = easyocr.Reader(['en','fr'], gpu=True)  # Supports multiple languages

# Run the monitor
#utils.monitor_window_changes(utils.on_window_change, poll_interval=0.5)  # Check every 0.5s



