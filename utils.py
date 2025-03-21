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
from PIL import Image, ImageShow

import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Enable DPI scaling
except Exception:
    pass  # Ignore if it fails

if platform.system() == "Windows":
    import win32gui
    import win32process
    import win32ui
    import win32con
    import psutil
    import os
    import tempfile

    def get_active_window_title():
        """Gets the title of the currently active window, ignoring the assistant itself."""
        hwnd = win32gui.GetForegroundWindow()
        window_name = win32gui.GetWindowText(hwnd)
        if window_name != "My Assistant" :
            return window_name
        return None
    
    def get_real_active_window():
        """Returns the actual active window title, ignoring the assistant."""
        hwnd = win32gui.GetForegroundWindow()
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        process_name = psutil.Process(pid).name()

        if win32gui.GetWindowText(hwnd) == "My Assistant":
            return None  # Ignore the assistant itself

        return hwnd, win32gui.GetWindowText(hwnd)

    def capture_active_window(base64_encode=True, debug=False):
        """
        Captures a screenshot, optionally encodes it in base64, and can display it for debugging.
        
        :param base64_encode: Whether to return the image as a base64-encoded string (default: True)
        :param debug: If True, displays the captured image (default: False)
        :return: Base64-encoded image bytes if base64_encode is True, else raw image bytes
        """
        # Capture the screenshot
        screenshot = pyautogui.screenshot()
        
       # Save the image to the specified path
        screenshot.save("capture.png", format='PNG')
        
        if debug:
            screenshot.show()
        
        return "capture.png"

    def get_active_window_rect():
        """Gets the active window's position (left, top, right, bottom)."""
        hwnd = win32gui.GetForegroundWindow()
        rect = win32gui.GetWindowRect(hwnd)
        return rect  # (left, top, right, bottom)

elif platform.system() == "Darwin":  # macOS
    print("Setup for Mac")
    from AppKit import NSWorkspace

    def get_active_window_title():
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        app_name = app.localizedName()

        if "python" in app_name.lower():  # Ignore the assistant itself
            return None
        return app_name
    
    def get_window_screenshot():
        """Captures a screenshot of the active window (cross-platform)."""
        return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
    
    import pygetwindow as gw

    def get_active_window_rect():
        """Gets the active window's position for macOS/Linux (if supported)."""
        win = gw.getActiveWindow()
        if win:
            return (win.left, win.top, win.right, win.bottom)
        return None

elif platform.system() == "Linux":
    print("Setup for Linux")
    import subprocess

    def get_active_window_title():
        """Gets the title of the currently active window on Linux, ignoring the assistant itself."""
        try:
            output = subprocess.check_output(["xdotool", "getactivewindow", "getwindowname"], text=True).strip()

            if "python" in output.lower():  # Ignore the assistant itself
                return None
            return output
        except Exception:
            return None
        
    def get_window_screenshot():
        """Captures a screenshot of the active window (cross-platform)."""
        return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2GRAY)
    
    import pygetwindow as gw

    def get_active_window_rect():
        """Gets the active window's position for macOS/Linux (if supported)."""
        win = gw.getActiveWindow()
        if win:
            return (win.left, win.top, win.right, win.bottom)
        return None
    
else:
    print("Platform not recognized or incompatible")
    def get_active_window_title():
        return None
    def get_window_screenshot():
        return None
    def get_active_window_rect():
        return None
    
def preprocess_image_for_llava(image):
    """Resize image for faster processing in LLaVA, using optimized settings."""
    max_size = (512, 512)  # Resize to fixed dimensions (faster than scaling dynamically)
    
    # Convert to RGB and resize
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize(max_size, Image.Resampling.LANCZOS)  

    # Save as JPEG for better performance
    img_bytes_io = io.BytesIO()
    pil_image.save(img_bytes_io, format="JPEG", quality=85)  # Faster than PNG

    return img_bytes_io.getvalue()

def preprocess_image_for_ocr(img, debug=False):
    """Enhances image contrast and removes noise for better OCR."""
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image to improve OCR accuracy
    scale_percent = 200  # Scale up for better recognition
    width = int(gray_img.shape[1] * scale_percent / 100)
    height = int(gray_img.shape[0] * scale_percent / 100)
    scaled_img = cv2.resize(gray_img, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian Blur to reduce noise
    blurred_img = cv2.GaussianBlur(scaled_img, (3, 3), 0)

    # Apply adaptive thresholding for better text contrast
    filtered_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

    # Sharpen the image using kernel filtering
    #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #filtered_img = cv2.filter2D(filtered_img, -1, kernel)

    if debug:
        cv2.imshow("Original Screenshot", img)
        cv2.imshow("Grayscale", gray_img)
        cv2.imshow("Preprocessed Image", filtered_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered_img

def extract_text_from_window(char_limit=500, debug=False):
    """Extracts text from the active screen region using OCR with advanced preprocessing."""

    screenshot = np.array(pyautogui.screenshot())  # Take a screenshot
    processed_img = preprocess_image_for_ocr(screenshot, debug=debug)   # Preprocess for better OCR
    
    # Use advanced Tesseract settings
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? "

    # Run OCR with better configuration
    text = pytesseract.image_to_string(processed_img, lang="eng", config=custom_config)
    
    return text.strip()[:char_limit]  # Trim text to character limit

def extract_text_with_easyocr(image, char_limit=500, confidence_threshold=0.3, debug=False):
    """Extracts text using EasyOCR."""

    # Use EasyOCR to extract text
    results = reader.readtext(image, detail=1)

    # Filter out text with low confidence
    filtered_text = [text for (bbox, text, confidence) in results if confidence > confidence_threshold]

    text = " ".join(filtered_text)  # Combine detected text
    return text[:char_limit]

def wait_for_keypress_and_extract(key="space", debug=False):
    """Loops until a key is pressed, then extracts text from the active window."""
    print(f"\nPress '{key}' to capture text from the active window...")

    # Wait for the key press 
    while True:
        if keyboard.is_pressed(key): 
            if debug :
                print(f"'{key}' pressed! Extracting text...")
            time.sleep(0.2)  # Small delay to prevent multiple triggers

            # Get window info
            title = get_active_window_title()

            screenshot = np.array(pyautogui.screenshot())  # Fallback to full screen

            text = ""
            #processed_img = preprocess_image_for_ocr(screenshot,debug=debug)   # Preprocess image
            #text = extract_text_with_easyocr(processed_img, 5000, debug=debug) # Use easyocr to retrieve text
            if debug : 
                print("\nExtracted Text:\n", text)
                        # If no text detected, return nothing
            #if not text.strip():
            #    return {"image":None, "content":"", "title":title}
            
            return {"image":screenshot, "content":text, "title":title}

def call_ollama(model_name, input_text="", context=None, image=None, debug=True):
    """
    Calls an Ollama model with the given input text and optional image.

    :param model_name: The name of the model (e.g., 'deepseek-r1' or 'llava')
    :param input_text: The text input for the model
    :param image: Image as a NumPy array or file path
    :param debug: Whether to print debug information
    :return: The response from the model
    """
    if debug:
        print("Calling Ollama model:", model_name)
        print("Input text:", input_text)


    # Convert image if provided
    image_bytes = None
    if model_name.lower() == "llava" and image is not None:
        if isinstance(image, str):  # If image is a file path
            with open(image, "rb") as img_file:
                image_bytes = img_file.read()
        elif isinstance(image, np.ndarray):  # If image is a NumPy array (screenshot)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_bytes_io = io.BytesIO()
            pil_image.save(img_bytes_io, format="PNG")
            image_bytes = img_bytes_io.getvalue()
        else:
            raise ValueError("Invalid image format. Provide a file path or a NumPy array.")

    # Prepare text prompt
    messages = context
    messages.append({"role": "user", "content": input_text, "images" : [image_bytes] if image_bytes else None})

    # Call Ollama with or without image
    response = ollama.chat(model=model_name, messages=messages, stream=True, options={
        "temperature": 0.5,      # ↓ Less randomness
        "top_k": 20,             # ↓ Restrict choices
        "top_p": 0.8,            # Optional: more focused outputs
        "stop": ["\n\n"]         # Stop early after a sentence
        })

    for chunk in response:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]

def monitor_window_changes(callback, poll_interval=1.0, debug=False):
    """
    Monitors for active window changes and calls a callback function.

    :param callback: Function to call when the window changes.
    :param poll_interval: Time in seconds between checks.
    """
    last_window = get_active_window_title()

    while True:
        current_window = get_active_window_title()
        if current_window != last_window and current_window:
            callback(current_window, debug)  # Call user-defined function
            last_window = current_window

        time.sleep(poll_interval)  # Wait before checking again

def on_window_change(new_window, debug=False):
    if debug :
        print(f"User switched to: {new_window}")
    time.sleep(0.2)  # Small delay to prevent multiple triggers

    # Get window info
    title = get_active_window_title()

    screenshot = np.array(pyautogui.screenshot())  # Fallback to full screen

    text = ""
    #processed_img = preprocess_image_for_ocr(screenshot,debug=debug)   # Preprocess image
    #text = extract_text_with_easyocr(processed_img, 5000, debug=debug) # Use easyocr to retrieve text
    if debug : 
        print("\nExtracted Text:\n", text)
                # If no text detected, return nothing
    #if not text.strip():
    #    return {"image":None, "content":"", "title":title}
    
    input_prompt = "Window title : " + title #+ "\n" + text
    response = call_ollama("llava", input_prompt, screenshot, True)

# App features import
import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QTextBrowser, QSizeGrip, QSlider, QPushButton, QFileDialog
from PyQt6.QtGui import QPixmap, QPainter, QColor, QShortcut, QKeySequence, QTextCursor
from PyQt6.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
import json

CONFIG_FILE = "config.json"
CHARACTER_FOLDER = "characters"  # Root character directory
MAX_CONTEXT_LENGTH = 5  # Number of past exchanges to keep

class ChatOverlay(QWidget):
    def __init__(self, requested_name=None):
        super().__init__()
        self.timer = 60000
        self.setWindowTitle("My Assistant")  # Custom window title

        # Window settings
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(200, 300)  # Set minimum size for resizing
        self.resize(300, 400)  # Default size

        # AI response label (click to view previous messages)
        self.response_label = QTextBrowser(self)
        self.response_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 250); padding: 5px; border-radius: 5px;")
        self.response_label.setFixedHeight(80)  # Limit height, enables scrolling
        self.response_label.setOpenExternalLinks(True)  # Allows clickable links in AI responses
        self.response_label.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # Always show scrollbar
        self.response_label.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scrolling
        self.response_label.mousePressEvent = self.show_message_history  # Click to show history

        # User input field
        self.input_field = QLineEdit(self)
        self.input_field.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 250); padding: 5px; border-radius: 5px;")
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.returnPressed.connect(self.send_message)

        """
        # Message history viewer (hidden by default)
        self.message_history = QTextBrowser(self)
        self.message_history.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 250); padding: 5px; border-radius: 5px;")
        self.message_history.setVisible(False)
        """
        # Resize handle (allows resizing)
        self.size_grip = QSizeGrip(self)

        # Layouts
        self.main_layout = QHBoxLayout(self)
        self.chat_layout = QVBoxLayout(self)
        # Parameters container (hidden by default)
        self.parameters_widget = QWidget(self)
        params_layout = QVBoxLayout()

        # Font size slider
        self.font_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.font_slider.setMinimum(8)
        self.font_slider.setMaximum(24)
        self.font_slider.setValue(12)  # Default font size
        self.font_slider.setTickInterval(1)
        self.font_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_slider.valueChanged.connect(self.update_font_size)

        self.font_label = QLabel(f"Font Size: {self.font_slider.value()} px", self)
        self.font_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 250); padding: 5px; border-radius: 5px;")

        # AI timer slider
        self.timer_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.timer_slider.setMinimum(1)  # 1 minute
        self.timer_slider.setMaximum(10)  # 10 minutes
        self.timer_slider.setValue(5)  # Default
        self.timer_slider.setTickInterval(1)
        self.timer_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.timer_slider.valueChanged.connect(self.update_timer_interval)

        self.timer_label = QLabel(f"Auto AI Timer: {self.timer_slider.value()} min", self)
        self.timer_label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 250); padding: 5px; border-radius: 5px;")

        # Load Character button
        self.load_character_button = QPushButton("Load Character", self)
        self.load_character_button.clicked.connect(self.load_character)

        # Add sliders to parameters layout
        params_layout.addWidget(self.font_label)
        params_layout.addWidget(self.font_slider)
        params_layout.addWidget(self.timer_label)
        params_layout.addWidget(self.timer_slider)
        params_layout.addWidget(self.load_character_button)
        self.parameters_widget.setLayout(params_layout)
        self.parameters_widget.setVisible(False)  # Start hidden

        # Show/Hide button
        self.toggle_button = QPushButton("Show Parameters", self)
        self.toggle_button.clicked.connect(self.toggle_parameters)

        # Dragging functionality
        self.old_pos = None

        # Exit shortcut (Ctrl + Esc)
        self.exit_shortcut = QShortcut(QKeySequence("Esc"), self)
        self.exit_shortcut.activated.connect(self.close)

        # Position the overlay at the bottom-left
        QTimer.singleShot(100, self.position_overlay)  # Delay to ensure correct placement

        # Auto-trigger AI every 5 minutes
        self.auto_ai_timer = QTimer(self)
        self.auto_ai_timer.timeout.connect(self.auto_trigger_ai)
        self.auto_ai_timer.start(self.timer)

        # Load saved settings
        self.load_config() # Includes the character name
        if requested_name :
            self.character_name = requested_name
            self.save_config()

        # Load character assets
        self.character_path = os.path.join(CHARACTER_FOLDER, self.character_name)
        self.character_image = os.path.join(self.character_path, "character.png")
        self.character_context = os.path.join(self.character_path, "context.txt")
        self.context_text = self.load_character_context()
        # Load character image
        self.character_label = QLabel(self)
        self.load_character_image()

        # Connect sliders to auto-save
        self.font_slider.valueChanged.connect(self.save_config)
        self.timer_slider.valueChanged.connect(self.save_config)

        # Chat Layout
        self.chat_layout.addWidget(self.character_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.chat_layout.addWidget(self.response_label)
        self.chat_layout.addWidget(self.input_field)
        #layout.addWidget(self.message_history)
        self.chat_layout.addWidget(self.size_grip, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)  # Resize grip
        self.chat_layout.addWidget(self.toggle_button)
        # Add chat layout (left) and parameters (right) to main layout
        self.main_layout.addLayout(self.chat_layout)  
        self.main_layout.addWidget(self.parameters_widget)  # Parameters on the right

        self.setLayout(self.main_layout)

        # Ollama worker
        self.worker = OllamaWorker("llava",self)

    def load_config(self):
        """Loads the saved parameters from a config file."""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                config = json.load(file)

            # Apply settings
            self.character_name = config.get("character_name", "Barry")  # Default to "Barry"
            self.font_slider.setValue(config.get("font_size", 12))  # Default 12px
            self.timer_slider.setValue(config.get("auto_timer", 5))  # Default 5 min
        else:
            self.save_config()  # Create config file if not found

    def save_config(self):
        """Saves the current parameters to a config file."""
        config = {
            "character_name": self.character_name,
            "font_size": self.font_slider.value(),
            "auto_timer": self.timer_slider.value(),
        }

        with open(CONFIG_FILE, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)

    def load_character(self):
        """Opens a file dialog to select a character folder and updates the character."""
        character_dir = QFileDialog.getExistingDirectory(self, "Select Character Folder", "character")

        if character_dir:
            character_name = os.path.basename(character_dir)  # Extract folder name
            self.character_name = character_name
            self.character_path = character_dir
            self.character_image = os.path.join(character_dir, "character.png")
            self.character_context = os.path.join(character_dir, "context.txt")

            # Load new character image
            self.load_character_image()

            # Load new character context
            self.context_text = self.load_character_context()
            
            self.save_config()  # Save new character selection

            # Update UI
            text = f"<b><span style='color:DarkOrchid;'>Loaded character: </span></b> <span style='color:white;'>{self.character_name}"
            self.response_label.setText(text)

    def load_character_image(self):
        """Loads the character sprite."""
        print("Searching for picture at : " + self.character_image)
        if os.path.exists(self.character_image):
            pixmap = QPixmap(self.character_image)
            print("\033[32mCharacter image found.\033[0m")
        else:
            pixmap = QPixmap(150, 150)
            pixmap.fill(Qt.GlobalColor.transparent)  # Fallback to an empty image if missing
        self.character_label.setPixmap(pixmap)
        self.character_label.setScaledContents(True)
        self.character_label.setFixedSize(200, 300)  # Adjust character size
        self.character_label.lower() # Move to background

    def load_character_context(self):
        """Loads the character's context from the text file."""
        if os.path.exists(self.character_context):
            with open(self.character_context, "r", encoding="utf-8") as file:
                return file.read().strip()
        return "No context available."

    def position_overlay(self):
        """Anchors the overlay to the bottom-left corner of the screen."""
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        x = screen_geometry.left() + 30  # 10px padding from the left
        y = screen_geometry.bottom() - self.height() - 30  # 10px padding above the taskbar
        self.move(x, y)

    def old_send_message(self):
        """Handles user input and updates AI response."""
        user_text = self.input_field.text().strip()
        if user_text:
            self.input_field.clear()
            self.update_response(f"You: {user_text}\nAI: Thinking...")  # Placeholder before AI response

            response = call_ollama("llava:7b", input_text=user_text ,context=self.context_text)
            ai_response = "AI: " + response  # Example: Reverse text as AI response
            self.update_response(ai_response)
    
    def send_message(self):
        """Handles user input and starts the Ollama response stream."""
        self.user_text = self.input_field.text().strip()
        if not self.user_text:
            return
        self.input_image = None #TODO
        self.input_image = capture_active_window()

        self.input_field.clear()
        text = f"<b><span style='color:lightblue;'>You:</span></b> <span style='color:white;'>{self.user_text}</span><br>" \
            f"<b><span style='color:lightgreen;'>{self.character_name}:</span></b> <span style='color:white;'></span>"
        #self.message_history.append(f"You: {self.user_text}")  # Store user message
        self.response_label.setText(text)  # Show user message

        # Reset the automatic AI call timer
        self.auto_ai_timer.start(self.timer)  # Restart 5-minute countdown

        # Disconnect previous connections (avoids duplicate signals)
        try:
            self.worker.new_text.disconnect()
            self.worker.finished.disconnect()
        except TypeError:
            pass  # Happens if they were never connected before

        # Start Ollama in a background thread
        self.worker.new_text.connect(self.append_response)  # Update response in real-time
        self.worker.finished.connect(self.finalize_response)  # Mark when done
        self.worker.start()

    def append_response(self, text):
        """Appends streamed text to the response label."""
        cursor = self.response_label.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)  # Move cursor to the end
        cursor.insertText(text)  # Insert text without new lines
        self.response_label.setTextCursor(cursor)  # Update cursor position
        self.response_label.verticalScrollBar().setValue(self.response_label.verticalScrollBar().maximum())  # Auto-scroll

    def finalize_response(self):
        """Handles any final adjustments when streaming is done."""
        self.auto_ai_timer.start(self.timer)  # Restart 5-minute countdown
        #self.message_history.append(self.response_label.text())  # Store final response

    def update_response(self, response):
        """Updates the response label and stores history."""
        self.message_history.append(response)  # Store in history
        self.response_label.setText(response)

    def show_message_history(self, event):
        """Toggles visibility of message history."""
        #self.message_history.setVisible(not self.message_history.isVisible())

    def mousePressEvent(self, event):
        """Allows dragging the overlay."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        """Updates window position while dragging, but keeps it within screen bounds."""
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            new_x = self.x() + delta.x()
            new_y = self.y() + delta.y()
            self.old_pos = event.globalPosition().toPoint()

            # Get screen boundaries
            screen = QApplication.primaryScreen()
            screen_geometry = screen.availableGeometry()

            # Ensure window stays inside screen
            min_x = screen_geometry.left()
            max_x = screen_geometry.right() - self.width()
            min_y = screen_geometry.top()
            max_y = screen_geometry.bottom() - self.height()

            # Clamp values to keep within screen
            new_x = max(min_x, min(new_x, max_x))
            new_y = max(min_y, min(new_y, max_y))

            self.move(new_x, new_y)

    def mouseReleaseEvent(self, event):
        """Stops dragging."""
        self.old_pos = None

    def auto_trigger_ai(self):
        """Automatically calls Ollama every 5 minutes."""
        if self.worker.available :
            self.user_text = "Ask me something or make a comment."  # Placeholder text
            self.input_image = capture_active_window()

            #self.response_label.setText(f"<b><span style='color:lightblue;'>Auto:</span></b> <span style='color:white;'>Analyzing...</span>")  # Placeholder
            #self.message_history.append(f"Auto: Analyzing...")

            # Disconnect previous connections to avoid duplicate messages
            try:
                self.worker.new_text.disconnect()
                self.worker.finished.disconnect()
            except TypeError:
                pass  # Happens if signals were never connected before

            # Print character name
            text = f"<br><b><span style='color:lightgreen;'>{self.character_name}:</span></b> <span style='color:white;'></span>"
            self.response_label.insertHtml(text)

            # Reconnect signals
            self.worker.new_text.connect(self.append_response)
            self.worker.finished.connect(self.finalize_response)

            # Start worker
            self.worker.start()
        else :
            print("AI busy")

    def toggle_parameters(self):
        """Toggles visibility of the parameters section."""
        is_visible = self.parameters_widget.isVisible()
        self.parameters_widget.setVisible(not is_visible)
        self.toggle_button.setText("Hide Parameters" if not is_visible else "Show Parameters")

    def update_timer_interval(self):
        """Updates the auto AI call interval based on slider value."""
        interval_minutes = self.timer_slider.value()
        self.auto_ai_timer.start(interval_minutes * 60000)  # Convert minutes to milliseconds
        self.timer_label.setText(f"Auto AI Timer: {interval_minutes} min")  # Update label

    def update_font_size(self):
        """Updates the font size based on slider value."""
        font_size = self.font_slider.value()
        self.font_label.setText(f"Font Size: {font_size} px")  # Update label

        font = self.response_label.font()
        font.setPointSize(font_size)
        self.response_label.setFont(font)  # Apply font size change

class OllamaWorker(QThread):
    """Runs Ollama chat in a separate thread and streams the response."""
    new_text = pyqtSignal(str)  # Signal to send new text chunks
    finished = pyqtSignal()  # Signal when response is complete

    def __init__(self, model_name, chat_handler):
        super().__init__()
        self.chat_handler = chat_handler
        self.model_name = model_name
        self.context = [{"role" : "system", "content": chat_handler.context_text}]
        self.context.append({"role": "system", "content": "React in maximum 3 short sentences. Talk in the first person.<"})  # Base prompt
        self.ai_response = ""  # Store AI response for context update
        self.last_valid_window = "No window opened"

        # Set up a QTimer to call `get_real_active_window()` every 10 seconds
        self.window_timer = QTimer(self)
        self.window_timer.timeout.connect(self.update_active_window)
        self.window_timer.start(5000)  # 10,000 ms = 10 seconds

        self.available = True

    def get_real_active_window(self):
        """Returns the actual active window, ignoring the assistant."""
        active_window = get_active_window_title()

        if active_window is None:  # If it's the assistant, return the last real window
            return self.last_valid_window

        # Store the last non-self window
        self.last_valid_window = active_window
        return active_window
    
    def update_active_window(self):
        """Regularly updates the last known active window."""
        self.last_valid_window = self.get_real_active_window()
        #print(f"Updated active window: {self.last_valid_window}")  # Debugging

    def run(self):
        """Runs the Ollama chat function in a separate thread and emits text chunks."""
        self.available = False
        screenshot = capture_active_window()
        window_title = "The active window is named : " + self.last_valid_window
        print(window_title)
        self.context.append({"role":"system", "content":window_title})
        self.ai_response = ""  # Reset AI response
        for chunk in call_ollama(
            model_name=self.model_name,
            input_text=self.chat_handler.user_text,
            context=self.context,
            image=screenshot,
        ):
            #print(chunk, end="", flush=True)  # Debugging
            self.new_text.emit(chunk)  # Send text to UI immediately
            self.ai_response += chunk

        self.context.append({"role": "user", "content": self.chat_handler.user_text + "Please be concise, answer in maximum 3 sentences."})
        self.context.append({"role": "assistant", "content": self.ai_response})
        if len(self.context) > MAX_CONTEXT_LENGTH:
            self.context = self.context[:1] + self.context[-(MAX_CONTEXT_LENGTH - 1):]  # Keep system + last N messages
        self.finished.emit()  # Notify when done
        self.available = True