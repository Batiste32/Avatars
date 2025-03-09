import pytesseract
import pyautogui
import numpy as np
import cv2
import platform

if platform.system() == "Windows":
    import win32gui

    def get_active_window_rect():
        """Gets the active window's position (left, top, right, bottom)."""
        hwnd = win32gui.GetForegroundWindow()
        rect = win32gui.GetWindowRect(hwnd)
        return rect  # (left, top, right, bottom)
else:
    import pygetwindow as gw

    def get_active_window_rect():
        """Gets the active window's position for macOS/Linux (if supported)."""
        win = gw.getActiveWindow()
        if win:
            return (win.left, win.top, win.right, win.bottom)
        return None

def preprocess_image(img):
    """Preprocess the image to improve OCR accuracy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR accuracy
    scale_percent = 200  # Scale image by 200%
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Apply adaptive thresholding
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Sharpen image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray

def extract_text_from_window(char_limit=500):
    """Extracts text from the active window, cropping out the title bar."""
    rect = get_active_window_rect()
    
    if not rect:
        print("Could not determine active window size.")
        return ""

    left, top, right, bottom = rect
    width, height = right - left, bottom - top

    # Define title bar height (adjust if needed)
    title_bar_height = 30  # Approximate height of the title bar

    # Capture the full screen
    screenshot = np.array(pyautogui.screenshot())

    # Crop the active window only
    cropped_img = screenshot[top + title_bar_height:bottom, left:right]  # Remove title bar

    # Check if the crop is valid
    if cropped_img.size == 0:
        print("Invalid crop! Ensure the window is in focus and visible.")
        return ""

    processed_img = preprocess_image(cropped_img)  # Preprocess for OCR

    # Run OCR
    text = pytesseract.image_to_string(processed_img, lang="eng", config="--psm 6")
    
    return text.strip()[:char_limit]  # Trim text

# Example usage
print(extract_text_from_window(300))
