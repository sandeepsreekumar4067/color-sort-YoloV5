import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("yolov5s.pt")  # Replace with your YOLO model file

# Color ranges in HSV format mapped to names
COLOR_RANGES = {
    "Red": [(0, 50, 50), (10, 255, 255)],
    "Orange": [(11, 50, 50), (25, 255, 255)],
    "Yellow": [(26, 50, 50), (35, 255, 255)],
    "Green": [(36, 50, 50), (85, 255, 255)],
    "Blue": [(86, 50, 50), (125, 255, 255)],
    "Purple": [(126, 50, 50), (150, 255, 255)],
    "Pink": [(151, 50, 50), (170, 255, 255)],
    "Brown": [(10, 50, 50), (20, 150, 150)],
    "Black": [(0, 0, 0), (180, 255, 30)],  
    "White": [(0, 0, 231), (180, 18, 255)],  
    "Gray": [(0, 0, 50), (180, 18, 230)],  
    # "Teal": [(85, 50, 50), (95, 255, 255)], 
    # "Light Blue": [(90, 50, 50), (110, 150, 255)],  
    # "Beige": [(15, 20, 70), (25, 150, 200)],  
    # "Olive Green": [(60, 50, 50), (80, 150, 150)], 
    # "Peach": [(10, 50, 70), (20, 150, 255)],  
    # "Maroon": [(0, 50, 20), (10, 150, 100)],  
    # "Gold": [(20, 100, 100), (30, 255, 255)],  
    # "Silver": [(0, 0, 150), (180, 50, 255)],  
    # "Neon Green": [(40, 255, 255), (55, 255, 255)],  
    # "Neon Pink": [(170, 100, 100), (180, 255, 255)], 
}


def get_color_name(hsv_value):
    """
    Map an HSV value to a color name based on predefined ranges.
    """
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower = np.array(lower)
        upper = np.array(upper)
        if cv2.inRange(np.uint8([[hsv_value]]), lower, upper):
            return color_name
    return "Unknown"

def get_dominant_color(image):
    """
    Get the dominant color of the image in HSV.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    dominant_hue = np.argmax(hist) // 256
    dominant_sat = np.argmax(hist) % 256
    return np.array([dominant_hue, dominant_sat, 200])  # Fixed value for brightness

def update_frame():
    global cap, canvas, photo, color_label

    ret, frame = cap.read()
    if not ret:
        return

    # Run YOLO on the frame
    results = model.predict(frame, stream=True)

    detected_colors = []
    for result in results:
        for box in result.boxes:
            if hasattr(box, 'xyxy') and len(box.xyxy[0]) == 4:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf
                class_id = int(box.cls)

                # Crop the detected object
                cropped_object = frame[y1:y2, x1:x2]

                # Check the dominant color
                dominant_color_hsv = get_dominant_color(cropped_object)
                color_name = get_color_name(dominant_color_hsv)

                # Display the bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{color_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add color name to detected colors
                detected_colors.append(color_name)

    # Update the detected colors label
    if detected_colors:
        color_label.config(text="Detected Colors:\n" + ", ".join(set(detected_colors)))
    else:
        color_label.config(text="Detected Colors:\nNone")

    # Convert the frame to RGB for Tkinter display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    photo = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # Schedule the next frame update
    window.after(10, update_frame)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create the main Tkinter window
window = tk.Tk()
window.title("YOLO Color Detection")

# Create a canvas to display the video feed
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Create a label to display detected colors
color_label = tk.Label(window, text="Detected Colors:\nNone", font=("Arial", 14))
color_label.pack()

# Start the video feed update
update_frame()

# Run the Tkinter main loop
window.mainloop()

# Release the camera after the window is closed
cap.release()
cv2.destroyAllWindows()
