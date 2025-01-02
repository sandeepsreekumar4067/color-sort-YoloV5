import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("yolov5s.pt")  # Replace with your YOLO model file

# Define the target color range in HSV format
TARGET_COLOR_LOWER = np.array([0, 100, 100])  # Lower HSV bounds for target color
TARGET_COLOR_UPPER = np.array([10, 255, 255])  # Upper HSV bounds for target color

def get_dominant_color(image):
    """
    Get the dominant color of the image in HSV.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, TARGET_COLOR_LOWER, TARGET_COLOR_UPPER)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the mean color in the masked area
    hsv_mean = cv2.mean(hsv_image, mask)
    return hsv_mean

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

                # Check if the dominant color matches the target color
                dominant_color = get_dominant_color(cropped_object)
                if TARGET_COLOR_LOWER[0] <= dominant_color[0] <= TARGET_COLOR_UPPER[0]:
                    # Display the bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Detected: {class_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Add color to detected colors
                    detected_colors.append(f"HSV: {dominant_color[:3]}")

    # Update the detected colors label
    if detected_colors:
        color_label.config(text="Detected Colors:\n" + "\n".join(detected_colors))
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
