import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO model (using YOLOv8 pre-trained on COCO dataset)
# The model will recognize people as class "0" in COCO dataset
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a small model; can use 'yolov8m.pt' for a more accurate model

# Open a video capture stream (0 means default camera)
cap = cv2.VideoCapture(0)

# Define a function to process each frame and detect humans
def process_frame(frame):
    # Run YOLOv8 model on the frame
    results = model(frame)
    
    # Extract detections and filter by class (0 = person)
    detections = results[0].boxes
    for box in detections:
        class_id = box.cls
        if class_id == 0:  # Person class in COCO dataset
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf  # Confidence of detection

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Person: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

# Loop to continuously capture and process video frames
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break
    
    # Process the frame to detect and track humans
    frame = process_frame(frame)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Human Detection', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
