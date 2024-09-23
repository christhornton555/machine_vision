import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO model (using YOLOv8 pre-trained on COCO dataset)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for the nano version

# COCO classes (80 different object categories)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush'
]

# Open a video capture stream (0 means default camera)
cap = cv2.VideoCapture(0)

# Define a function to process each frame and detect objects
def process_frame(frame):
    # Run YOLOv8 model on the frame
    results = model(frame)

    # Extract detections
    detections = results[0].boxes
    for box in detections:
        class_id = int(box.cls.item())  # Convert the tensor to an integer
        confidence = box.conf.item()  # Extract the float value from the tensor

        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class label from COCO_CLASSES list
        label = f'{COCO_CLASSES[class_id]}: {confidence:.2f}'

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Loop to continuously capture and process video frames
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Process the frame to detect and track objects
    frame = process_frame(frame)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Object Detection', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
