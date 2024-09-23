import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO pose model (using YOLOv8 pre-trained for keypoints detection)
model = YOLO('yolov8n-pose.pt')  # Use the pose detection model (nano version)

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

# Skeleton connections (pairs of keypoint indices)
SKELETON_CONNECTIONS = [
    (5, 6),   # Shoulders
    (5, 7),   # Left Shoulder to Elbow
    (7, 9),   # Left Elbow to Wrist
    (6, 8),   # Right Shoulder to Elbow
    (8, 10),  # Right Elbow to Wrist
    (11, 12), # Hips
    (5, 11),  # Left Shoulder to Left Hip
    (6, 12),  # Right Shoulder to Right Hip
    (11, 13), # Left Hip to Knee
    (13, 15), # Left Knee to Ankle
    (12, 14), # Right Hip to Knee
    (14, 16), # Right Knee to Ankle
    (0, 5),   # Nose to Left Shoulder
    (0, 6),   # Nose to Right Shoulder
]

# OpenPose colors: (B, G, R) for left side, right side, and central body
POSE_COLORS = {
    'left': (255, 0, 0),  # Blue
    'right': (0, 0, 255), # Red
    'center': (0, 255, 0) # Green
}

# Open a video capture stream (0 means default camera)
cap = cv2.VideoCapture(0)

# Function to draw skeleton connections
def draw_skeleton(image, keypoints):
    """Draw skeleton connections between keypoints."""
    for (start_idx, end_idx) in SKELETON_CONNECTIONS:
        if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:  # Only draw if confidence is high
            x_start, y_start = int(keypoints[start_idx][0]), int(keypoints[start_idx][1])
            x_end, y_end = int(keypoints[end_idx][0]), int(keypoints[end_idx][1])
            
            # Assign colors based on body part
            if start_idx in [5, 7, 9, 11, 13, 15]:  # Left side
                color = POSE_COLORS['left']
            elif start_idx in [6, 8, 10, 12, 14, 16]:  # Right side
                color = POSE_COLORS['right']
            else:  # Central body (e.g., torso, head)
                color = POSE_COLORS['center']
            
            cv2.line(image, (x_start, y_start), (x_end, y_end), color, 2)
    return image

# Function to draw keypoints (optional circles on joints)
def draw_keypoints(image, keypoints, color=(0, 255, 255)):
    """Draw keypoints on the image."""
    for kp in keypoints:
        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
        if conf > 0.5:  # Only draw if confidence is high enough
            cv2.circle(image, (x, y), 5, color, -1)
    return image

# Function to process each frame and detect objects, keypoints, and segmentation
def process_frame(frame):
    # Run YOLOv8 model on the frame
    results = model(frame)
    
    # Extract detections (boxes, masks, keypoints)
    detections = results[0]
    boxes = detections.boxes
    keypoints = detections.keypoints.data.cpu().numpy() if detections.keypoints else None
    
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())  # Convert the tensor to an integer
        
        # Get bounding box coordinates (for drawing if needed)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw keypoints and skeleton
        if keypoints is not None:
            kp = keypoints[i]
            frame = draw_keypoints(frame, kp)
            frame = draw_skeleton(frame, kp)
        
        # Get class label from COCO_CLASSES list
        label = f'{COCO_CLASSES[class_id]}'
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame

# Loop to continuously capture and process video frames
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Process the frame to detect and track objects with keypoints and skeleton
    frame = process_frame(frame)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Keypoint Detection with Skeleton', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
