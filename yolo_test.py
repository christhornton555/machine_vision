import cv2
import numpy as np
from ultralytics import YOLO

# Initialize the YOLO segmentation model (using YOLOv8 pre-trained on COCO dataset)
model = YOLO('yolov8n-seg.pt')  # Use the segmentation model (nano version)

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

# Define random colors for each class (for segment masks)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype='uint8')

# Function to overlay the segmentation mask on the frame
def apply_mask(image, mask, color, alpha=0.5):
    """Overlay mask on image with a given color and alpha transparency."""
    for c in range(3):  # Apply the color to each channel
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])
    return image

# Function to process each frame and detect objects with segmentation
def process_frame(frame):
    # Run YOLOv8 model on the frame
    results = model(frame)
    
    # Extract detections (boxes, masks, and class predictions)
    detections = results[0]
    boxes = detections.boxes
    masks = detections.masks.data.cpu().numpy() if detections.masks else None
    
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())  # Convert the tensor to an integer
        confidence = box.conf.item()  # Extract the float value from the tensor
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get the segmentation mask for the current object
        if masks is not None:
            # The mask will be in the format of (num_objects, height, width), so we resize it to the frame size
            mask = masks[i]  # Get the mask for the ith object
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize mask to the frame size
            mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Binarize the mask

            # Apply the mask with a color overlay
            color = COLORS[class_id]
            frame = apply_mask(frame, mask_binary, color)
        
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

    # Process the frame to detect and track objects with segmentation
    frame = process_frame(frame)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Instance Segmentation', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
