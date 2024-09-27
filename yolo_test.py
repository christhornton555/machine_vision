import cv2
import numpy as np
import dlib
import mediapipe as mp
from ultralytics import YOLO
import torch
from math import dist

# Function to choose device (CPU or GPU)
def select_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Running on GPU")
    else:
        device = torch.device('cpu')
        print("Running on CPU")
    return device

# Ask the user to choose between CPU and GPU
device = select_device()

# Initialize the YOLO pose model (using YOLOv8 pre-trained for keypoints detection) on the selected device
model = YOLO('yolov8n-pose.pt')  # Use the pose detection model (nano version)
model.to(device)  # Move model to the specified device (CPU or GPU)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize Mediapipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize drawing utility for Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

# Skeleton connections (pairs of keypoint indices for YOLO)
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

# Function to apply gamma correction to adjust brightness
def apply_gamma_correction(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to apply contrast adjustment
def adjust_contrast(image, alpha=1.0, beta=0):
    """Apply contrast and brightness adjustment."""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Function to preprocess the input frame for YOLO
def preprocess_frame_for_yolo(frame, target_size=640):
    """Resize and normalize the frame for YOLO model."""
    # Resize to the target size (640x640 or any size YOLO expects)
    resized_frame = cv2.resize(frame, (target_size, target_size))

    # Convert to float32 and normalize the image
    resized_frame = resized_frame.astype(np.float32) / 255.0

    # Convert the frame from HWC (height, width, channels) to CHW (channels, height, width)
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension (B, C, H, W)
    
    # Move to the same device as the model
    frame_tensor = frame_tensor.to(device)
    return frame_tensor

# Function to calculate the brightness of the frame
def calculate_brightness(image):
    """Calculate the average brightness of the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])  # V channel represents brightness
    return brightness

# Function to adjust frame for both low-light and bright-light conditions
def adjust_brightness_contrast(frame, brightness, gamma, alpha_contrast):
    """Dynamically adjust the gamma and contrast based on brightness."""
    # Only adjust if the brightness is extremely high or low
    if brightness > 220:  # Very bright
        gamma = max(gamma - 0.1, 0.9)  # Slightly decrease gamma
        alpha_contrast = min(alpha_contrast + 0.1, 1.6)  # Slightly increase contrast
    elif brightness < 60:  # Very dark
        gamma = min(gamma + 0.1, 2.0)  # Slightly increase gamma
        alpha_contrast = min(alpha_contrast + 0.1, 1.6)  # Slightly increase contrast
    else:
        gamma = 1.0  # Reset gamma to default
        alpha_contrast = 1.0  # Reset contrast to default

    return apply_gamma_correction(frame, gamma), adjust_contrast(frame, alpha_contrast), gamma, alpha_contrast

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

# Function to draw YOLO keypoints (optional circles on joints)
def draw_keypoints(image, keypoints, color=(0, 255, 255)):
    """Draw keypoints on the image."""
    for kp in keypoints:
        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
        if conf > 0.5:  # Only draw if confidence is high enough
            cv2.circle(image, (x, y), 5, color, -1)
    return image

# Function to process each frame and detect objects, keypoints, hand tracking, and facial landmarks
def process_frame(frame, gamma, default_gamma, alpha_contrast):
    # Calculate brightness
    brightness = calculate_brightness(frame)

    # Adjust gamma and contrast based on brightness
    if brightness < 60 or brightness > 220:  # Only adjust in extreme lighting
        frame_gamma_adjusted, frame_adjusted, gamma, alpha_contrast = adjust_brightness_contrast(frame, brightness, gamma, alpha_contrast)
    else:
        frame_adjusted = frame
        gamma = 1.0
        alpha_contrast = 1.0

    # Preprocess the frame for YOLO
    frame_tensor = preprocess_frame_for_yolo(frame_adjusted)

    # Run YOLOv8 model on the frame for body keypoints
    results = model(frame_tensor)
    
    # Extract detections (boxes, masks, keypoints)
    detections = results[0]
    boxes = detections.boxes
    keypoints = detections.keypoints.data.cpu().numpy() if detections.keypoints else None

    head_keypoints_found = False

    # Process YOLO keypoints and skeleton
    for i, box in enumerate(boxes):
        class_id = int(box.cls.item())  # Convert the tensor to an integer
        
        # Get bounding box coordinates (for drawing if needed)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Check head-related keypoints (e.g., nose and shoulders)
        if keypoints is not None:
            kp = keypoints[i]
            if kp[0][2] > 0.5 and kp[5][2] > 0.5 and kp[6][2] > 0.5:  # If nose, left shoulder, and right shoulder are detected
                head_keypoints_found = True
                nose_x, nose_y = int(kp[0][0]), int(kp[0][1])
                left_shoulder_x, left_shoulder_y = int(kp[5][0]), int(kp[5][1])
                right_shoulder_x, right_shoulder_y = int(kp[6][0]), int(kp[6][1])

                # Draw keypoints and skeleton
                frame_adjusted = draw_keypoints(frame_adjusted, kp)
                frame_adjusted = draw_skeleton(frame_adjusted, kp)
        
        # Get class label from COCO_CLASSES list
        label = f'{COCO_CLASSES[class_id]}'
        
        # Draw bounding box and label on the frame
        cv2.rectangle(frame_adjusted, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_adjusted, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Convert frame to RGB for Mediapipe hand tracking
    rgb_frame = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2RGB)
    
    # Process hand landmarks with Mediapipe
    result_hands = hands.process(rgb_frame)
    if result_hands.multi_hand_landmarks:
        for hand_landmarks in result_hands.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame_adjusted,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    return frame_adjusted, gamma, alpha_contrast

# Loop to continuously capture and process video frames
default_gamma = 1.0
gamma = default_gamma
alpha_contrast = 1.0
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Process the frame with minimal gamma and contrast adjustment, only for extreme lighting
    frame, gamma, alpha_contrast = process_frame(frame, gamma, default_gamma, alpha_contrast)

    # Display the resulting frame
    cv2.imshow('YOLOv8 Keypoint Detection with Hand, Facial Tracking, and Minimal Gamma/Contrast Adjustments', frame)

    # Press 'q' to quit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
