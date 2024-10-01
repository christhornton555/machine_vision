import cv2
import numpy as np
import argparse
import time
from collections import deque
from config.config import select_device
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.postprocessing import apply_instance_mask, display_brightness, draw_skeleton
from core.camera_control import camera_settings
from core.postprocessing import calculate_brightness

# Buffer size for temporal smoothing (number of frames to buffer)
BUFFER_SIZE = 5

# Detection threshold to consider an object valid (e.g., detected in 3 out of 5 frames)
DETECTION_THRESHOLD = 3

# Brightness thresholds for low-light conditions
LOW_LIGHT_THRESHOLD_1 = 65
LOW_LIGHT_THRESHOLD_2 = 30

# Frame buffer size (store the most recent 3 frames)
FRAME_BUFFER_SIZE = 3

# Cooldown time (in seconds) to prevent switching between brightness thresholds too quickly
COOLDOWN_TIME = 2

# Threshold for deciding if a person is large enough in the frame
MIN_PERSON_SIZE = 0.02  # Fraction of the frame size (e.g., 20%)

# OpenPose-like connections for limbs
OPENPOSE_CONNECTIONS = [
    (0, 1),  # Nose to Neck
    (1, 2),  # Neck to Right Shoulder
    (1, 5),  # Neck to Left Shoulder
    (2, 3),  # Right Shoulder to Right Elbow
    (3, 4),  # Right Elbow to Right Wrist
    (5, 6),  # Left Shoulder to Left Elbow
    (6, 7),  # Left Elbow to Left Wrist
    (1, 8),  # Neck to Pelvis
    (8, 9),  # Pelvis to Right Hip
    (9, 10), # Right Hip to Right Knee
    (10, 11), # Right Knee to Right Ankle
    (8, 12), # Pelvis to Left Hip
    (12, 13), # Left Hip to Left Knee
    (13, 14)  # Left Knee to Left Ankle
]

# Skeleton color scheme
SKELETON_COLORS = {
    'left': (255, 0, 0),  # Blue for left side
    'right': (0, 0, 255),  # Red for right side
    'center': (0, 255, 0)  # Green for center parts (neck, spine, pelvis)
}

def add_frames(*frames):
    """
    Add multiple frames by summing pixel values and clipping the result to [0, 255].
    """
    added_frame = np.sum(frames, axis=0)
    return np.clip(added_frame, 0, 255).astype(np.uint8)

def smooth_detections(detection_buffer):
    """
    Smooth detection results across the buffer by considering objects present in a majority of frames.
    """
    smoothed_results = None

    if len(detection_buffer) > 0:
        results_count = {}

        # Go through the buffer and count valid detections
        for results in detection_buffer:
            if results.boxes is not None and len(results.boxes.cls) > 0:
                for obj_class in results.boxes.cls.cpu().numpy():
                    obj_class = int(obj_class)
                    results_count[obj_class] = results_count.get(obj_class, 0) + 1

        # Filter detections that meet the threshold
        valid_results = [cls for cls, count in results_count.items() if count >= DETECTION_THRESHOLD]
        
        # Only keep the valid results in the smoothed_results list
        smoothed_results = [results for results in detection_buffer if len(results.boxes.cls) > 0 and int(results.boxes.cls.cpu().numpy()[0]) in valid_results]

    return smoothed_results[-1] if smoothed_results else None

def main(source):
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam or MP4)
    video_capture = get_video_stream(source=source)

    if source == 0:
        # Set manual focus for the Logitech C920
        camera_settings(video_capture, auto_focus=False, focus_value=255)

    # Initialize the object detector with both segmentation and pose models
    segmentation_model_path = 'models/yolov8n-seg.pt'
    pose_model_path = 'models/yolov8n-pose.pt'
    detector = ObjectDetector(segmentation_model_path, pose_model_path, device)

    detection_buffer = deque(maxlen=BUFFER_SIZE)  # Initialize the detection buffer
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # Buffer to hold the three most recent frames

    # Initialize the cooldown timer for switching between brightness thresholds
    last_threshold_switch_time = time.time()
    current_threshold = None  # Track the current threshold level

    # Use a simple flag to track if the app has just started, to negate the risk of an empty frame buffer
    just_started = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame or end of video.")
            break

        # Calculate brightness
        brightness = calculate_brightness(frame)

        # Add the current frame to the frame buffer
        frame_buffer.append(frame)
        # Fill the buffer with this one frame on the first loop
        if just_started:
            for i in range(FRAME_BUFFER_SIZE - 1):
                frame_buffer.append(frame)
            just_started = False

        # Check how much time has passed since the last threshold switch
        time_since_last_switch = time.time() - last_threshold_switch_time

        # Select which light threshold to use, with cooldown to prevent rapid switching
        if time_since_last_switch > COOLDOWN_TIME:
            if brightness > LOW_LIGHT_THRESHOLD_1:
                current_threshold = None  # Use default threshold if brightness is above 65
            elif brightness < LOW_LIGHT_THRESHOLD_1 and brightness > LOW_LIGHT_THRESHOLD_2:
                current_threshold = "LOW_LIGHT_THRESHOLD_1"  # Update the current threshold
            elif brightness < LOW_LIGHT_THRESHOLD_2:
                current_threshold = "LOW_LIGHT_THRESHOLD_2"  # Update the current threshold
            last_threshold_switch_time = time.time()  # Reset the cooldown timer

        # Handle low-light conditions with different thresholds
        if current_threshold == "LOW_LIGHT_THRESHOLD_1":
            # If brightness is between LLT1 and LLT2, combine the current frame with the previous frame
            frame = add_frames(frame_buffer[-1], frame_buffer[-2])
        elif current_threshold == "LOW_LIGHT_THRESHOLD_2":
            # If brightness is below LLT2, combine the current frame with the previous two frames
            frame = add_frames(frame_buffer[-1], frame_buffer[-2], frame_buffer[-3])

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)

        # Perform pose detection for skeleton tracking
        pose_results = detector.detect_pose(frame)

        # Add current segmentation results to the buffer (only if valid results exist)
        if segmentation_results[0].boxes is not None and len(segmentation_results[0].boxes.cls) > 0:
            detection_buffer.append(segmentation_results[0])

        # Smooth detection results across frames
        smoothed_results = smooth_detections(detection_buffer)

        if smoothed_results and smoothed_results.masks is not None:
            # Get masks and classes for segmentation
            masks = smoothed_results.masks.data.cpu().numpy()  # Mask data
            classes = smoothed_results.names  # Class names
            class_ids = smoothed_results.boxes.cls.cpu().numpy()  # Detected class indices
            boxes = smoothed_results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates

            # Apply instance masks with different colors and labels
            frame = apply_instance_mask(frame, masks, class_ids, classes)

            # Process skeletons for detected people
            for i, class_id in enumerate(class_ids):
                if classes[class_id] == "person":
                    # Check if keypoints are available from the pose model
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > i:
                        # Get bounding box for the person
                        x1, y1, x2, y2 = boxes[i]
                        print(x1, y1, x2, y2)  # TODO - testing

                        # Calculate the size of the bounding box relative to the frame size
                        box_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        relative_size = box_area / frame_area

                        # Only track skeleton if person is sufficiently large in the frame
                        if relative_size > MIN_PERSON_SIZE:
                            # Extract keypoints
                            keypoints = pose_results[0].keypoints[i].cpu().numpy()
                            # Draw skeleton
                            frame = draw_skeleton(frame, keypoints, OPENPOSE_CONNECTIONS, SKELETON_COLORS)

        # Calculate and display brightness
        frame = display_brightness(frame, brightness, current_threshold)

        # Show the frame with instance segmentation, skeleton, and brightness applied
        cv2.imshow('YOLOv8 Segmentation and Pose Detection', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse arguments for choosing the video source
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation and Pose Detection with Video/Camera')
    parser.add_argument('--video', type=str, default=None, help='Path to an MP4 video file. If not provided, webcam will be used.')
    args = parser.parse_args()

    # Use webcam (source=0) if no video file is provided, otherwise use the provided MP4 file
    video_source = 0 if args.video is None else args.video

    main(source=video_source)

