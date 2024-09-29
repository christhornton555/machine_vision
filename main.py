import cv2
import numpy as np
import argparse
from collections import deque
from config.config import select_device
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.postprocessing import apply_instance_mask, display_brightness
from core.camera_control import camera_settings
from core.postprocessing import calculate_brightness

# Buffer size for temporal smoothing (number of frames to buffer)
BUFFER_SIZE = 5

# Detection threshold to consider an object valid (e.g., detected in 3 out of 5 frames)
DETECTION_THRESHOLD = 3

# Brightness thresholds for low-light conditions
LOW_LIGHT_THRESHOLD_65 = 65
LOW_LIGHT_THRESHOLD_45 = 45

# Frame buffer size (store the most recent 3 frames)
FRAME_BUFFER_SIZE = 3

def add_frames(*frames):
    """
    Add multiple frames by summing pixel values and clipping the result to [0, 255].

    Args:
        *frames: Variable number of frames to be added together.

    Returns:
        np.array: The resulting frame after adding the pixel values.
    """
    added_frame = np.sum(frames, axis=0)
    return np.clip(added_frame, 0, 255).astype(np.uint8)

def smooth_detections(detection_buffer):
    """
    Smooth detection results across the buffer by considering objects present in a majority of frames.

    Args:
        detection_buffer (deque): Buffer of detection results over the last few frames.

    Returns:
        smoothed_results: Averaged or most common detection results from the buffer.
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

    # Initialize the object detector with the segmentation model (for instance segmentation)
    segmentation_model_path = 'models/yolov8n-seg.pt'
    detector = ObjectDetector(segmentation_model_path, device)

    detection_buffer = deque(maxlen=BUFFER_SIZE)  # Initialize the detection buffer
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # Buffer to hold the three most recent frames

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame or end of video.")
            break

        # Calculate brightness
        brightness = calculate_brightness(frame)

        # Add the current frame to the frame buffer
        frame_buffer.append(frame)

        # Handle low-light conditions with different thresholds
        if brightness < LOW_LIGHT_THRESHOLD_45 and len(frame_buffer) == 3:
            # If brightness is below 45, combine the current frame with the previous two frames
            frame = add_frames(frame_buffer[-1], frame_buffer[-2], frame_buffer[-3])
        elif brightness < LOW_LIGHT_THRESHOLD_65 and len(frame_buffer) >= 2:
            # If brightness is between 45 and 65, combine the current frame with the previous frame
            frame = add_frames(frame_buffer[-1], frame_buffer[-2])

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)
        
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

            # Apply instance masks with different colors and labels
            frame = apply_instance_mask(frame, masks, class_ids, classes)

        # Calculate and display brightness
        frame = display_brightness(frame, brightness, LOW_LIGHT_THRESHOLD_65)

        # Show the frame with instance segmentation and brightness applied
        cv2.imshow('YOLOv8 Instance Segmentation with Moving Labels and Temporal Smoothing', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse arguments for choosing the video source
    parser = argparse.ArgumentParser(description='YOLOv8 Instance Segmentation with Video/Camera')
    parser.add_argument('--video', type=str, default=None, help='Path to an MP4 video file. If not provided, webcam will be used.')
    args = parser.parse_args()

    # Use webcam (source=0) if no video file is provided, otherwise use the provided MP4 file
    video_source = 0 if args.video is None else args.video

    main(source=video_source)
