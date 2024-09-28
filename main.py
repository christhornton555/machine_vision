import cv2
import numpy as np
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

def add_frames(frame1, frame2):
    """
    Add two frames by summing pixel values and clipping the result to [0, 255].
    """
    added_frame = cv2.add(frame1, frame2)
    return np.clip(added_frame, 0, 255).astype(np.uint8)

def smooth_detections(detection_buffer):
    """
    Smooth detection results across the buffer by considering objects present in a majority of frames.

    Args:
        detection_buffer (deque): Buffer of detection results over the last few frames.

    Returns:
        smoothed_results: Averaged or most common detection results from the buffer.
    """
    # Initialize the result storage for smoothed results
    smoothed_results = None
    
    if len(detection_buffer) > 0:
        # Take the detection results that are present in the majority of the frames
        frame_count = len(detection_buffer)
        results_count = {}
        for results in detection_buffer:
            for obj_class in results.boxes.cls.cpu().numpy():
                obj_class = int(obj_class)
                results_count[obj_class] = results_count.get(obj_class, 0) + 1

        # Keep only the results that appear in at least the threshold of frames
        valid_results = [cls for cls, count in results_count.items() if count >= DETECTION_THRESHOLD]
        
        # Filter out the detection results for the valid objects
        smoothed_results = [results for results in detection_buffer if int(results.boxes.cls.cpu().numpy()[0]) in valid_results]

    return smoothed_results[-1] if smoothed_results else None

def main():
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam)
    video_capture = get_video_stream(source=0)

    # Set manual focus for the Logitech C920
    camera_settings(video_capture, auto_focus=False, focus_value=255)

    # Initialize the object detector with the segmentation model (for instance segmentation)
    segmentation_model_path = 'models/yolov8n-seg.pt'
    detector = ObjectDetector(segmentation_model_path, device)

    low_light_threshold = 20
    detection_buffer = deque(maxlen=BUFFER_SIZE)  # Initialize the detection buffer

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate brightness
        brightness = calculate_brightness(frame)

        # If brightness is below threshold (e.g., 20), add the current frame with the next frame
        if brightness < low_light_threshold:
            ret_next, next_frame = video_capture.read()
            if ret_next:
                frame = add_frames(frame, next_frame)

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)
        
        # Add current segmentation results to the buffer
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
        frame = display_brightness(frame, brightness, low_light_threshold)

        # Show the frame with instance segmentation and brightness applied
        cv2.imshow('YOLOv8 Instance Segmentation with Moving Labels and Temporal Smoothing', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
