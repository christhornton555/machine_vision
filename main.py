import cv2
import numpy as np
from config.config import select_device
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.postprocessing import apply_instance_mask, display_brightness
from core.camera_control import camera_settings
from core.postprocessing import calculate_brightness

def add_frames(frame1, frame2):
    """
    Add two frames by summing pixel values and clipping the result to [0, 255].

    Args:
        frame1 (np.array): The first video frame.
        frame2 (np.array): The second video frame.

    Returns:
        np.array: The frame resulting from adding the pixel values of both frames.
    """
    # Add pixel values and clip to the range [0, 255] to prevent overflow
    added_frame = cv2.add(frame1, frame2)
    return np.clip(added_frame, 0, 255).astype(np.uint8)

def main():
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam)
    video_capture = get_video_stream(source=0)

    # Set manual focus for the Logitech C920 (disable auto-focus and set specific focus value)
    camera_settings(video_capture, auto_focus=False, focus_value=255)  # Adjust focus value as needed

    # Initialize the object detector with the segmentation model (for instance segmentation)
    segmentation_model_path = 'models/yolov8n-seg.pt'
    detector = ObjectDetector(segmentation_model_path, device)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate brightness
        brightness = calculate_brightness(frame)
        low_light_threshold = 20

        # If brightness is below threshold (e.g., 20), add the current frame with the next frame
        if brightness < low_light_threshold:
            ret_next, next_frame = video_capture.read()
            if ret_next:
                frame = add_frames(frame, next_frame)

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)
        if segmentation_results[0].masks is not None:
            # Get masks and classes for segmentation
            masks = segmentation_results[0].masks.data.cpu().numpy()  # Mask data
            classes = segmentation_results[0].names  # Class names
            class_ids = segmentation_results[0].boxes.cls.cpu().numpy()  # Detected class indices

            # Apply instance masks with different colors
            frame = apply_instance_mask(frame, masks, class_ids, classes)

        # Calculate and display brightness
        frame = display_brightness(frame, brightness, low_light_threshold)

        # Show the frame with instance segmentation and brightness applied
        cv2.imshow('YOLOv8 Instance Segmentation with Brightness and Frame Addition', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
