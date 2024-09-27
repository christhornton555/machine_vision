import cv2
import numpy as np
from config.config import select_device
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.postprocessing import apply_instance_mask

def main():
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam)
    video_capture = get_video_stream(source=0)

    # Initialize the object detector with the segmentation model (for instance segmentation)
    segmentation_model_path = 'models/yolov8n-seg.pt'
    detector = ObjectDetector(segmentation_model_path, device)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)
        if segmentation_results[0].masks is not None:
            # Get masks and classes for segmentation
            masks = segmentation_results[0].masks.data.cpu().numpy()  # Mask data
            classes = segmentation_results[0].names  # Class names
            class_ids = segmentation_results[0].boxes.cls.cpu().numpy()  # Detected class indices

            # Apply instance masks with different colors and display brightness
            frame = apply_instance_mask(frame, masks, class_ids, classes)

        # Show the frame with instance segmentation applied
        cv2.imshow('YOLOv8 Instance Segmentation with Brightness', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
