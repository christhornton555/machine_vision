import cv2
from config.config import select_device
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.postprocessing import apply_mask

def main():
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam)
    video_capture = get_video_stream(source=0)

    # Initialize the object detector with detection and segmentation models
    detection_model_path = 'models/yolov8n.pt'
    segmentation_model_path = 'models/yolov8n-seg.pt'
    detector = ObjectDetector(detection_model_path, segmentation_model_path, device)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Object detection
        detection_results = detector.detect(frame)

        # Apply segmentation masks
        segmentation_results = detector.segment(frame)
        if segmentation_results[0].masks is not None:
            masks = segmentation_results[0].masks.data.cpu().numpy()
            frame = apply_mask(frame, masks)

        # Show the frame with masks applied
        cv2.imshow('YOLO Object Detection & Segmentation', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
