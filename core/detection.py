from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, detection_model_path, segmentation_model_path, device):
        # Load models with YOLO
        self.detection_model = YOLO(detection_model_path).to(device)
        self.segmentation_model = YOLO(segmentation_model_path).to(device)

    def detect(self, frame):
        """
        Detect objects in the frame using YOLO object detection model.

        Args:
            frame (np.array): Video frame from the webcam.

        Returns:
            results: YOLO detection results.
        """
        return self.detection_model(frame)

    def segment(self, frame):
        """
        Segment objects in the frame using YOLO segmentation model.

        Args:
            frame (np.array): Video frame from the webcam.

        Returns:
            results: YOLO segmentation results.
        """
        return self.segmentation_model(frame)
