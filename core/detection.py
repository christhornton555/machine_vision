from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, segmentation_model_path, device):
        # Load the YOLO segmentation model for instance segmentation
        self.segmentation_model = YOLO(segmentation_model_path).to(device)

    def segment(self, frame):
        """
        Perform instance segmentation using YOLOv8 segmentation model.

        Args:
            frame (np.array): Video frame from the webcam.

        Returns:
            results: YOLO segmentation results.
        """
        return self.segmentation_model(frame)
