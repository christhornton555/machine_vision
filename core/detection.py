from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, segmentation_model_path, pose_model_path, device):
        # Load the YOLO segmentation model for instance segmentation
        self.segmentation_model = YOLO(segmentation_model_path).to(device)
        # Load the YOLO pose model for keypoint detection
        self.pose_model = YOLO(pose_model_path).to(device)

    def segment(self, frame):
        """
        Perform instance segmentation using YOLOv8 segmentation model.

        Args:
            frame (np.array): Video frame from the webcam.

        Returns:
            results: YOLO segmentation results.
        """
        return self.segmentation_model(frame)

    def detect_pose(self, frame):
        """
        Perform pose detection using YOLOv8 pose model.

        Args:
            frame (np.array): Video frame from the webcam.

        Returns:
            results: YOLO pose detection results.
        """
        return self.pose_model(frame)
