import cv2

def get_video_stream(source=0):
    """
    Capture video from the given source. Defaults to webcam (source=0).

    Args:
        source (int or str): 0 for webcam or the path to a video file.

    Returns:
        cap: VideoCapture object.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Unable to open video source {source}")
    return cap
