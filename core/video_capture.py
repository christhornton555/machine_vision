import cv2

def get_video_stream(source=0):
    """
    Get the video stream either from the webcam (source=0) or an MP4 file.

    Args:
        source (int or str): If source=0, use the webcam. Otherwise, it should be the path to an MP4 file.

    Returns:
        cv2.VideoCapture: The video capture object for either the webcam or the MP4 file.
    """
    video_capture = cv2.VideoCapture(source)

    # Check if the video capture source was opened correctly
    if not video_capture.isOpened():
        raise ValueError(f"Unable to open video source: {source}")

    return video_capture
