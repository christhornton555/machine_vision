import cv2

def camera_settings(cap, auto_focus=False, focus_value=255):
    """
    Apply camera settings using OpenCV.
    
    Args:
        cap (cv2.VideoCapture): The video capture object.
        auto_focus (bool): If True, enables auto-focus. If False, disables auto-focus and sets manual focus.
        focus_value (int): Focus value between 0 (near) and 255 (far). Only used when auto_focus is False.
    
    Returns:
        None
    """
    
    # Other capture settings
    cap.set(cv2.CAP_PROP_FPS, 30.0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if auto_focus:
        # Enable auto-focus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        print("Auto-focus enabled.")
    else:
        # Disable auto-focus and set manual focus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)
        print(f"Auto-focus disabled. Focus set to {focus_value}.")
