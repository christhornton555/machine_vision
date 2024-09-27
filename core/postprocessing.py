import cv2
import numpy as np

def apply_mask(frame, masks, color=(0, 255, 0), alpha=0.5):
    """
    Apply semi-transparent masks to the frame.

    Args:
        frame (np.array): The video frame.
        masks (list): List of masks to apply.
        color (tuple): The color of the mask in BGR format.
        alpha (float): Transparency of the mask.

    Returns:
        np.array: The frame with applied masks.
    """
    overlay = frame.copy()
    
    for mask in masks:
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        for c in range(3):  # Apply mask color to each channel
            frame[:, :, c] = np.where(mask_binary == 1, frame[:, :, c] * (1 - alpha) + alpha * color[c], frame[:, :, c])

    return frame
