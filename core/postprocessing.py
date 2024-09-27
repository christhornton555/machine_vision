import cv2
import numpy as np

# Define colors for different object classes (COCO dataset classes)
COCO_COLORS = {
    'person': (255, 0, 0),       # Blue
    'bicycle': (0, 255, 0),      # Green
    'car': (0, 0, 255),          # Red
    'dog': (255, 255, 0),        # Cyan
    'cat': (255, 0, 255),        # Magenta
    # Add more classes and colors as needed, or generate random colors dynamically
}

def apply_mask(frame, masks, classes, labels, confidences, alpha=0.5):
    """
    Apply semi-transparent masks to the frame, label objects, and assign colors to different objects.

    Args:
        frame (np.array): The video frame.
        masks (list): List of masks to apply.
        classes (list): List of class indices corresponding to detected objects.
        labels (list): List of class labels (object names).
        confidences (list): Confidence scores of detected objects.
        alpha (float): Transparency of the mask.

    Returns:
        np.array: The frame with applied masks, colors, and labels.
    """
    overlay = frame.copy()
    
    for i, mask in enumerate(masks):
        label = labels[i]
        confidence = confidences[i]

        # Assign color for the current object class
        color = COCO_COLORS.get(label, (255, 255, 255))  # Default to white if class not in COCO_COLORS

        # Resize and binarize the mask
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # Apply the mask to the frame
        for c in range(3):  # Apply mask color to each channel
            frame[:, :, c] = np.where(mask_binary == 1, frame[:, :, c] * (1 - alpha) + alpha * color[c], frame[:, :, c])

        # Add label and confidence score to the frame
        cv2.putText(frame, f'{label} {confidence:.2f}', (10, 30 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, lineType=cv2.LINE_AA)

    return frame
