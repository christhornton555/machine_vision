import cv2
import numpy as np

def apply_instance_mask(frame, masks, class_ids, class_names, alpha=0.5):
    """
    Apply semi-transparent instance segmentation masks to the frame with unique colors.

    Args:
        frame (np.array): The video frame.
        masks (list): List of masks to apply.
        class_ids (list): List of class IDs corresponding to the detected objects.
        class_names (list): Class names of detected objects.
        alpha (float): Transparency of the mask.

    Returns:
        np.array: The frame with applied instance segmentation masks.
    """
    overlay = frame.copy()
    h, w, _ = frame.shape

    # Generate random colors for each class (indexed by class_ids)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

    # Apply masks with distinct colors for each instance
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        # Convert class ID to integer
        class_id = int(class_ids[i])

        # Get the color for the corresponding class
        color = colors[class_id]

        for c in range(3):  # Apply mask color to each channel
            frame[:, :, c] = np.where(mask_binary == 1, frame[:, :, c] * (1 - alpha) + alpha * color[c], frame[:, :, c])

        # Get class name and add a label on the object
        class_name = class_names[class_id]
        cv2.putText(frame, class_name, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color.tolist(), 2)

    return frame
