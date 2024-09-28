import cv2
import numpy as np

def calculate_brightness(frame):
    """
    Calculate the average brightness of the frame using the V channel from HSV color space.

    Args:
        frame (np.array): The video frame.

    Returns:
        float: Average brightness value (0 to 255).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])  # V channel represents brightness
    return brightness

def display_brightness(frame, pre_calculated_brightness, low_light_threshold):
    """
    Calculate and display the brightness value on the frame.

    Args:
        frame (np.array): The video frame.

    Returns:
        np.array: The frame with brightness displayed in the top-right corner.
    """
    # Set the text with brightness value
    brightness_text = f'Brightness: {pre_calculated_brightness:.2f}'
    
    # Get frame dimensions
    h, w, _ = frame.shape

    # Draw the brightness in the top-right corner (color: 255, 127, 255)
    cv2.putText(frame, brightness_text, (w - 240, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 127, 255), 2)

    if pre_calculated_brightness < low_light_threshold:
        low_light_warning = 'LOW'
        cv2.putText(frame, low_light_warning, (w - 86, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return frame

def apply_instance_mask(frame, masks, class_ids, class_names, alpha=0.5):
    """
    Apply semi-transparent instance segmentation masks to the frame with unique colors and label objects.

    Args:
        frame (np.array): The video frame.
        masks (list): List of masks to apply.
        class_ids (list): List of class IDs corresponding to the detected objects.
        class_names (list): Class names of detected objects.
        alpha (float): Transparency of the mask.

    Returns:
        np.array: The frame with applied instance segmentation masks and object labels.
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

        # Get class name and label size for the object
        class_name = class_names[class_id]
        label = f"{class_name}"

        # Calculate the centroid of the mask to place the label
        M = cv2.moments(mask_binary)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])  # Centroid X
            cY = int(M["m01"] / M["m00"])  # Centroid Y
        else:
            # If the centroid calculation fails (shouldn't happen), place the label in the top-left corner
            cX, cY = 10, 30 * (i + 1)

        # Set the label font size smaller
        font_scale = 0.5
        label_thickness = 1
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)

        # Add a small rectangle behind the label for better visibility
        cv2.rectangle(frame, (cX, cY - label_size[1] - 2), (cX + label_size[0], cY + 2), (0, 0, 0), -1)

        # Draw the label on top of the object
        cv2.putText(frame, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), label_thickness)

    return frame
