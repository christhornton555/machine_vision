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

def display_brightness(frame, pre_calculated_brightness, current_threshold):
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

    if current_threshold == "LOW_LIGHT_THRESHOLD_1":
        low_light_warning = 'LOW1'
        cv2.putText(frame, low_light_warning, (w - 86, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif current_threshold == "LOW_LIGHT_THRESHOLD_2":
        low_light_warning = 'LOW2'
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

def draw_skeleton(frame, keypoints, connections, colors, keypoint_colors):
    """
    Draw a color-coded skeleton on a person based on keypoints and connections.

    Args:
        frame (np.array): The video frame.
        keypoints (np.array): The array of keypoints for a person (x, y, confidence).
        connections (list): List of tuples defining the connections between keypoints.
        colors (dict): A dictionary of colors for left, right, and center parts of the skeleton.

    Returns:
        np.array: The frame with the skeleton drawn.
    """
    keypoint_count = keypoints.shape[0]  # Get the total number of detected keypoints (should be 17 now)

    # Check if both left and right shoulders are present with confidence
    if keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
        # Calculate the neck keypoint as the midpoint between the shoulders, LShoulder = 5, RShoulder = 6
        neck_x = (keypoints[5][0] + keypoints[6][0]) / 2
        neck_y = (keypoints[5][1] + keypoints[6][1]) / 2
        neck_conf = (keypoints[5][2] + keypoints[6][2]) / 2

        # Append the neck keypoint to the keypoints array (x, y, confidence)
        neck_keypoint = np.array([neck_x, neck_y, neck_conf])
        keypoints = np.vstack([keypoints, neck_keypoint])
        keypoint_count = keypoints.shape[0]  # Update this if a neck has been calculated

        # Add connections for neck
        connections.append((17, 0))  # Connecting nose to neck
        connections.append((17, 5))  # Connecting neck to rshoulder
        connections.append((17, 6))  # Connecting neck to lshoulder
        connections.append((17, 11))  # Connecting neck to rhip
        connections.append((17, 12))  # Connecting neck to lhip

    for start_idx, end_idx in connections:
        # Ensure both keypoints exist within the detected keypoints array
        if start_idx < keypoint_count and end_idx < keypoint_count:
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # Check if both keypoints have a high enough confidence to be drawn
            if start_point[2] > 0.5 and end_point[2] > 0.5:
                start_x, start_y = int(start_point[0]), int(start_point[1])
                end_x, end_y = int(end_point[0]), int(end_point[1])

                # Assign the color based on the connection type (left, right, or center)
                if (start_idx, end_idx) == (17, 5):
                    color = colors['right_shoulderblade']
                elif (start_idx, end_idx) == (17, 6):
                    color = colors['left_shoulderblade']
                elif (start_idx, end_idx) == (6, 8):
                    color = colors['right_arm']
                elif (start_idx, end_idx) == (8, 10):
                    color = colors['right_forearm']
                elif (start_idx, end_idx) == (5, 7):
                    color = colors['left_arm']
                elif (start_idx, end_idx) == (7, 9):
                    color = colors['left_forearm']
                elif (start_idx, end_idx) == (17, 12):
                    color = colors['right_torso']
                elif (start_idx, end_idx) == (12, 14):
                    color = colors['right_upper_leg']
                elif (start_idx, end_idx) == (14, 16):
                    color = colors['right_lower_leg']
                elif (start_idx, end_idx) == (17, 11):
                    color = colors['left_torso']
                elif (start_idx, end_idx) == (11, 13):
                    color = colors['left_upper_leg']
                elif (start_idx, end_idx) == (13, 15):
                    color = colors['left_lower_leg']
                elif (start_idx, end_idx) == (17, 0):
                    color = colors['head']
                elif (start_idx, end_idx) == (0, 2):
                    color = colors['right_eyebrow']
                elif (start_idx, end_idx) == (2, 4):
                    color = colors['right_ear']
                elif (start_idx, end_idx) == (0, 1):
                    color = colors['left_eyebrow']
                elif (start_idx, end_idx) == (1, 3):
                    color = colors['left_ear']
                else:  # default fallback, just in case
                    color = colors['default']

                # Draw the line connecting the two keypoints
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)

    # Draw circles for the keypoints with their respective colors
    for i, keypoint in enumerate(keypoints):
        if keypoint[2] > 0.5:  # Only draw if the keypoint has a high enough confidence
            keypoint_x, keypoint_y = int(keypoint[0]), int(keypoint[1])

            # Use specific color for the keypoint
            keypoint_color = keypoint_colors.get(i, (153, 153, 153))  # Default to grey if not specified
            cv2.circle(frame, (keypoint_x, keypoint_y), 4, keypoint_color, -1)

    return frame
