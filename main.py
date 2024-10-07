import cv2
import numpy as np
import argparse
import time
from collections import deque
from config.config import (
    select_device,
    BUFFER_SIZE,
    DETECTION_THRESHOLD,
    LOW_LIGHT_THRESHOLD_1,
    LOW_LIGHT_THRESHOLD_2,
    FRAME_BUFFER_SIZE,
    COOLDOWN_TIME,
    MIN_PERSON_SIZE,
    MAX_HANDS,
    MAX_FACES,
    OPENPOSE_CONNECTIONS,
    SKELETON_COLORS,
    KEYPOINT_COLORS,
    VIDEO_OUTPUT_FILENAME,
    FPS,
    RESCALE_VIDEO_OUTPUT,
    VIDEO_OUTPUT_RESOLUTION
)
from core.video_capture import get_video_stream
from core.detection import ObjectDetector
from core.hand_tracking import HandTracker
from core.face_tracking import FaceTracker
from core.postprocessing import apply_instance_mask, display_brightness, draw_skeleton
from core.camera_control import camera_settings
from core.postprocessing import calculate_brightness

def add_frames(*frames):
    """
    Add multiple frames by summing pixel values and clipping the result to [0, 255].
    """
    added_frame = np.sum(frames, axis=0)
    return np.clip(added_frame, 0, 255).astype(np.uint8)

def smooth_detections(detection_buffer):
    """
    Smooth detection results across the buffer by considering objects present in a majority of frames.
    """
    smoothed_results = None

    if len(detection_buffer) > 0:
        results_count = {}

        # Go through the buffer and count valid detections
        for results in detection_buffer:
            if results.boxes is not None and len(results.boxes.cls) > 0:
                for obj_class in results.boxes.cls.cpu().numpy():
                    obj_class = int(obj_class)
                    results_count[obj_class] = results_count.get(obj_class, 0) + 1

        # Filter detections that meet the threshold
        valid_results = [cls for cls, count in results_count.items() if count >= DETECTION_THRESHOLD]
        
        # Only keep the valid results in the smoothed_results list
        smoothed_results = [results for results in detection_buffer if len(results.boxes.cls) > 0 and int(results.boxes.cls.cpu().numpy()[0]) in valid_results]

    return smoothed_results[-1] if smoothed_results else None

def main(source, save_output):
    # Choose between CPU or GPU
    device = select_device(prefer_gpu=True)

    # Initialize video capture (from webcam or MP4)
    video_capture = get_video_stream(source=source)

    if source == 0:
        # Try to set manual focus for the camera
        camera_settings(video_capture, auto_focus=False, focus_value=255)

    # Get video dimensions from the input stream
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if save_output and RESCALE_VIDEO_OUTPUT:
        # Resize to a smaller resolution to reduce the file size (optional)
        target_width, target_height = VIDEO_OUTPUT_RESOLUTION[0], VIDEO_OUTPUT_RESOLUTION[1]
    elif save_output and not RESCALE_VIDEO_OUTPUT:
        target_width = frame_width
        target_height = frame_height

    # Initialize VideoWriter to save the output video
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec with .mp4 format
        video_writer = cv2.VideoWriter(VIDEO_OUTPUT_FILENAME, fourcc, FPS, (target_width, target_height))

    # Initialize the object detector with both segmentation and pose models
    segmentation_model_path = 'models/yolov8n-seg.pt'
    pose_model_path = 'models/yolov8n-pose.pt'
    detector = ObjectDetector(segmentation_model_path, pose_model_path, device)

    # Initialize the hand tracker
    hand_tracker = HandTracker(max_hands=MAX_HANDS)

    # Initialize the face tracker
    face_tracker = FaceTracker(max_faces=MAX_FACES)

    detection_buffer = deque(maxlen=BUFFER_SIZE)  # Initialize the detection buffer
    frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)  # Buffer to hold the three most recent frames

    # Initialize the cooldown timer for switching between brightness thresholds
    last_threshold_switch_time = time.time()
    current_threshold = None  # Track the current threshold level

    # Use a simple flag to track if the app has just started, to negate the risk of an empty frame buffer
    just_started = True

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame or end of video.")
            break

        # Calculate brightness
        brightness = calculate_brightness(frame)

        # Add the current frame to the frame buffer
        frame_buffer.append(frame)
        # Fill the buffer with this one frame on the first loop
        if just_started:
            for i in range(FRAME_BUFFER_SIZE - 1):
                frame_buffer.append(frame)
            just_started = False

        # Check how much time has passed since the last threshold switch
        time_since_last_switch = time.time() - last_threshold_switch_time

        # Select which light threshold to use, with cooldown to prevent rapid switching
        if time_since_last_switch > COOLDOWN_TIME:
            if brightness > LOW_LIGHT_THRESHOLD_1:
                current_threshold = None  # Use default threshold if brightness is above 65
            elif brightness < LOW_LIGHT_THRESHOLD_1 and brightness > LOW_LIGHT_THRESHOLD_2:
                current_threshold = "LOW_LIGHT_THRESHOLD_1"  # Update the current threshold
            elif brightness < LOW_LIGHT_THRESHOLD_2:
                current_threshold = "LOW_LIGHT_THRESHOLD_2"  # Update the current threshold
            last_threshold_switch_time = time.time()  # Reset the cooldown timer

        # Handle low-light conditions with different thresholds
        if current_threshold == "LOW_LIGHT_THRESHOLD_1":
            # If brightness is between LLT1 and LLT2, combine the current frame with the previous frame
            frame = add_frames(frame_buffer[-1], frame_buffer[-2])
        elif current_threshold == "LOW_LIGHT_THRESHOLD_2":
            # If brightness is below LLT2, combine the current frame with the previous two frames
            frame = add_frames(frame_buffer[-1], frame_buffer[-2], frame_buffer[-3])

        # Perform instance segmentation
        segmentation_results = detector.segment(frame)

        # Perform pose detection for skeleton tracking
        pose_results = detector.detect_pose(frame)

        # Detect hand landmarks
        hand_landmarks = hand_tracker.detect_hands(frame)

        # Detect face landmarks
        face_landmarks = face_tracker.detect_faces(frame)

        # Add current segmentation results to the buffer (only if valid results exist)
        if segmentation_results[0].boxes is not None and len(segmentation_results[0].boxes.cls) > 0:
            detection_buffer.append(segmentation_results[0])

        # Smooth detection results across frames
        smoothed_results = smooth_detections(detection_buffer)

        if smoothed_results and smoothed_results.masks is not None:
            # Get masks and classes for segmentation
            masks = smoothed_results.masks.data.cpu().numpy()  # Mask data
            classes = smoothed_results.names  # Class names
            class_ids = smoothed_results.boxes.cls.cpu().numpy()  # Detected class indices
            boxes = smoothed_results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates

            # Apply instance masks with different colors and labels
            frame = apply_instance_mask(frame, masks, class_ids, classes)

            # Process skeletons for detected people
            for i, class_id in enumerate(class_ids):
                if classes[class_id] == "person":
                    # Check if keypoints are available from the pose model
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints.data) > i:
                        # Get bounding box for the person
                        x1, y1, x2, y2 = boxes[i]

                        # Calculate the size of the bounding box relative to the frame size
                        box_area = (x2 - x1) * (y2 - y1)
                        frame_area = frame.shape[0] * frame.shape[1]
                        relative_size = box_area / frame_area

                        # Only track skeleton if person is sufficiently large in the frame
                        if relative_size > MIN_PERSON_SIZE and pose_results[0].keypoints.conf is not None:
                            # Extract keypoints: xy contains the coordinates, conf contains confidence values
                            keypoints_xy = pose_results[0].keypoints.xy[i].cpu().numpy()  # (x, y) coordinates
                            keypoints_conf = pose_results[0].keypoints.conf[i].cpu().numpy()  # confidence values

                            # Create a combined keypoints array (x, y, conf) for drawing
                            keypoints = np.hstack((keypoints_xy, keypoints_conf[:, np.newaxis]))

                            # Draw skeleton
                            frame = draw_skeleton(frame, keypoints, OPENPOSE_CONNECTIONS, SKELETON_COLORS, KEYPOINT_COLORS)
                    else:
                        print("No keypoints detected for this frame.")

        # Draw hand landmarks
        frame = hand_tracker.draw_hands(frame, hand_landmarks)

        # Draw face landmarks
        frame = face_tracker.draw_faces(frame, face_landmarks)

        # Calculate and display brightness
        frame = display_brightness(frame, brightness, current_threshold)

        # Write the frame to the video file
        if save_output and RESCALE_VIDEO_OUTPUT:
            # Resize to a smaller resolution to reduce the file size (optional)
            frame_resized = cv2.resize(frame, (target_width, target_height))
            video_writer.write(frame_resized)
        elif save_output and not RESCALE_VIDEO_OUTPUT:
            video_writer.write(frame)

        # Show the frame with instance segmentation, skeleton, and brightness applied
        cv2.imshow('YOLOv8 Segmentation and Pose Detection with Hand Tracking', frame)

        # Press 'q' to quit the video stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    video_capture.release()
    if save_output:
        video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse arguments for choosing the video source
    parser = argparse.ArgumentParser(description='YOLOv8 Segmentation and Pose Detection with Video/Camera')
    parser.add_argument('--video', type=str, default=None, help='Path to an MP4 video file. If not provided, webcam will be used.')
    parser.add_argument('--save-output', action='store_true', help='Flag to save the output to a video file.')
    args = parser.parse_args()

    # Use webcam (source=0) if no video file is provided, otherwise use the provided MP4 file
    video_source = 0 if args.video is None else args.video

    main(source=video_source, save_output=args.save_output)
