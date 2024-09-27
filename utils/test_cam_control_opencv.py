import cv2

# Initialize video capture for the Logitech C920 (assuming source 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Set manual focus (0: auto-focus off, 1: auto-focus on)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Turn off auto-focus

# Set focus manually (value depends on the camera)
focus_value = 0  # Adjust this value between 0 (near) and 255 (far)
cap.set(cv2.CAP_PROP_FOCUS, focus_value)

# Other capture settings
cap.set(cv2.CAP_PROP_FPS, 30.0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('m','j','p','g'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Main loop to capture frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame
    cv2.imshow('Camera Feed with Manual Focus', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
