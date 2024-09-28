import cv2

# Load Haar Cascades face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def pixelate_region(image, x1, y1, x2, y2, pixelation_level=20):
    """Apply pixelation to a region of the image (face)."""
    # Extract the region of interest (ROI)
    face_region = image[y1:y2, x1:x2]
    
    # Determine the size of the pixel blocks (downscale)
    h, w = face_region.shape[:2]
    temp = cv2.resize(face_region, (pixelation_level, pixelation_level), interpolation=cv2.INTER_LINEAR)
    
    # Upscale back to the original size
    pixelated_face = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Replace the pixelated face in the original image
    image[y1:y2, x1:x2] = pixelated_face
    return image

# Open webcam stream (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar Cascades
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Apply pixelation to each detected face
    for (x, y, w, h) in faces:
        frame = pixelate_region(frame, x, y, x + w, y + h, pixelation_level=20)

    # Display the resulting frame with pixelated faces
    cv2.imshow('Real-time Face Pixelation', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
