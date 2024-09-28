import uvc
import cv2

# List available devices
devices = uvc.device_list()
print(devices)

# Open the camera (usually the first device)
cap = uvc.Capture(devices[0]['uid'])

# Set manual focus
cap.set_focus_auto(0)  # Turn off auto-focus
cap.set_focus_abs(10)  # Set manual focus (range 0-255)

while True:
    frame = cap.get_frame()
    cv2.imshow('Manual Focus Control', frame.img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.close()
cv2.destroyAllWindows()
