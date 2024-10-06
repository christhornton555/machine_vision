import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_connections = self.mp_hands.HAND_CONNECTIONS

    def detect_hands(self, frame):
        """
        Detect hands in a frame using MediaPipe Hands.

        Args:
            frame (np.array): Input video frame.

        Returns:
            hand_landmarks: List of hand landmarks detected by MediaPipe Hands.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        return results.multi_hand_landmarks

    def draw_hands(self, frame, hand_landmarks):
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame (np.array): The video frame.
            hand_landmarks (list): List of hand landmarks from MediaPipe Hands.
        """
        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmark, self.hand_connections)
        return frame
