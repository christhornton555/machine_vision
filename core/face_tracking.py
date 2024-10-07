import mediapipe as mp
import cv2

class FaceTracker:
    def __init__(self, max_faces=2, detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_faces(self, frame):
        """
        Detect face landmarks in a frame using MediaPipe Face Mesh.

        Args:
            frame (np.array): Input video frame.

        Returns:
            face_landmarks: List of face landmarks detected by MediaPipe Face Mesh.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        return results.multi_face_landmarks

    def draw_faces(self, frame, face_landmarks):
        """
        Draw face landmarks and connections on the frame.

        Args:
            frame (np.array): The video frame.
            face_landmarks (list): List of face landmarks from MediaPipe Face Mesh.
        """
        if face_landmarks:
            for face_landmark in face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmark,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,  # Use default for landmarks
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()  # Use default tesselation style
                )
                # Draw the contours as well
                self.mp_draw.draw_landmarks(
                    frame,
                    face_landmark,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,  # Use default for landmarks
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()  # Use default contour style
                )
        return frame
