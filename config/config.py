import torch

# Buffer size for temporal smoothing (number of frames to buffer)
BUFFER_SIZE = 5

# Detection threshold to consider an object valid (e.g., detected in 3 out of 5 frames)
DETECTION_THRESHOLD = 3

# Brightness thresholds for low-light conditions
LOW_LIGHT_THRESHOLD_1 = 65
LOW_LIGHT_THRESHOLD_2 = 30

# Frame buffer size (store the most recent 3 frames)
FRAME_BUFFER_SIZE = 3

# Cooldown time (in seconds) to prevent switching between brightness thresholds too quickly
COOLDOWN_TIME = 2

# Threshold for deciding if a person is large enough in the frame
MIN_PERSON_SIZE = 0.000001  # Fraction of the frame size (e.g., 20%)

# Maximum number of hands to detect
MAX_HANDS = 2

# Maximum number of faces to detect
MAX_FACES = 2

# OpenPose-like body parts
BODY_PARTS = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16
}

# OpenPose-like connections for limbs
OPENPOSE_CONNECTIONS = [
    # Neck placement will be calculated in postprocessing, then those connections made
    (BODY_PARTS['RShoulder'], BODY_PARTS['RElbow']),  # Right Shoulder to Right Elbow
    (BODY_PARTS['RElbow'], BODY_PARTS['RWrist']),  # Right Elbow to Right Wrist
    (BODY_PARTS['LShoulder'], BODY_PARTS['LElbow']),  # Left Shoulder to Left Elbow
    (BODY_PARTS['LElbow'], BODY_PARTS['LWrist']),  # Left Elbow to Left Wrist
    (BODY_PARTS['RHip'], BODY_PARTS['RKnee']),  # Right Hip to Right Knee
    (BODY_PARTS['RKnee'], BODY_PARTS['RAnkle']),  # Right Knee to Right Ankle
    (BODY_PARTS['LHip'], BODY_PARTS['LKnee']),  # Left Hip to Left Knee
    (BODY_PARTS['LKnee'], BODY_PARTS['LAnkle']),  # Left Knee to Left Ankle
    (BODY_PARTS['Nose'], BODY_PARTS['REye']),  # Nose to Right Eye
    (BODY_PARTS['REye'], BODY_PARTS['REar']),  # Right Eye to Right Ear
    (BODY_PARTS['Nose'], BODY_PARTS['LEye']),  # Nose to Left Eye
    (BODY_PARTS['LEye'], BODY_PARTS['LEar'])  # Left Eye to Left Ear
]

# Skeleton color scheme
SKELETON_COLORS = {
    'right_shoulderblade': (0, 0, 153),
    'left_shoulderblade': (0, 51, 153),
    'right_arm': (0, 102, 153),
    'right_forearm': (0, 153, 153),
    'left_arm': (0, 153, 102),
    'left_forearm': (0, 153, 51),
    'right_torso': (0, 153, 0),
    'right_upper_leg': (51, 153, 0),
    'right_lower_leg': (102, 153, 0),
    'left_torso': (153, 153, 0),
    'left_upper_leg': (153, 102, 0),
    'left_lower_leg': (153, 51, 0),
    'head': (153, 0, 0),
    'right_eyebrow': (153, 0, 51),
    'right_ear': (153, 0, 102),
    'left_eyebrow': (153, 0, 153),
    'left_ear': (102, 0, 153),
    'default': (153, 153, 153)  # fallback grey colour
}

KEYPOINT_COLORS = {
    BODY_PARTS['Nose']: (0, 0, 255),
    BODY_PARTS['LEye']: (255, 0, 255),
    BODY_PARTS['REye']: (255, 0, 170),
    BODY_PARTS['LEar']: (85, 0, 255),
    BODY_PARTS['REar']: (170, 0, 255),
    BODY_PARTS['LShoulder']: (0, 255, 85),
    BODY_PARTS['RShoulder']: (0, 170, 255),
    BODY_PARTS['LElbow']: (0, 255, 0),
    BODY_PARTS['RElbow']: (0, 255, 255),
    BODY_PARTS['LWrist']: (85, 255, 0),
    BODY_PARTS['RWrist']: (0, 255, 170),
    BODY_PARTS['LHip']: (255, 85, 0),
    BODY_PARTS['RHip']: (170, 255, 0),
    BODY_PARTS['LKnee']: (255, 0, 0),
    BODY_PARTS['RKnee']: (255, 255, 0),
    BODY_PARTS['LAnkle']: (255, 0, 85),
    BODY_PARTS['RAnkle']: (255, 170, 0),
    17: (0, 85, 255),  # Neck keypoint is calculated, to mimic the OpenPose model
}

# Video output filename and codec
VIDEO_OUTPUT_FILENAME = 'output/output_video.mp4'
FPS = 25  # Desired framerate of the output video
RESCALE_VIDEO_OUTPUT = False  # Change resolution of output video?
VIDEO_OUTPUT_RESOLUTION = [640, 480]  # Resolution to use if above value set to True


def select_device(prefer_gpu=True):
    """
    Select the device to use for processing: GPU (if available) or CPU.

    Args:
        prefer_gpu (bool): If True, will prefer GPU if available. If False, always use CPU.

    Returns:
        torch.device: The device (CPU or GPU) to be used.
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Device selected: GPU (CUDA)")
    else:
        device = torch.device('cpu')
        print("Device selected: CPU")

    return device
