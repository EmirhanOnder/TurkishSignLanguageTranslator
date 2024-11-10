import cv2
import numpy as np
import os
import mediapipe as mp

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


base_path = "C:\\Users\\emirh\\Downloads"
video_folders = ['train', 'val', 'test']

def mediapipe_detection(image, model):
    """
    Performs sign detection on an image using a Mediapipe model.
    Args:
        image (numpy.ndarray): The input image in BGR format.
        model: The Mediapipe model instance.
    Returns:
        Tuple: The processed image in BGR format and detection results from the model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draws styled landmark connections on an image with different colors.
    Args:
        image (numpy.ndarray): The input image in BGR format.
        results: Detection results from the Mediapipe model.
    """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(0, 153, 204), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(102, 204, 255), thickness=1, circle_radius=1)
                              )

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 153, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=2)
                              )

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(153, 0, 153), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(204, 102, 255), thickness=2, circle_radius=2)
                              )

    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255, 153, 204), thickness=2, circle_radius=2)
                              )

def extract_keypoints(results):
    """
    Extracts keypoints from Mediapipe detection results and returns them as a single array.
    Args:
        results: Detection results from the Mediapipe model.
    Returns:
        numpy.ndarray: Flattened and concatenated array of pose, face, left hand, and right hand keypoints.
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    left_hand = np.array([[res.x, res.y, res.z] for res in
                          results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)
    right_hand = np.array([[res.x, res.y, res.z] for res in
                           results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, left_hand, right_hand])


for folder in video_folders:
    folder_path = os.path.join(base_path, folder)
    output_folder = os.path.join(base_path, folder + '_keypoints')

    os.makedirs(output_folder, exist_ok=True)

    for video_name in os.listdir(folder_path):
        if video_name.endswith('_color.mp4'):
            video_path = os.path.join(folder_path, video_name)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Error: Video {video_name} could not be opened.")
                continue

            # Store keypoints for each frame
            keypoints_list = []
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Detection and extract keypoints
                    image, results = mediapipe_detection(frame, holistic)
                    keypoints = extract_keypoints(results)
                    keypoints_list.append(keypoints)

            # Convert to numpy array and save
            keypoints_array = np.array(keypoints_list)
            save_path = os.path.join(output_folder, f"{os.path.splitext(video_name)[0]}_keypoints.npy")
            np.save(save_path, keypoints_array)
            print(f"Saved keypoints for {video_name} to {save_path}")

            cap.release()
cv2.destroyAllWindows()
