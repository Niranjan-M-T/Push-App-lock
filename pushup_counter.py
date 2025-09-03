import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for the counter and status
counter = 0
status = "up"
# The pose model will be initialized in the process_frame function
# to allow the confidence threshold to be adjusted dynamically.
pose = None

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def process_frame(frame, elbow_down_thresh, elbow_up_thresh, shoulder_down_thresh, shoulder_up_thresh, confidence_thresh):
    """Processes a single video frame to detect pose and count push-ups."""
    global counter, status, pose

    # Initialize the Pose model with the dynamic confidence threshold.
    # This is done in each frame to ensure the setting from the GUI is always used.
    pose = mp_pose.Pose(min_detection_confidence=confidence_thresh, min_tracking_confidence=0.5)

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image and find pose.
    results = pose.process(image)

    # Convert the image back to BGR.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    try:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for both sides of the body
        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        # Calculate angles
        angle_l_elbow = calculate_angle(shoulder_l, elbow_l, wrist_l)
        angle_r_elbow = calculate_angle(shoulder_r, elbow_r, wrist_r)
        angle_l_shoulder = calculate_angle(elbow_l, shoulder_l, hip_l)
        angle_r_shoulder = calculate_angle(elbow_r, shoulder_r, hip_r)

        # Use the average angle of both sides for stability
        avg_elbow_angle = (angle_l_elbow + angle_r_elbow) / 2
        avg_shoulder_angle = (angle_l_shoulder + angle_r_shoulder) / 2

        # Visualize angles on the image
        cv2.putText(image, f"{int(avg_elbow_angle)}", tuple(np.multiply(elbow_l, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{int(avg_shoulder_angle)}", tuple(np.multiply(shoulder_l, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Push-up counting logic based on the user's specifications
        # Down Position: Elbow < threshold AND Shoulder > threshold
        is_down = avg_elbow_angle < elbow_down_thresh and avg_shoulder_angle > shoulder_down_thresh
        # Up Position: Elbow > threshold AND Shoulder < threshold
        is_up = avg_elbow_angle > elbow_up_thresh and avg_shoulder_angle < shoulder_up_thresh

        if status == "up" and is_down:
            status = "down"
        elif status == "down" and is_up:
            status = "up"
            counter += 1

    except:
        # If landmarks are not detected, do nothing
        pass

    # Draw the pose annotation on the image.
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    return image, counter, status
