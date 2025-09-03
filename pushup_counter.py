import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def _calculate_angle(a, b, c):
    """Helper function to calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

class PushupCounter:
    def __init__(self):
        self.counter = 0
        self.status = "up"
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.last_confidence = 0.5

    def reset(self):
        """Resets the counter and status."""
        self.counter = 0
        self.status = "up"

    def process_frame(self, frame, elbow_down_thresh, elbow_up_thresh, shoulder_down_thresh, shoulder_up_thresh, confidence_thresh):
        """Processes a single video frame to detect pose and count push-ups."""

        # Re-initialize the Pose model only if the confidence threshold has changed.
        if confidence_thresh != self.last_confidence:
            self.pose = mp_pose.Pose(min_detection_confidence=confidence_thresh, min_tracking_confidence=0.5)
            self.last_confidence = confidence_thresh

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip_l = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip_r = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            angle_l_elbow = _calculate_angle(shoulder_l, elbow_l, wrist_l)
            angle_r_elbow = _calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle_l_shoulder = _calculate_angle(elbow_l, shoulder_l, hip_l)
            angle_r_shoulder = _calculate_angle(elbow_r, shoulder_r, hip_r)

            avg_elbow_angle = (angle_l_elbow + angle_r_elbow) / 2
            avg_shoulder_angle = (angle_l_shoulder + angle_r_shoulder) / 2

            cv2.putText(image, f"{int(avg_elbow_angle)}", tuple(np.multiply(elbow_l, [w, h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"{int(avg_shoulder_angle)}", tuple(np.multiply(shoulder_l, [w, h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            is_down = avg_elbow_angle < elbow_down_thresh and avg_shoulder_angle > shoulder_down_thresh
            is_up = avg_elbow_angle > elbow_up_thresh and avg_shoulder_angle < shoulder_up_thresh

            if self.status == "up" and is_down:
                self.status = "down"
            elif self.status == "down" and is_up:
                self.status = "up"
                self.counter += 1

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        return image, self.counter, self.status
