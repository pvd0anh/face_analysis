from __future__ import division

import os

import cv2
import dlib
import numpy as np

from .calibration import Calibration
from .eye import Eye


class FatigueTracking(object):
    """
    This class tracks the user's Fatigue.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.landmarks = None
        self.flag = True

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "../../Models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_detector(frame)

        try:
            self.landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, self.landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, self.landmarks, 1, self.calibration)

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the Fatigue. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the Fatigue. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def yawn_detection(self, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (self.landmarks.part(i).x, self.landmarks.part(i).y)

        top_lips = []
        bottom_lips = []
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (self.landmarks.part(i).x, self.landmarks.part(i).y)

            if 50 <= i <= 53 or 61 <= i <= 64:
                top_lips.append(coords[i])

            elif 65 <= i <= 68 or 56 <= i <= 59:
                bottom_lips.append(coords[i])
        mouth_wid = coords[55] - coords[49]
        toplipsall = np.squeeze(np.asarray(top_lips))
        bottomlipsall = np.squeeze(np.asarray(bottom_lips))
        top_lips_mean = np.array(np.mean(toplipsall, axis=0), dtype=dtype)
        bottom_lips_mean = np.array(np.mean(bottomlipsall, axis=0), dtype=dtype)
        top_lips_mean = top_lips_mean.reshape(-1)
        bottom_lips_mean = bottom_lips_mean.reshape(-1)

        # distance=math.sqrt((bottom_lips_mean[0] - top_lips_mean[0])**2 + (bottom_lips_mean[-1] - top_lips_mean[-1])**2)
        distance1 = bottom_lips_mean[-1] - top_lips_mean[-1]
        radio = distance1 * 1.0 / mouth_wid[0]
        # print(bottom_lips_mean[-1], top_lips_mean[-1])
        yawn = False

        if (radio > 0.72):
            # print("yawn ratio", radio)
            self.flag += 1
        if (self.flag > 10):
            yawn = True
            self.flag = 0

        return yawn

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame
