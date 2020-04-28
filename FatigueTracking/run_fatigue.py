import time

import cv2
from fatigue_tracking import FatigueTracking

fatigue = FatigueTracking()
video_input = cv2.VideoCapture('/data/Smiling.mp4')
cat, frame = video_input.read()
fps = 0
while cat:

    start = time.time()

    # We send this frame to FatigueTracking to analyze it
    fatigue.refresh(frame)
    frame = fatigue.annotated_frame()
    text = ""

    if fatigue.is_blinking():
        text = "Blinking"
    elif fatigue.is_right():
        text = "Looking right"
    elif fatigue.is_left():
        text = "Looking left"
    elif fatigue.is_center():
        text = "Looking center"
    if fatigue.yawn_detection():
        yawn = "Yawning"
        print(yawn)
    else:
        yawn = ""

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, yawn, (90, 100), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, "fps: {}".format(int(fps)), (90, 150), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = fatigue.pupil_left_coords()
    right_pupil = fatigue.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 190), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 225), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
    fps = 1 / (time.time() - start)
    # We get a new frame from the video_input
    cat, frame = video_input.read()

cv2.destroyAllWindows()
video_input.release()
