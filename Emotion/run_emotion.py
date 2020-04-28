import pickle
import time

import cv2
import numpy as np
from frames import FrameHandler

if __name__ == "__main__":
    emotions = ['anger', 'contempt', 'disgust', 'fear',
                'happy', 'neutral', 'sadness', 'surprise']

    with open('../Models/trained_svm_model', 'rb') as f:
        model = pickle.load(f)

    video_input = cv2.VideoCapture('../data/Smiling.mp4')
    cat, frame = video_input.read()

    while cat:
        start = time.time()
        handler = FrameHandler(frame)

        # if args.landmarks:
        #     handler.draw_landmarks()

        faces = np.array([handler.get_vectorized_landmarks()])
        if faces[0] is not None:
            prediction_time = time.time()
            prediction = model.predict(faces)
            # print("predict_time:", time.time() - prediction_time)

            if len(prediction) > 0:
                text = emotions[prediction[0]]
                if text != "disgust":
                    print(text)
                cv2.putText(handler.frame, text, (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            thickness=2)

        cv2.imshow('image', handler.frame)

        if cv2.waitKey(1) == 27:
            break
        fps = 1 / (time.time() - start)
        # print(int(fps))
        # We get a new frame from the video_input
        cat, frame = video_input.read()

    cv2.destroyAllWindows()
    video_input.release()
