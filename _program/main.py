import cv2 as cv

from yolo import findObjects

from model_3d.opencv_utils import draw_boundingbox, label_ball, get_pitch_recognition_img
from model_3d import utils

import tensorflow as tf
import numpy as np




def get_camera_data_prediction(model, image):
    return list(model.predict(
        image
    )[0])


get_model = lambda m: tf.keras.models.load_model('./models/' + m)



def get_frame_prediction(frame):
    def get_model_pred(model_names, X):
        if model_names == []:
            return X

        pred = get_camera_data_prediction(
            get_model(model_names[0]),
            [
                get_pitch_recognition_img(frame),
                (
                    [] if X == [] else np.array([X])
                )
            ]
        )

        return get_model_pred(
            model_names[1:],
            X + pred
        )

    return get_model_pred(
        [
            "cam_origin_vec",
            "frame_vectors",
            "pitch_corner_vecs",
            "pitch_vectors"
        ],
        []
    )




objs = []
ball = False
ball_tracker = cv.legacy.TrackerCSRT_create()

frame_count = -1
count = 0


cap = cv.VideoCapture('football.mp4')

while cap.isOpened():


    ret, frame = cap.read()


    pred = get_frame_prediction(
        frame
    )

    print(pred)


    dec_data = utils.decode_camera_data(pred)

    print(dec_data)

    pitch_corner_vecs = utils.get_pitch_corner_vecs(dec_data)

    print(pitch_corner_vecs)



    """
    frame_count = frame_count + 1
    if frame_count < 1000: continue


    window_name = "Image"
    perform_detection = count % 30 == 0

    if ret:


        wait_key = lambda key: cv.waitKey(25) & 0xFF == ord(key)



        if wait_key('q'): # quit
          break


        if wait_key('f'): # fix ball tracking
            print("KEY f PRESSED!!!!")
            ball_tracker = label_ball(window_name, frame)


        if perform_detection:

            objs, ball = findObjects(frame)


            if len(objs) > 0:
                mtracker = cv.legacy.MultiTracker_create()

                for obj in objs:
                    mtracker.add(
                        cv.legacy.TrackerCSRT_create(),
                        frame,
                        tuple(obj[0])
                    )

                count = count + 1

            else:
                print("No objects detected")


            if ball:
                ball_tracker.init(frame, ball)

            else:
                ball_tracker = label_ball(window_name, frame)


        else:

            is_tracking, bboxes = mtracker.update(frame)

            if is_tracking:
                for i, bbox in enumerate(bboxes):
                    draw_boundingbox(
                        frame,
                        tuple(
                            map(
                                lambda x: int(x),
                                bbox
                            )
                        ),
                        (0, 255, 0),
                        "person"
                    )

                count = count + 1
            else:
                count = 0

            is_tracking, ball = ball_tracker.update(frame)

            if is_tracking:
                draw_boundingbox(
                    frame,
                    tuple(
                        map(
                            lambda x: int(x),
                            ball
                        )
                    ),
                    (255, 255, 0),
                    "ball"
                )

            else:
                ball_tracker = label_ball(window_name, frame)


        cv.imshow(window_name, frame)

    else:
        print("end video stream")
        break

    """




cap.release()
cv.destroyAllWindows()

