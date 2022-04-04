import cv2 as cv
import tensorflow as tf
import numpy as np


from model_3d.opencv_utils import draw_boundingbox, label_ball, get_pitch_recognition_img
from model_3d import utils

from yolo import findObjects

import geometry as gmt




def get_frame_prediction(frame):

    def get_camera_data_prediction(model, image):
        return list(model.predict(
            image
        )[0])

    get_model = lambda m: tf.keras.models.load_model('./models/' + m)

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
            ball_tracker = label_ball(
                window_name,
                frame
            )


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
                ball_tracker = label_ball(
                    window_name,
                    frame
                )


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
                ball_tracker = label_ball(
                    window_name,
                    frame
                )


        cv.imshow(window_name, frame)

    else:
        print("end video stream")
        break


    pred = get_frame_prediction(
        frame
    )

    print(pred)

    cam_origin, frames_vectors, pitch_vectors = utils.decode_camera_data(pred)
    print("\n\n\n\n")
    #print(cam_origin)
    #print("\n")
    #print(frames_vectors)
    #print("\n")
    #print(pitch_vectors)
    #print("\n")

    pitch_corner_vecs = utils.get_pitch_corner_vecs(pitch_vectors)
    #print(pitch_corner_vecs)
    #print("\n")


    frames_pitch_int_vecs = list(map(
        lambda v: gmt.get_pitch_intersection(
            cam_origin,
            v
        ),
        frames_vectors
    ))
    #print(frames_pitch_int_vecs)


    frames_pitch_int_vecs = [
        ( -3, -3,  0),
        ( -3,  2,  0),
        (  2,  2,  0),
        (  2, -3,  0),
    ]
    pitch_corner_vecs = [
        ( -2, -2,  0),
        ( -2,  3,  0),
        (  3,  3,  0),
        (  3, -2,  0),
    ]

    inner_section_vecs = gmt.get_inner_section(
        frames_pitch_int_vecs,
        pitch_corner_vecs
    )
    print(inner_section_vecs)


    frame_intersection_vecs = list(map(
        lambda v: gmt.get_screen_intersection(
            cam_origin,
            frames_vectors,
            v
        ),
        inner_section_vecs
    ))
    print(frame_intersection_vecs)




cap.release()
cv.destroyAllWindows()

