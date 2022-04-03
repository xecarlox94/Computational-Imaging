import cv2 as cv

import numpy as np


def get_pitch_recognition_img(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_RGB2HSV)
    final_img = cv.resize(
        cv.cvtColor(
            cv.bitwise_and(
                hsv,
                frame,
                mask=cv.inRange(
                    hsv,
                    (36, 25, 25),
                    (86, 255, 255)
                )
            ),
            cv.COLOR_BGR2GRAY
        ),
        (256, 256)
    )
    return np.array(final_img).reshape(
        1,
        final_img.shape[0],
        final_img.shape[1],
        1
    )

def draw_boundingbox(frame, dimensions, colour, title):
    x, y, w, h = dimensions

    cv.rectangle(
        frame,
        (x, y),
        (x+w, y+h),
        colour,
        2
    )

    """
    cv.putText(
        frame,
        title,
        (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        2
    )
    """


def label_ball(window_name, frame):
    ball_tracker = cv.legacy.TrackerCSRT_create()

    bbox = cv.selectROI(window_name, frame, False)

    def f():
        for i in bbox:
            if i != 0: return False
        return True

    if not f():
        ball_tracker.init(frame, bbox)

    return ball_tracker


