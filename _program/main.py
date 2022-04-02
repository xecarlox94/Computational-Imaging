import cv2 as cv



"""
import tensorflow as tf
import numpy as np



def get_camera_data_prediction(model, image):
    return list(model.predict(
        image
    )[0])


get_model = lambda m: tf.keras.models.load_model('./models/' + m)


get_image_input = lambda frame: np.array([
    cv.resize(
        cv.cvtColor(
            frame,
            cv.COLOR_BGR2GRAY
        ),
        (256, 256)
    )
])


def get_frame_prediction(frame):
    def get_model_pred(model_names, X):
        print(model_names)
        if model_names == []:
            return X

        pred = get_camera_data_prediction(
            get_model(model_names[0]),
            [
                get_image_input(frame),
                (
                    [] if X == [] else np.array([X])
                )
            ]
        )

        get_model_pred(
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


def get_frame_prediction(frame):

    pred = get_camera_data_prediction(
        get_model("cam_origin_vec"),
        [get_image_input(frame), []]
    )

    X = pred + []

    pred = get_camera_data_prediction(
        get_model("frame_vectors"),
        [get_image_input(frame), np.array([X])]
    )

    X = X + pred

    pred = get_camera_data_prediction(
        get_model("pitch_corner_vecs"),
        [get_image_input(frame), np.array([X])]
    )

    X = X + pred

    pred = get_camera_data_prediction(
        get_model("pitch_vectors"),
        [get_image_input(frame), np.array([X])]
    )

    X = X + pred

    return X
"""

from model_3d.opencv_utils import draw_boundingbox, label_ball







from yolo import findObjects

from model_3d import utils


cap = cv.VideoCapture('football.mp4')





objs = []
ball = False
ball_tracker = cv.legacy.TrackerCSRT_create()


frame_count = -1
count = 0





# to add to opencv program and to blender rendering obj
def process_image(img_org):
    # https://stackoverflow.com/questions/60352448/homography-from-football-soccer-field-lines
    hsv = cv.cvtColor(img_org, cv.COLOR_RGB2HSV)
    mask_green = cv.inRange(hsv, (36, 25, 25), (86, 255, 255))
    img_masked = cv.bitwise_and(img_org, img_org, mask=mask_green)
    img_gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(img_gray, 50, 200, apertureSize=3)

    cv.imshow("canny", canny)





while cap.isOpened():


    ret, frame = cap.read()


    process_image(frame)


    """
    frame_count = frame_count + 1
    if frame_count < 0000: continue


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


    pred = get_frame_prediction(frame)

    #print(pred)
    #print(len(list(pred)))

    dec_data = utils.decode_camera_data(pred)

    print(dec_data[2])

    print(utils.get_pitch_corner_vecs(dec_data))
    """





cap.release()
cv.destroyAllWindows()

