import cv2 as cv

from model_3d import utils

cap = cv.VideoCapture('football.mp4')


"""
argmax = lambda l: max(range(len(l)), key=(lambda i: l[i]))



whT = 320


confThreshold = 0.4
nmsThreshold = 0.2

classesFile = 'coco.names'
classNames = []
with open(classesFile, 'r') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def findObjects(frame):

    blob = cv.dnn.blobFromImage(frame, 1/255, (whT, whT), [0,0,0,0], crop=False)

    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(outputNames)


    hT, wT, cT = frame.shape

    bbox = []
    classIds = []
    confs = []

    max_ball_conf = 0
    ball = False


    def get_dimensions(detection, wT, hT):

        w = int(detection[2] * wT)
        h = int(detection[3] * hT)

        x = int( (detection[0] * wT) - (w / 2) )
        y = int( (detection[1] * hT) - (h / 2) )

        return x, y, w, h


    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = argmax(scores)
            confidence = scores[classId]


            class_name = classNames[classId]

            if class_name == "person" and confidence > confThreshold:

                bbox.append(list(get_dimensions(det, wT, hT)))

                classIds.append(classId)
                confs.append(float(confidence))

            elif class_name == "sports ball" and confidence > max_ball_conf:

                max_ball_conf = confidence

                ball = tuple(get_dimensions(det, wT, hT))
                print("sports ball")


    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    objs = []

    for i in indices:
        box = bbox[i]
        cl = classIds[i]
        cf = bbox[i]

        objs.append((box, cf, cl))

        draw_boundingbox(frame, tuple(box), (255, 0, 255),
                f'{classNames[classIds[i]]} {int(confs[i] * 100)}%')

    return objs, ball


objs = []

ball = False
ball_tracker = cv.legacy.TrackerCSRT_create()

def label_ball():
    global ball_tracker
    ball_tracker = cv.legacy.TrackerCSRT_create()

    bbox = cv.selectROI(window_name, frame, False)

    def f():
        for i in bbox:
            if i != 0: return False
        return True

    if not f():
        ball_tracker.init(frame, bbox)




window_name = "Image"
frame_count = -1
count = 0



def draw_boundingbox(frame, dimensions, colour, title):
    x, y, w, h = dimensions
    cv.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
    cv.putText(frame, title, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)



wait_key = lambda key: cv.waitKey(25) & 0xFF == ord(key)

"""













"""
# to add to opencv program and to blender rendering obj
def process_image(i_name):
    img_org = cv.imread(i_name, cv.IMREAD_COLOR)
    # https://stackoverflow.com/questions/60352448/homography-from-football-soccer-field-lines
    hsv = cv.cvtColor(img_org, cv.COLOR_RGB2HSV)
    mask_green = cv.inRange(hsv, (36, 25, 25), (86, 255, 255))
    img_masked = cv.bitwise_and(img_org, img_org, mask=mask_green)
    img_gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(img_gray, 50, 200, apertureSize=3)
"""



import tensorflow as tf
import numpy as np


def get_camera_data_prediction(model, image):
    return list(model.predict(
        image
    )[0])


get_model = lambda m: tf.keras.models.load_model('./models/' + m)


get_image_input = lambda frame: np.array([cv.resize(
    cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    ),
    (256, 256)
)])




"""
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
"""


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




while cap.isOpened():

    ret, frame = cap.read()

    pred = get_frame_prediction(frame)

    #print(pred)
    #print(len(list(pred)))

    dec_data = utils.decode_camera_data(pred)

    print(dec_data[2])

    print(utils.get_pitch_corner_vecs(dec_data))













    break


"""

    frame_count = frame_count + 1
    if frame_count < 0: continue


    perform_detection = count % 30 == 0


    if ret:

        if wait_key('q'): # quit
          break


        if wait_key('f'): # fix ball tracking
            print("KEY f PRESSED!!!!")
            label_ball()


        if perform_detection:

            objs, ball = findObjects(frame)


            if len(objs) > 0:
                mtracker = cv.legacy.MultiTracker_create()

                for obj in objs:
                    mtracker.add(cv.legacy.TrackerCSRT_create(), frame, tuple(obj[0]))

                count = count + 1

            else:
                print("No objects detected")


            if ball:
                ball_tracker.init(frame, ball)

            else:
                label_ball()


        else:

            is_tracking, bboxes = mtracker.update(frame)

            if is_tracking:
                for i, bbox in enumerate(bboxes):
                    draw_boundingbox(frame, tuple(map(lambda x: int(x), bbox)), (0, 255, 0), "person")

                count = count + 1
            else:
                count = 0

            is_tracking, ball = ball_tracker.update(frame)

            if is_tracking:
                draw_boundingbox(frame, tuple(map(lambda x: int(x), ball)), (255, 255, 0), "ball")

            else:
                label_ball()


        cv.imshow(window_name, frame)

    else:
        print("end video stream")
        break

cap.release()
cv.destroyAllWindows()

"""
