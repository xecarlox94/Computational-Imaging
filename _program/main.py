import cv2 as cv


"""
cap = cv.VideoCapture('football.mp4')



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



import tensorflow as tf
import numpy as np


def get_camera_data_prediction(model, image):
    return list(model.predict(
        np.array([image])
    )[0])


#model = tf.keras.models.load_model('./my_model')


get_image_input = lambda frame: cv.resize(
    cv.cvtColor(
        frame,
        cv.COLOR_BGR2GRAY
    ),
    (256, 256)
)
"""


import csv

#from model_3d.generate_dataset import *
from model_3d import utils


def read_csv_data(f_name):
    with open(f_name) as f:
        reader = csv.reader(f)

        return utils.decode_camera_data(list(map(
                lambda dec_row: list(map(
                    lambda r: float(r),
                    dec_row[1:]
                )),
                reader
        ))[0])






cam_origin, frames_vectors, pitch_vectors = read_csv_data('./model_3d/dataset/data.csv')
pitch_corners = utils.get_pitch_corners(pitch_vectors)



#print(cam_origin)
#print(frames_vectors)
#print(pitch_vectors)
#print(pitch_corners)


screen_centroid = utils.get_3d_centroid(frames_vectors)






from Geometry3D import *

get_point = lambda v: Point(v[0], v[1], v[2])


def get_screen_intersection(point):
    return get_intersection(
        Plane(
            get_point(frames_vectors[0]),
            get_point(frames_vectors[1]),
            get_point(frames_vectors[2]),
        ),
        point
    )

def get_pitch_intersection(point):
    return get_intersection(
        Plane(origin(),Vector(0,0,1)),
        point
    )


def get_intersection(plane, point):
    return tuple(intersection(
        Line(
            get_point(cam_origin),
            get_point(point)
        ),
        plane
    ))

print(get_screen_intersection(
    pitch_vectors[19]
))

"""
print(get_pitch_intersection(
    frames_vectors[3]
))
"""



frames_pitch_vectors = list(map(
    get_pitch_intersection,
    frames_vectors
))


#print(frames_pitch_vectors)







from shapely.geometry import Polygon, mapping

def get_inner_section(y, x):
    get_vector2d_list = lambda lst: list(map(
        lambda v: (v[0], v[1]),
        lst
    ))

    x = Polygon(get_vector2d_list(x))
    y = Polygon(get_vector2d_list(y))

    intersection = x.intersection(y)
    if intersection.is_empty:
        return []
    else:
        return list(mapping(
            intersection
        )['coordinates'][0])


inner_section = get_inner_section(pitch_corners, frames_pitch_vectors)

#print(inner_section)








"""
while cap.isOpened():

    ret, frame = cap.read()


    pred = get_camera_data_prediction(
        model,
        get_image_input(frame)
    )


    camera_data = utils.decode_camera_data(pred)

    print(frame_count)

    frame_count = frame_count + 1
    if frame_count < 0:
        continue


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
