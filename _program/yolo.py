import cv2 as cv

from model_3d.opencv_utils import draw_boundingbox






def findObjects(frame):

    whT = 320

    confThreshold = 0.4
    nmsThreshold = 0.2

    classesFile = './yolo/coco.names'
    classNames = []
    with open(classesFile, 'r') as f:
        classNames = f.read().rstrip('\n').split('\n')

    modelConfiguration = './yolo/yolov3.cfg'
    modelWeights = './yolo/yolov3.weights'

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

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


    argmax = lambda l: max(range(len(l)), key=(lambda i: l[i]))


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


    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    objects = []

    for i in indices:
        box = bbox[i]
        cl = classIds[i]
        cf = bbox[i]

        objects.append((box, cf, cl))

        draw_boundingbox(frame, tuple(box), (255, 0, 255),
                f'{classNames[classIds[i]]} {int(confs[i] * 100)}%')

    return objects, ball

