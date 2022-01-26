import cv2 as cv

cap = cv.VideoCapture('football.mp4')


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

argmax = lambda l: max(range(len(l)), key=(lambda i: l[i]))



wait_key = lambda key: cv.waitKey(25) & 0xFF == ord(key)



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

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = argmax(scores)
            confidence = scores[classId]


            class_name = classNames[classId]

            if class_name == "person" and confidence > confThreshold:

                w = int(det[2] * wT)
                h = int(det[3] * hT)

                x = int( (det[0] * wT) - (w / 2) )
                y = int( (det[1] * hT) - (h / 2) )

                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

            elif class_name == "sports ball" and confidence > max_ball_conf:

                max_ball_conf = confidence

                w = int(det[2] * wT)
                h = int(det[3] * hT)

                x = int( (det[0] * wT) - (w / 2) )
                y = int( (det[1] * hT) - (h / 2) )

                ball = x, y, w, h
                print("sports ball")


    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    objs = []

    for i in indices:
        box = bbox[i]
        cl = classIds[i]
        cf = bbox[i]

        objs.append((box, cf, cl))

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_boundingbox(frame, (x, y, w, h), (255, 0, 255),
                f'{classNames[classIds[i]]} {int(confs[i] * 100)}%')

    return objs, ball

window_name = "Image"
frame_count = -1
count = 0

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

    if not f(): ball_tracker.init(frame, bbox)



def draw_boundingbox(frame, dimensions, colour, title):
    x, y, w, h = dimensions
    cv.rectangle(frame, (x, y), (x+w, y+h), colour, 2)
    cv.putText(frame, title, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)



while cap.isOpened():

    ret, frame = cap.read()


    frame_count = frame_count + 1
    if frame_count < 1500:
        print(frame_count)
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

            if not ball:
                print("NO BALL FOUND")


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
