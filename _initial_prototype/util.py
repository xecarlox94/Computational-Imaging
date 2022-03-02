import cv2 av




def process_image(i_name):
    img_org = cv.imread(i_name, cv.IMREAD_COLOR)

    # https://stackoverflow.com/questions/60352448/homography-from-football-soccer-field-lines
    hsv = cv.cvtColor(img_org, cv.COLOR_RGB2HSV)
    mask_green = cv.inRange(hsv, (36, 25, 25), (86, 255, 255))
    img_masked = cv.bitwise_and(img_org, img_org, mask=mask_green)

    img_gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(img_gray, 50, 200, apertureSize=3)


    # https://answers.opencv.org/question/222388/detect-ellipses-ovals-in-images-with-opencv-pythonsolved/



    lines = cv.HoughLinesP(canny,1,3.14/180,100,minLineLength=100,maxLineGap=10)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img_org,(x1,y1),(x2,y2),(0,255,0),2)

