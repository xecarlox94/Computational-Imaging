import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def plot_image(img):
    plt.figure(figsize=(20,15))
    plt.imshow(img, aspect='auto')



def process_image(i_name):
    img_org = cv.imread(i_name, cv.IMREAD_COLOR)

    # https://stackoverflow.com/questions/60352448/homography-from-football-soccer-field-lines
    hsv = cv.cvtColor(img_org, cv.COLOR_RGB2HSV)
    mask_green = cv.inRange(hsv, (36, 25, 25), (86, 255, 255))
    img_masked = cv.bitwise_and(img_org, img_org, mask=mask_green)

    img_gray = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(img_gray, 50, 200, apertureSize=3)


    # https://answers.opencv.org/question/222388/detect-ellipses-ovals-in-images-with-opencv-pythonsolved/

    # https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    # circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1.2, 200)
    # if circles is not None:
        # circles = np.round(circles[0, :]).astype("int")
        # for (x, y, r) in circles:
            # cv.circle(img_org, (x, y), r, (0, 255, 0), 4)
            # cv.rectangle(img_org, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


    lines = cv.HoughLinesP(canny,1,np.pi/180,100,minLineLength=100,maxLineGap=10)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(img_org,(x1,y1),(x2,y2),(0,255,0),2)


    plot_image(img_org)

    plot_image(canny)
    
    plot_image(img_masked)

"""
    plot_image(hsv)

    plot_image(img_gray)
"""

