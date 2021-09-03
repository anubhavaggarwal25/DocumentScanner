import cv2 as cv
import numpy as np
import imutils

img = cv.imread('img3.jpg')
rows = img.shape[0]
coloumns = img.shape[1]

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurr = cv.GaussianBlur(gray, (5, 5), 0)

edge = cv.Canny(blurr, 80, 255)

contours, hierarchy = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv.contourArea, reverse = True)

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv.contourArea(i)
        if area > 5000:
            peri = cv.arcLength(i, True)
            approx = cv.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area

biggest, max_area = biggestContour(contours)

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

biggest = reorder(biggest)

cv.drawContours(img, biggest, -1, (0, 255, 0), 20)

cv.drawContours()
actualArea = np.float32(biggest)
expectedArea = np.float32([[0, 0], [coloumns, 0], [0, rows], [coloumns, rows]])
prepTransform = cv.getPerspectiveTransform(actualArea, expectedArea)
imgTransform = cv.warpPerspective(img, prepTransform, (coloumns, rows))

grayImgTransform = cv.cvtColor(imgTransform, cv.COLOR_BGR2GRAY)

se=cv.getStructuringElement(cv.MORPH_RECT , (8,8))
bg=cv.morphologyEx(grayImgTransform, cv.MORPH_DILATE, se)
out_gray=cv.divide(grayImgTransform, bg, scale=255)
out_binary=cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]

imgAdaptiveThre = cv.adaptiveThreshold(out_binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 7)

#out_binary = cv.threshold(grayImgTransform, 190, 200, cv.THRESH_OTSU)[1]

#imgAdaptiveThre = cv.adaptiveThreshold(grayImgTransform, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 6)

cv.imshow('image', imgAdaptiveThre)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

#cv.imwrite('Test_image_gray_after.jpg', grayImgTransform)