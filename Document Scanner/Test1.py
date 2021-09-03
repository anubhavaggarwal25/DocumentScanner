import cv2

img = cv2.imread('img.jpg', 1)
img = cv2.resize(img,(600,800))
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


blurr = cv2.GaussianBlur(grey, (5,5), 0)

edge = cv2.Canny(blurr, 0, 50)

contours = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('Test1', img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()