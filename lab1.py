import imutils as imutils
import numpy as np
import cv2


def main():
    img = cv2.imread('cats.jpeg')
    img_gray = cv2.imread('cats.jpeg', 0)

    cv2.imshow('cats', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('file.jpg', img_gray)

    (h, w, d) = img.shape
    print(" width ={}, height ={}, depth ={}".format(w, h, d))

    (B, G, R) = img[100, 50]
    print("R={}, G={}, B={}".format(R, G, B))

    # extract a 100x100 pixel square ROI ( Region of Interest ) from the
    # input image starting at x=320 ,y=60 at ending at x=420 ,y=160
    roi = img[160:260, 420:520]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)

    # resize the image to 200x200px , ignoring aspect ratio
    resized = cv2.resize(img, (200, 200))
    cv2.imshow("Fixed Resizing ", resized)
    cv2.waitKey(0)

    # Resize, with aspect ratio
    h, w = img.shape[0:2]
    h_new = 300
    ratio = w / h
    w_new = int(h_new * ratio)
    resized = cv2.resize(img, (w_new, h_new))
    print(resized.shape)
    cv2.imshow("1", resized)
    cv2.waitKey(0)

    # Rotation
    h, w = img.shape[0:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -45, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imshow("OpenCV Rotation", rotated)
    cv2.waitKey(0)

    rotated = imutils.rotate(img, -45)
    cv2.imshow("Imutils Rotation", rotated)
    cv2.waitKey(0)

    # Blur
    # apply a Gaussian blur with a 11x11 kernel to the image to smooth it ,
    # useful when reducing high frequency noise
    blurred = cv2.GaussianBlur(img, (11, 11), 0)
    cv2.imshow("Blurred", blurred)
    cv2.waitKey(0)

    blurred = cv2.GaussianBlur(img, (27, 27), 0)
    resized = imutils.resize(img, width=460)
    bresized = imutils.resize(blurred, width=460)
    suming = np.hstack((resized, bresized))
    cv2.imshow("Suming", suming)
    cv2.waitKey(0)

    # Drawing
    output = img.copy()
    cv2.rectangle(output, (270, 50), (420, 260), (0, 0, 255), 2)
    cv2.imshow("Rectangle", output)
    cv2.waitKey(0)

    # Create a black image
    img = np.zeros((200, 200, 3), np.uint8)
    # Draw a diagonal blue line with thickness of 5 px
    cv2.line(img, (0, 0), (200, 200), (255, 0, 0), 5)
    cv2.imshow("Rectangle", img)
    cv2.waitKey(0)

    img = np.zeros((1000, 1000, 3), np.uint8)
    points = np.array([[600, 200], [910, 641], [300, 300], [0, 0]])
    cv2.polylines(img, np.int32([points]), 1, (255, 255, 255))
    cv2.imshow('1', img)
    cv2.waitKey(0)

    img = np.zeros((200, 200, 3), np.uint8)
    output = img.copy()
    cv2.circle(output, (100, 100), 50, (0, 0, 255), 2)
    cv2.imshow("Circle", output)
    cv2.waitKey(0)

    img = cv2.imread('cats.jpeg')
    font = cv2.FONT_HERSHEY_SIMPLEX
    font1 = cv2.FONT_HERSHEY_COMPLEX
    font2 = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    cv2.putText(img, 'OpenCV ', (10, 300), font, 2, (0, 0, 0), 2, cv2.LINE_4)
    cv2.putText(img, 'OpenCV ', (10, 200), font1, 2, (0, 0, 0), 2, cv2.LINE_4)
    cv2.putText(img, 'OpenCV ', (10, 100), font2, 2, (0, 0, 0), 4, cv2.LINE_4)
    cv2.imshow("Text", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
