import math

import cv2
import numpy as np


def canny(img, canny_low_threshold, canny_high_threshold):
    return cv2.Canny(img, canny_low_threshold, canny_high_threshold)


def task_one():
    cv2.startWindowThread()
    cap = cv2.VideoCapture("road_trimmed.mp4")

    ret, frame = cap.read()
    h, w = frame.shape[0:2]
    h_new = 700
    ratio = w / h
    w_new = int(h_new * ratio)
    frame = cv2.resize(frame, (w_new, h_new))
    cv2.imwrite('1_frame.jpeg', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video", frame)
    cv2.imwrite('1_frame_gray.jpeg', frame)
    cap.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blur_kernel_size = (15, 15)
    gray_img = cv2.imread('1_frame_gray.jpeg')
    gray_blur = cv2.GaussianBlur(gray_img, blur_kernel_size, 0)
    cv2.imshow("Grayslace", gray_blur)
    cv2.imwrite('1_frame_gray_blurred.jpeg', gray_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    canny_low_threshold = 20
    canny_high_threshold = 100
    grey_blur = cv2.imread('1_frame_gray_blurred.jpeg')
    blur_canny = canny(gray_blur, canny_low_threshold, canny_high_threshold)
    cv2.imshow("Photo", blur_canny)
    cv2.imwrite('1_frame_gray_blurred_canny.jpeg', blur_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread('1_frame_gray_blurred_canny.jpeg')
    h = 270
    w = 512
    x = 400
    y = 0
    img1 = img[x:x + h, y:y + w]
    img2 = np.zeros_like(img)
    img2[x:x + h, y:y + w] = img1
    cv2.imwrite('roi.jpeg', img2)
    cv2.imshow("Photo", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    src = cv2.imread('roi.jpeg', cv2.IMREAD_GRAYSCALE)
    dst = cv2.Canny(src, 50, 200, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    red = np.copy(cdst)
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    for d in red:
        for d1 in d:
            d1[0] = 0
            d1[1] = 0
            d1[2] = 0

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(red, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Image", cdstP)
    cv2.imwrite('red.jpeg', red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.imread('1_frame.jpeg')
    img1 = cv2.imread("red.jpeg")
    img = cv2.resize(img, (512, 270))
    img1 = cv2.resize(img1, (512, 270))
    img2 = cv2.addWeighted(img, 0.8, img1, 1, 0)
    cv2.imshow("2", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(image):
    # Convert to Grey Image
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(grey_image, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # Defining Region of Interest
    imshape = image.shape
    # vertices = np.array([[(0, imshape[0]), (450, 320), (500, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    vertices = np.array([[(0, imshape[0]), (500, 525), (520, 525), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 30  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the original image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


def task_two():
    video_capture = cv2.VideoCapture('road_trimmed.mp4')

    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames: {length}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            output = process_image(frame)
            cv2.putText(output, 'Ihor Klimov', (900, 700), font, 2, (255, 255, 255), 2, cv2.LINE_4)
            cv2.imshow('frame', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # task_one()
    task_two()
