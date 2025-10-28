# This script is to preprocess the frame to extract positions for the markings
import cv2
import numpy as np

def colour_correction(frame):
    # Convert to HSV for color masking (red in this case)
    # ADD FOR OTHER COLOURS, CREATE ARG FOR DIFF COLOURS
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow("mask", masked_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return masked_frame


def trace_contour(frame, approx_factor=0.009):
    # Finds the contour using findContour() and packs relevant coordinates into a nested list
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Gaussian blur?

    ###### Erosion - is it okay though cause the distance needs to be very exact
    # kernel_e = np.ones((5,5), np.uint8)
    # erosion = cv2.erode(gray, kernel_e, iterations = 1)
    # cv2.imshow("erosion", erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ##### Dilation
    # kernel_d = np.ones((4,4), np.uint8)
    # edges_dilated = cv2.dilate(gray, kernel_d, iterations=1)

    ##### TRY CANNY EDGE DETECTION 

    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lst = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, approx_factor * cv2.arcLength(contour, True), True)
        n = approx.ravel()
        lst.append(n)

    contour_lists = []
    for j in lst:
        i = 0
        internal_list = []
        for k in j:
            if i % 2 == 0:
                x, y = j[i], j[i + 1]
                internal_list.append([int(x), int(y)]) # need to convert to int() from np int32
            i += 1 
        contour_lists.append(internal_list)
             
    return contour_lists

def find_contour_coordinates(contour_lists):
    # We are expecting only two nested lists
    # Find 4 points - 2 for start/end line and return them
    contour_lists.sort(key=lambda contour: min(x[1] for x in contour))

    i = 0
    # Sort by y value then by x value
    for contour in contour_lists:
        if i % 2 == 0:
            contour.sort(key=lambda x: (-x[1], x[0])) # we want the lower line for start line 
        else: 
            contour.sort(key=lambda x: (x[1], x[0])) # we want the upper line for end line
        i += 1

    start_line = contour_lists[0][:2]
    end_line = contour_lists[1][:2]

    #print(f"Sorted list using list.sort(): {contour_lists}")

    return start_line, end_line

# masked_frame = colour_correction("../data/walking_with_markings.jpg")
# contour_lists = trace_contour(masked_frame)
# coordinates = find_contour_coordinates(contour_lists)
# print(coordinates)