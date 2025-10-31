# This script is to preprocess the frame to extract positions for the markings
import cv2
import numpy as np
import math

def colour_correction(frame, colour = 'red'):
    # Convert to HSV for color masking
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red
    if colour == "red":
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Blue 
    elif colour == "blue":
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        masked_frame = cv2.bitwise_and(frame, frame, mask=blue_mask)

    # Green
    elif colour == "green":
        lower_green = np.array([35, 70, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        masked_frame = cv2.bitwise_and(frame, frame, mask=green_mask)

    # cv2.imshow("mask", masked_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return masked_frame

def is_convex_polygon(polygon):
    """Return True if the polynomial defined by the sequence of 2D
    points is 'strictly convex': points are valid, side lengths non-
    zero, interior angles are strictly between zero and a straight
    angle, and the polygon does not intersect itself.

    NOTES:  1.  Algorithm: the signed changes of the direction angles
                from one side to the next side must be all positive or
                all negative, and their sum must equal plus-or-minus
                one full turn (2 pi radians). Also check for too few,
                invalid, or repeated points.
            2.  No check is explicitly done for zero internal angles
                (180 degree direction-change angle) as this is covered
                in other ways, including the `n < 3` check.
    """
    TWO_PI = 2 * math.pi
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon[-2]
        new_x, new_y = polygon[-1]
        new_direction = math.atan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = math.atan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -math.pi:
                angle += TWO_PI  # make it in half-open interval (-Pi, Pi]
            elif angle > math.pi:
                angle -= TWO_PI
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / TWO_PI)) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon


def trace_contour(frame, approx_factor=0.02):
    # Finds the contour using findContour() and packs relevant coordinates into a nested list
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grey", grey)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    blur = cv2.GaussianBlur(grey, (7, 7), 0.7)
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)  # fills small gaps
    edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)
    #edge = cv2.Canny(blur, 0, 50, 3)
    # cv2.imshow("edge", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    _, binary = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    lst = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, approx_factor * cv2.arcLength(contour, True), True)
        polygon_points = [tuple(pt[0]) for pt in approx] 
        area = cv2.contourArea(contour)
        if is_convex_polygon(polygon_points) and area > 120:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            n = approx.ravel()
            lst.append(n)    

    # cv2.imshow("Filtered Contours", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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

    if contour_lists[:2]:
        start_line = contour_lists[0][:2]
        end_line = contour_lists[1][:2]
    else:
        start_line = None
        end_line = None

    #print(f"Sorted list using list.sort(): {contour_lists}")

    return start_line, end_line

# masked_frame = colour_correction("../data/green_office_pic.png", colour = "green")
# contour_lists = trace_contour(masked_frame)
# coordinates = find_contour_coordinates(contour_lists)
# print(coordinates)