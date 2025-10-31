from rtmlib import BodyWithFeet, draw_skeleton
import cv2
import time
import numpy as np
from math import floor
from jitter import OneEuroFilter
from frame_preprocessing import colour_correction, trace_contour, find_contour_coordinates


filters = [
    {"x": OneEuroFilter(frequency=30, min_cutoff=1.0, beta=0.01),
     "y": OneEuroFilter(frequency=30, min_cutoff=1.0, beta=0.01)}
    for _ in range(4)
]


class WalkingTest:
    LINE_COLOUR = (0, 255, 0)
    LINE_THICKNESS = 2


    def __init__(self, model, source=None):
        self.cap = cv2.VideoCapture(source)
        self.source = source
        self.model = model
        self.start_time = 0.0
        self.end_time = 0.0
        self.status = None
        self.width = None
        self.height = None
        self.start_line = None
        self.end_line = None 
        self.start_frame_count = 0
        self.end_frame_count = 0
        self.frame = 0


    def calculate_line(self, x1, y1, x2, y2):
        m = (y2-y1)/(x2-x1)
        c = y1 - m*x1
        return m, c
    

    def calculate_y_on_line(self, m, x, c):
        return m*x + c
    

    def getMarkings(self, frame):
        masked_frame = colour_correction(frame, colour = "green")
        contour_lists = trace_contour(masked_frame)
        start_line, end_line = find_contour_coordinates(contour_lists)

        self.start_line = start_line
        self.end_line = end_line
        return None


    def getLine(self, type):
        # Finds the two endpoints of a line 
        if type == "Start":
            (x1, y1), (x2, y2) = self.start_line

        elif type == "End":
            (x1, y1), (x2, y2) = self.end_line

        m, c = self.calculate_line(x1, y1, x2, y2)
        y_start = int(self.calculate_y_on_line(m, x1, c))
        y_end = int(self.calculate_y_on_line(m, x2, c))

        return (0, y_start), (self.WIDTH, y_end)


    def isBelowLine(self, toe):
        if self.status == None:
            # Find the line from START_LINE
            (x1, y1), (x2, y2) = self.start_line

        elif self.status == "Start":
            # Find the line from END_LINE
            (x1, y1), (x2, y2) = self.end_line
            
        m, c = self.calculate_line(x1, y1, x2, y2)

        # Find y along the line at toe[0]
        x = toe[0]
        y = toe[1]
        y_on_line = m*x + c
        if y >= y_on_line:
            return True
        else: 
            return False


    def crossStartLine(self, LeftBigToe, RightBigToe, LeftToe, RightToe, curr_time):
        # The foot has crossed the start line if the big toe or toe is above (y smaller than) the start line
        left_foot = (self.isBelowLine(LeftBigToe) or self.isBelowLine(LeftToe))
        right_foot = (self.isBelowLine(RightBigToe) or self.isBelowLine(RightToe))

        if left_foot or right_foot:
            self.status = "Start"
            self.start_time = curr_time
            self.start_frame_count = self.frame


    def crossEndLine(self, LeftBigToe, RightBigToe, LeftToe, RightToe, curr_time):
        left_foot = (self.isBelowLine(LeftBigToe) or self.isBelowLine(LeftToe))
        right_foot = (self.isBelowLine(RightBigToe) or self.isBelowLine(RightToe))

        if left_foot or right_foot:
            self.status = "End"
            self.end_time = curr_time
            self.end_frame_count = self.frame


    def process_frame(self, frame, fps):
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)
        curr_time = time.time()
        self.frame += 1

        if not results:
            return frame, None
        
        # Do image preprocessing here
        if self.start_line is None or self.end_line is None:
            self.getMarkings(frame)

        # Get required keypoints
        keypoints, scores = results
        annotated_image = draw_skeleton(frame, keypoints, scores)
        required_keypoints = keypoints[0]
        LeftBigToe = required_keypoints[20]
        RightBigToe = required_keypoints[21]
        LeftToe = required_keypoints[22]
        RightToe = required_keypoints[23]

        # Apply the OneEuroFilter
        LeftBigToe[0] = filters[0]["x"].apply(LeftBigToe[0], curr_time)
        LeftBigToe[1] = filters[0]["y"].apply(LeftBigToe[1], curr_time)

        RightBigToe[0] = filters[1]["x"].apply(RightBigToe[0], curr_time)
        RightBigToe[1] = filters[1]["y"].apply(RightBigToe[1], curr_time)

        LeftToe[0] = filters[2]["x"].apply(LeftToe[0], curr_time)
        LeftToe[1] = filters[2]["y"].apply(LeftToe[1], curr_time)

        RightToe[0] = filters[3]["x"].apply(RightToe[0], curr_time)
        RightToe[1] = filters[3]["y"].apply(RightToe[1], curr_time)

        # Check line crossings
        if self.status is None:
            self.crossStartLine(LeftBigToe, RightBigToe, LeftToe, RightToe, curr_time)
            elapsed_time = 0.0
        elif self.status == "Start":
            self.crossEndLine(LeftBigToe, RightBigToe, LeftToe, RightToe, curr_time)
            elapsed_time = (self.frame - self.start_frame_count) / fps
            #elapsed_time = curr_time - self.start_time
        elif self.status == "End":
            # frames = (self.end_frame_count - self.start_frame_count) / fps #fps
            # elapsed_time = (self.end_time - self.start_time) * frames
            elapsed_time = (self.end_frame_count - self.start_frame_count) / fps


        # Draw start and end lines
        start_line1, start_line2 = self.getLine("Start")
        end_line1, end_line2 = self.getLine("End")
        cv2.line(annotated_image, start_line1, start_line2, WalkingTest.LINE_COLOUR, WalkingTest.LINE_THICKNESS)
        cv2.line(annotated_image, end_line1, end_line2, WalkingTest.LINE_COLOUR, WalkingTest.LINE_THICKNESS)

        # Display text overlay
        overlay_text = f"Status: {self.status or 'Waiting'}"
        elapsed_minutes = floor(elapsed_time / 60)
        elapsed_seconds = round(elapsed_time % 60, 2)
        time_text = f"Elapsed: {elapsed_minutes}m {elapsed_seconds}s"

        cv2.putText(annotated_image, overlay_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_image, time_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return annotated_image, self.status


    def process_video(self, output_path="../output/green_office_output.mp4"):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = 360  # GET FROM THE VIDEO ITSELF
        height = 640

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (360, 640))
            self.WIDTH = frame.shape[0]
            self.HEIGHT = frame.shape[1]
            image, _ = self.process_frame(frame, fps)
            out.write(image)

        self.cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model = BodyWithFeet(mode="performance", to_openpose=False, backend="onnxruntime", device="cpu")

    video_path = "../data/green_office.mp4"
    walking_test = WalkingTest(model, source=video_path)
    walking_test.process_video()