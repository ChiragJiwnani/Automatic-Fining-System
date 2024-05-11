from ultralytics import YOLO
import cv2
import os
import sys
import numpy as np
import csv
from util import get_car, read_license_plate
from sortmaster.sort import *

sys.path.append(os.path.abspath("./Automatic-License-Plate-Recognition-using-YOLOv8-main/sort-master"))
from sortmaster.sort import *

mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')  # Use a smaller model variant
license_plate_detector = YOLO('license_plate_detector.pt')  # Use a smaller model variant

# Open the camera
cap = cv2.VideoCapture("./static/2103099-uhd_3840_2160_30fps.mp4")  # 0 for the default camera, or provide a camera index

vehicles = [2, 3, 5, 7]  # Vehicle classes

frame_counter = 0
frame_skip = 1  # Adjust this value as needed
prev_frame = None

# Open the CSV file for writing
csv_file = open('license_plate_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Car ID', 'License Plate', 'Confidence'])  # Write the header row

while True:
    ret, frame = cap.read()

    if ret:
        frame_counter += 1
        if frame_counter % frame_skip == 0:
            # Resize the frame for faster processing
            # frame = cv2.resize(frame, (630, 462))

            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            if len(detections_) > 0:
                track_ids = mot_tracker.update(np.asarray(detections_))
            else:
                print("No detections found.")

            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                if car_id != -1:
                    # Crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                    # Process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # Read license plate number
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        print(f"Car ID: {car_id}, License Plate: {license_plate_text}, Confidence: {license_plate_text_score:.2f}")

                        # Write data to the CSV file
                        csv_writer.writerow([car_id, license_plate_text, license_plate_text_score])

                        # Draw bounding boxes and labels
                        cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"Car ID: {car_id}", (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

            # Display the frame
            cv2.imshow("License Plate Detection", frame)

        else:
            # Display the previous frame
            if prev_frame is not None:
                cv2.imshow("License Plate Detection", prev_frame)

        # Keep track of the previous frame
        prev_frame = frame.copy()

        # Check for the 'q' key press to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the CSV file
csv_file.close()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()