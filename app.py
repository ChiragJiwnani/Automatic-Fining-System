from flask import Flask, redirect, render_template, request
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import numpy as np
import csv
from util import get_car, read_license_plate
from sortmaster.sort import *
import time

import torch
print(torch.cuda.get_device_name())


app = Flask(__name__)

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

mot_tracker = Sort()
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
coco_model.to('cpu')
license_plate_detector.to('cpu')
vehicles = [2, 3, 5, 7]
frame_skip = 8
prev_frame = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    frame_skip = 10  
    prev_frame = None

    csv_file = open('app_license_plate.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Car ID', 'License Plate', 'Confidence'])
    cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            if frame_counter % frame_skip == 0:
                detections = coco_model(frame)[0]
                detections_ = []
                for detection in detections.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection
                    if int(class_id) in vehicles:
                        detections_.append([x1, y1, x2, y2, score])

                if len(detections_) > 0:
                    track_ids = mot_tracker.update(np.asarray(detections_))
                else:
                    print("No detections found.")

                license_plates = license_plate_detector(frame)[0]
                for license_plate in license_plates.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = license_plate

                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                    if car_id != -1:
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                        if license_plate_text is not None:
                            print(f"Car ID: {car_id}, License Plate: {license_plate_text}, Confidence: {license_plate_text_score:.2f}")
                            csv_writer.writerow([car_id, license_plate_text, license_plate_text_score])
                            
                            # Draw bounding boxes and labels
                            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"Car ID: {car_id}", (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)


            #     # Display the frame
                
            #     cv2.imshow("License Plate Detection", frame)
            # else:
            #     # Display the previous frame
            #     if prev_frame is not None:
            #         cv2.imshow("License Plate Detection", prev_frame)

            # # Keep track of the previous frame
            # prev_frame = frame.copy()

            # # Check for the 'q' key press to quit
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # Close the CSV file
    csv_file.close()

    cap.release()
    cv2.destroyAllWindows()
    return csv_file
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        csv_data = process_video(filepath)
        # Save data to CSV file
        with open('license_plate_data_app.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Car ID', 'License Plate', 'Confidence'])
            csv_writer.writerows(csv_data)

        print ('File uploaded and processed successfully')
        return redirect('/')  # Redirects to the home page
        # Redirect to home or a success page after processing
    
    return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=False)
