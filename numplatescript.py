import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from ultralytics import YOLO
import time
import os
import requests
import paramiko

# Load YOLOv8 models
model_detection = YOLO('numberplate_detector.pt')
model_np_en_separator = YOLO('np_en_separator.pt')
model_np_segmentation = YOLO('np_segmentation.pt')
model_en_segmentation = YOLO('en_segmentation.pt')

# Load CNN models for character identification
model_cnn_np = load_model('np.keras')
model_cnn_en = load_model('en.keras')

# Class labels for CNN models (Nepali and English characters)
class_labels_np = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ba', 'cha', 'pa']
class_labels_en = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def prepare_image(char_img):
    # Convert to grayscale and resize
    img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))

    # Normalize and reshape for the CNN model
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def identify_characters(char_img, model_segmentation, model_identify, class_labels):
    segmentation_results = model_segmentation(char_img)
    char_boxes = segmentation_results[0].boxes.xyxy.tolist()

    # Check if char_boxes is empty
    if not char_boxes:
        return []  # Return an empty list or handle the case as appropriate

    # Calculate the average height of characters, safely handling an empty list
    if len(char_boxes) > 0:
        avg_char_height = np.mean([box[3] - box[1] for box in char_boxes])
    else:
        avg_char_height = 0  # Use a default value to handle this case as needed

    threshold = avg_char_height / 2  # Adjust the threshold

    char_boxes.sort(key=lambda box: box[1])
    rows = []
    current_row = []

    if char_boxes:  # Ensure there is at least one character box to process
        current_row = [char_boxes[0]]

        for box in char_boxes[1:]:
            if box[1] > current_row[-1][3] - threshold:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = [box]
            else:
                current_row.append(box)

        rows.append(sorted(current_row, key=lambda b: b[0]))

    identified_chars = []
    for row in rows:
        for box in row:
            x1, y1, x2, y2 = box
            char_crop = char_img[int(y1):int(y2), int(x1):int(x2)]
            prepared_img = prepare_image(char_crop)

            predictions = model_identify.predict(prepared_img)
            predicted_class = np.argmax(predictions, axis=1)
            identified_char = class_labels[predicted_class[0]]
            identified_chars.append(identified_char)

    return identified_chars

def wait_for_stream(source):
    cap = None
    while cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                print("Stream opened successfully.")
                break
        except cv2.error as e:
            print("Error opening video stream, trying again...")
        cv2.waitKey(1000)  # Wait for 1 second before trying again
    return cap

def is_within_secondary_roi(box, x1_sec, y1_sec, x2_sec, y2_sec):
    x1, y1, x2, y2 = box
    #return not (x2 < x1_sec or x1 > x2_sec or y2 < y1_sec or y1 > y2_sec)
    # Check if there's any overlap between the bounding box and the secondary ROI
    overlap_x = max(0, min(x2, x2_sec) - max(x1, x1_sec))
    overlap_y = max(0, min(y2, y2_sec) - max(y1, y1_sec))
    return overlap_x > 0 and overlap_y > 0

def send_alert_to_java_backend(title, subtitle):
    url = 'https://nawadurgaparking.com/api/alert'
    data = {
        'title': title,
        'subtitle': subtitle
    }
    
    try:
        response = requests.post(url, data=data)
        print(f"Alert sent: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send alert: {e}")

def upload_image(image_data):
    url = 'https://nawadurgaparking.com/api/images'
    directory = 'C:\\Users\\Shopnil\\Desktop\\test'
    file_name = f"fail_{time.strftime('%Y%m%d_%H%M%S')}.jpeg"
    file_field_name = 'image'
    file_path = os.path.join(directory, file_name)
    try:
         # Define the file payload to send
        files = {file_field_name: (file_name, image_data, 'image/jpeg')}
        # Make the POST request to upload the file
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("Image uploaded successfully.")
        else:
            print(f"Failed to upload image. Status code: {response.status_code}")

        with open(file_path, 'wb') as image_file:
            image_file.write(image_data)

    except Exception as e:
        print(f"Error uploading image: {e}")

def fetch_target_strings(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises an HTTPError for 4xx/5xx responses
        target_strings = response.json()  # Assuming the response body contains a JSON array of strings
        return target_strings
    except requests.RequestException as e:
        print(f"Failed to fetch target strings: {e}")
        return []
    
def process_session_frames(session_frames, api_url, snapshot):
    detection_statuses = []

    target_strings = fetch_target_strings(api_url)

    for number_plate_img in session_frames:
        # Use the separator model to distinguish between Nepali and English number plates
        separator_results = model_np_en_separator(number_plate_img)
        if separator_results and len(separator_results[0].boxes.cls) > 0:
            separator_class = separator_results[0].boxes.cls.tolist()[0]
        else:
            # If no separator results, skip this number plate image
            continue

        # Choose the appropriate models based on the separator's output
        segmentation_model = model_np_segmentation if separator_class == 0 else model_en_segmentation
        cnn_model = model_cnn_np if separator_class == 0 else model_cnn_en
        class_labels = class_labels_np if separator_class == 0 else class_labels_en

        # Identify characters using the segmentation and CNN models
        characters = identify_characters(number_plate_img, segmentation_model, cnn_model, class_labels)
        detected_string = ''.join(characters)
        print("Detected String:", detected_string)

        # Compare the detected string with the target string
        #detection_status = "Pass" if detected_string == target_string else "Fail"
        if detected_string in target_strings:
            detection_status = "Pass"
            break
        else:
            detection_status = "Fail"

        detection_statuses.append(detection_status)

    # Determine the overall session result
    status_message = "Pass" if "Pass" in detection_statuses else "Fail"
    print(status_message)
         
    #Save the image if fail
    if status_message == "Fail": 
        upload_image(snapshot)
        message1 = f"Security Alert"
        message2 = f"Unauthorized Vehicle Detected"
        send_alert_to_java_backend(message1,message2)

    detection_statuses = []

def process_video_source(source, api_url, roi, secondary_roi, source_type, max_frames_without_detection=30):
    while True:
        cap = wait_for_stream(source) # Wait for the stream instead of directly opening it
        # Unpack the ROI coordinates
        x1_roi, y1_roi, x2_roi, y2_roi = roi  # Unpack the primary ROI coordinates
        x1_sec, y1_sec, x2_sec, y2_sec = secondary_roi  # Unpack the secondary ROI coordinates
        snapshot_taken = False
        snapshot = None
        jpeg_image_bytes = None
        
        if not cap.isOpened():
            print(f"Error: Could not open {source_type} source.")
            return
        
        check_every_n_seconds = 3  # Time interval to check for a new number plate when not currently processing a session
        last_frame_time = time.time() - check_every_n_seconds  # Initialize to ensure the first check happens immediately

        processing_session = False  # Flag to indicate whether we are currently processing a session

        session_frames = []  # Stores frames for the current detection session
        frames_since_last_detection = 0  # Counts frames since the last number plate detection

        try:
            while True:
                current_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    # Process the collected frames for the last vehicle before exiting
                    if session_frames:
                        process_session_frames(session_frames, api_url, snapshot)
                    print("End of stream")
                    break  # End of video stream

                current_time = time.time()

                 # Check frame only every 5 seconds if not in a session
                if not processing_session and current_time - last_frame_time < check_every_n_seconds:
                    continue
                
                last_frame_time = current_time
                frame_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
                detection_results = model_detection(frame_roi)
                detected_in_this_frame = False
                

                # Detect number plate in the current frame
                if detection_results:
                    boxes = detection_results[0].boxes.xyxy.tolist() if detection_results[0].boxes.xyxy.numel() else []
                    classes = detection_results[0].boxes.cls.tolist() if detection_results[0].boxes.cls.numel() else []
                    confidences = detection_results[0].boxes.conf.tolist() if detection_results[0].boxes.conf.numel() else []
                    
                    for box, class_id, confidence in zip(boxes, classes, confidences):
                        if class_id == 0 and confidence > 0.5:  # class ID for number plates
                            if is_within_secondary_roi(box, x1_sec, y1_sec, x2_sec, y2_sec) && snapshot_taken == False:
                                    # Take a snapshot if not already taken
                                    snapshot = frame.copy()
                                    _, jpeg_image = cv2.imencode('.jpg', snapshot)
                                    jpeg_image_bytes = jpeg_image.tobytes()
                                    snapshot_taken = True
                            
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            cv2.rectangle(frame_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                            number_plate_img = frame_roi[y1:y2, x1:x2]
                            detected_in_this_frame = True
                            session_frames.append(number_plate_img)  # Add the whole frame to the session
                            processing_session = True  # Indicate that a session is being processed
                            break

                if not detected_in_this_frame:
                    frames_since_last_detection += 1
                    if frames_since_last_detection >= max_frames_without_detection and session_frames:
                        # Vehicle considered to have left the frame, process the captured frames
                        process_session_frames(session_frames, api_url, jpeg_image_bytes)
                        snapshot = None
                        session_frames = []  # Reset for the next vehicle
                        frames_since_last_detection = 0
                        processing_session = False  # No longer processing a session
                        snapshot_taken = False
                        last_frame_time = current_time  # Reset timer to start checking every 5 seconds again
                else:
                    frames_since_last_detection = 0  # Reset counter on detection
                
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # Make the window resizable
                cv2.imshow('Frame', frame_roi)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
                print(f"An error occurred: {e}")            
        finally:
            cap.release()
            cv2.destroyAllWindows()

# ROI coordinates: (x1, y1, x2, y2)
roi_coordinates = (0, 50, 1920, 1080)  
secondary_roi_coordinates = (0, 270, 1920, 400)  

api_url = 'https://nawadurgaparking.com/api/vehicles/number-plates'
rtsp_url = 'rtsp://admin:123456@192.168.11.69:554/media/video1'
#rtsp_url = 'C:\\Users\\Shopnil\\Desktop\\test\\00_11_20240207_143500.avi'
#rtsp_url = 'C:\\Users\\Shopnil\\Desktop\\test\\00_11_20240207_134600.avi'

process_video_source(rtsp_url, api_url, roi_coordinates, secondary_roi_coordinates, source_type='stream')
#process_video_source(rtsp_url, api_url, roi_coordinates, secondary_roi_coordinates, source_type='video')