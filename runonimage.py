import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO
import os

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
class_labels_en = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def prepare_image(char_img):
    img_gray = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def identify_characters(char_img, model_segmentation, model_identify, class_labels):
    segmentation_results = model_segmentation(char_img)
    char_boxes = segmentation_results[0].boxes.xyxy.tolist()

    if not char_boxes:
        return []

    avg_char_height = np.mean([box[3] - box[1] for box in char_boxes]) if len(char_boxes) > 0 else 0
    threshold = avg_char_height / 2

    char_boxes.sort(key=lambda box: box[1])
    rows = []
    current_row = [char_boxes[0]] if char_boxes else []

    for box in char_boxes[1:]:
        if box[1] > current_row[-1][3] - threshold:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
        else:
            current_row.append(box)

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))

    identified_chars = []
    for row in rows:
        for box in row:
            x1, y1, x2, y2 = map(int, box)
            char_crop = char_img[y1:y2, x1:x2]
            prepared_img = prepare_image(char_crop)
            predictions = model_identify.predict(prepared_img)
            predicted_class = np.argmax(predictions, axis=1)
            identified_char = class_labels[predicted_class[0]]
            identified_chars.append((identified_char, box))

    return identified_chars 



def detect_number_plate(image):
    detection_results = model_detection(image)
    boxes = detection_results[0].boxes.xyxy.tolist()
    number_plate_imgs = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        number_plate_img = image[y1:y2, x1:x2]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        number_plate_imgs.append((number_plate_img, (x1, y1, x2, y2)))
    return number_plate_imgs, image

def separate_number_plate(image):
    separator_results = model_np_en_separator(image)
    if separator_results and len(separator_results[0].boxes.cls) > 0:
        separator_class = separator_results[0].boxes.cls.tolist()[0]
        return 'Nepali' if separator_class == 0 else 'English'
    else:
        return 'Unknown'

def segment_characters(number_plate_img, is_nepali=True):
    segmentation_model = model_np_segmentation if is_nepali else model_en_segmentation
    segmentation_results = segmentation_model(number_plate_img)
    char_boxes = segmentation_results[0].boxes.xyxy.tolist()
    char_imgs = []
    for box in char_boxes:
        x1, y1, x2, y2 = map(int, box)
        char_img = number_plate_img[y1:y2, x1:x2]
        cv2.rectangle(number_plate_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        char_imgs.append((char_img, (x1, y1, x2, y2)))
    return char_imgs, number_plate_img

def recognize_characters(char_imgs, model_identify, class_labels):
    recognized_chars = []
    for char_img, box in char_imgs:
        prepared_img = prepare_image(char_img)
        predictions = model_identify.predict(prepared_img)
        predicted_class = np.argmax(predictions, axis=1)
        identified_char = class_labels[predicted_class[0]]
        recognized_chars.append((identified_char, box))
    return recognized_chars

def annotate_image_with_characters(image, recognized_string):
    # Define the position for the text (bottom right corner)
    text_position = (image.shape[1] - 10, image.shape[0] - 10)
    font_scale = 10
    font_thickness = 20

    # Calculate the text size
    text_size, _ = cv2.getTextSize(recognized_string, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_width, text_height = text_size

    # Adjust the text position to be within the image
    text_position = (image.shape[1] - text_width - 10, image.shape[0] - 10)

    # Draw the text with a black border
    cv2.putText(image, recognized_string, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2)
    # Draw the text in white color
    cv2.putText(image, recognized_string, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return image

if __name__ == "__main__":
    input_image_path = 'input.jpg'
    output_directory = 'output_images'
    os.makedirs(output_directory, exist_ok=True)

    image = cv2.imread(input_image_path)
    
    # Step 1: Detect Number Plates
    number_plate_imgs, image_with_plates = detect_number_plate(image)
    detection_output_path = os.path.join(output_directory, 'detected_number_plates.jpg')
    cv2.imwrite(detection_output_path, image_with_plates)
    print(f"Detection results saved to {detection_output_path}")

    for i, (number_plate_img, plate_box) in enumerate(number_plate_imgs):
        plate_type = separate_number_plate(number_plate_img)
        print(f"Number plate {i} classified as: {plate_type}")

        is_nepali = plate_type == 'Nepali'
        char_imgs, segmented_image = segment_characters(number_plate_img, is_nepali)
        segmentation_output_path = os.path.join(output_directory, f'segmented_characters_{i}.jpg')
        cv2.imwrite(segmentation_output_path, segmented_image)
        print(f"Segmentation results saved to {segmentation_output_path}")

        recognized_chars = identify_characters(number_plate_img, model_np_segmentation if is_nepali else model_en_segmentation, model_cnn_np if is_nepali else model_cnn_en, class_labels_np if is_nepali else class_labels_en)
        
        # Print recognized characters to console
        recognized_string = ''.join([char for char, _ in recognized_chars])
        print(f"Recognized characters for number plate {i}: {recognized_string}")

        annotated_image = annotate_image_with_characters(image, recognized_string)
        final_output_path = os.path.join(output_directory, f'annotated_number_plate_{i}.jpg')
        cv2.imwrite(final_output_path, annotated_image)
        print(f"Annotated results saved to {final_output_path}")
