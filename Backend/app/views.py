import os
from io import BytesIO

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import requests
from django.conf import settings
from django.http import JsonResponse
from moviepy.editor import VideoFileClip
from PIL import Image
from tensorflow.keras.models import load_model

from . import f_detector

model_path = os.path.join(settings.BASE_DIR, 'model', 'Best_Model2.keras')
best_model = load_model(model_path, safe_mode=False)


def extract_frames_from_video(video_content):
    temp_filename = 'temp_video.mp4'
    with open(temp_filename, 'wb') as f:
        f.write(video_content)

    cap = cv2.VideoCapture(temp_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % fps == 0:
            frames.append(frame)
        frame_num += 1

    cap.release()

    os.remove(temp_filename)

    return frames

def process_image(image):
    resized_image = image.resize((300, 300))
    normalized_image = np.array(resized_image) / 255.0
    processed_image = np.expand_dims(normalized_image, axis=0)

    prediction = best_model.predict(processed_image)
    print(prediction)
    result_label = "Fake" if prediction < 0.5 else "Real"
    return result_label

def process_video(video_frames):
    predictions = []
    for frame in video_frames:
        frame_copy = frame.copy()
        resized_frame = cv2.resize(frame_copy, (300, 300))  
        normalized_frame = np.array(resized_frame) / 255.0 
        processed_frame = np.expand_dims(normalized_frame, axis=0)  
 
        prediction = best_model.predict(processed_frame)
        predictions.append(prediction)
    avg_prediction = np.mean(predictions)
    result_label = "Fake" if avg_prediction < 0.5 else "Real"
    return result_label

def detect_blinks(video_content):
    # Instantiate detector
    detector = f_detector.eye_blink_detector()
    
    # Initialize variables for blink detector
    COUNTER = 0
    TOTAL = 0
    ear_values = []
    
    # Open video file
    cap = cv2.VideoCapture(video_content)
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return None, None, None
    
    # Loop through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        rectangles = detector.detector_faces(gray, 0)
        boxes_face = f_detector.convert_rectangles2array(rectangles, frame)
        
        # If faces are detected
        if len(boxes_face) != 0:
            # Select the face with the largest area
            areas = f_detector.get_areas(boxes_face)
            index = np.argmax(areas)
            rectangles = rectangles[index]
            boxes_face = np.expand_dims(boxes_face[index], axis=0)
            
            # Detect blinks
            COUNTER, TOTAL, ear_value = detector.eye_blink(gray, rectangles, COUNTER, TOTAL)
            ear_values.append(ear_value)
    
    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate blinks per minute (BPM)
    total_blinks = TOTAL

    def get_video_duration(video_content):
        clip = VideoFileClip(video_content)
        duration_sec = clip.duration
        clip.close()
        return duration_sec
    
    video_duration_sec = get_video_duration(video_content)

    if video_duration_sec != 0:  # Avoid division by zero
        blinks_per_minute = (total_blinks / video_duration_sec) * 60
    else:
        blinks_per_minute = 0
    
    return total_blinks, blinks_per_minute, ear_values


def process_media(request):
    if request.method == 'GET':
        media_url = request.GET.get('url')
        print(media_url)
        response = requests.get(media_url)
        if response.status_code == 200:
            content_type = response.headers['content-type']
            if 'image' in content_type:
                image = Image.open(BytesIO(response.content))
                result_label = process_image(image)
                return JsonResponse({'result': result_label})
            elif 'video' in content_type:
                video_content = response.content
                # Save the video content to a temporary file
                temp_filename = 'temp_video.mp4'
                with open(temp_filename, 'wb') as f:
                    f.write(video_content)

                # Open the temporary file with cv2.VideoCapture
                cap = cv2.VideoCapture(temp_filename)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_duration = total_frames / fps

                # Release the video capture object
                cap.release()

                if video_duration < 10:  # Proceed with regular processing
                    video_frames = extract_frames_from_video(video_content)
                    result_label = process_video(video_frames)
                    print(result_label)
                    return JsonResponse({'result': result_label})
                else:  # Call the eye blink detection program
                    total_blinks, blinks_per_minute, ear_values = detect_blinks(temp_filename)
                    # Delete the temporary file
                    os.remove(temp_filename)
                    # Return the result
                    print(total_blinks, blinks_per_minute)
                    if(10 <= blinks_per_minute <= 20):
                        result = "Real"
                    else :
                        result = "Fake"
                    return JsonResponse({'Result': result, 'total_blinks': total_blinks, 'blinks_per_minute': blinks_per_minute})
            else:
                return JsonResponse({'error': 'Unsupported media type'})
        else:
            return JsonResponse({'error': 'Failed to download media'})
    else:
        return JsonResponse({'error': 'Invalid request method'})

