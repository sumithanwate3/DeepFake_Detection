import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

import f_detector


def detect_blinks(video_path):
    # Instantiate detector
    detector = f_detector.eye_blink_detector()
    
    # Initialize variables for blink detector
    COUNTER = 0
    TOTAL = 0
    ear_values = []
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
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

    def get_video_duration(video_path):
        clip = VideoFileClip(video_path)
        duration_sec = clip.duration
        clip.close()
        return duration_sec
    
    video_duration_sec = get_video_duration(video_path)

    if video_duration_sec != 0:  # Avoid division by zero
        blinks_per_minute = (total_blinks / video_duration_sec) * 60
    else:
        blinks_per_minute = 0
    
    return total_blinks, blinks_per_minute, ear_values

# Example usage
video_path = "/Users/sumit/Downloads/WhatsApp Video 2024-04-19 at 22.28.43.mp4"
total_blinks, blinks_per_minute, ear_values = detect_blinks(video_path)

# Print results
print("Total blinks:", total_blinks)
print("Blinks per minute:", blinks_per_minute)
print("EAR values:", ear_values)

# Plot the EAR values
plt.figure(figsize=(10, 5))
plt.plot(ear_values, label='EAR Values', color='blue')
plt.title('Eye Aspect Ratio (EAR) over Time')
plt.xlabel('Frame Number')
plt.ylabel('EAR Value')
plt.legend()
plt.grid(True)
plt.show()
