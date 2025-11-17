# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows


## Program:
```
#Name : Laakshit D
#Reg no: 212222230071
```
```py
#1. Import Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

#2. Load Haar Cascade Classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

#3. Helper Function to Display Images
def show(img, title="Image"):
    plt.figure(figsize=(6,6))
    if len(img.shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis("off")

#4. Load Multiple Test Images
img_single = cv2.imread("face_single.jpg")
img_glasses = cv2.imread("face_glasses.jpg")
img_group = cv2.imread("group_people.jpg")

show(img_single, "Single Face")
show(img_glasses, "Face with Glasses")
show(img_group, "Group Photo")


#5. Extract ROI from an Image
# Example: Extracting the top-left 200x200 region from the single face image
roi = img_single[50:250, 50:250]
show(roi, "Extracted ROI")

#6. Face Detection Function
def detect_faces(img, scaleFactor=1.1, minNeighbors=4):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

    output = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return output, faces

#7. Apply Face Detection on All Images
Single Face
out_single, faces_single = detect_faces(img_single)
show(out_single, f"Single Face – Detected: {len(faces_single)}")

Face with Glasses
out_glasses, faces_glasses = detect_faces(img_glasses)
show(out_glasses, f"Face with Glasses – Detected: {len(faces_glasses)}")

Group Photo
out_group, faces_group = detect_faces(img_group)
show(out_group, f"Group Photo – Faces Detected: {len(faces_group)}")

#8. Eye Detection
def detect_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    output = img.copy()
    for (x, y, w, h) in eyes:
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return output, eyes

eyes_out, eyes = detect_eyes(img_single)
show(eyes_out, f"Eyes Detected: {len(eyes)}")

#9. Real-Time Face Detection with Label (Webcam + Matplotlib)
import time
from IPython.display import clear_output

cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "FACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    clear_output(wait=True)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    plt.pause(0.001)

cap.release()
plt.close()
```
## Output:

<img width="415" height="564" alt="image" src="https://github.com/user-attachments/assets/1374ff07-3eb9-48f6-9b74-32d16cb7a9b3" />

<img width="521" height="549" alt="image" src="https://github.com/user-attachments/assets/8af73c23-2629-4177-a691-cddb8b63292c" />

## Result:
Thus executed successfully.
