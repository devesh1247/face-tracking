import cv2
import time

# Load the pre-trained face, eye, and smile detection models (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start video capture
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# To calculate frames per second (FPS)
prev_frame_time = 0
new_frame_time = 0

# Counter for face tracking
face_id = 0
faces_dict = {}

# Screenshot counter
screenshot_counter = 0

while True:
    ret, video_data = video_cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop through each face detected
    for (x, y, w, h) in faces:
        face_id += 1

        # Draw rectangle around the face
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Label faces with tracking ID
        face_label = f"Face {face_id}"
        faces_dict[face_id] = (x, y, w, h)
        cv2.putText(video_data, face_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Detect eyes in the face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = video_data[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Detect smiles in the face region
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    
    # Overlay FPS and face count on the video
    cv2.putText(video_data, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(video_data, f"Faces: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the video with face, eye, and smile detection
    cv2.imshow("Live Video - Face, Eye, and Smile Detection", video_data)
    
    # Press 'a' to exit, 's' to take a screenshot
    key = cv2.waitKey(10)
    if key == ord('a'):
        break
    elif key == ord('s'):
        # Save the screenshot
        screenshot_filename = f"screenshot_{screenshot_counter}.png"
        cv2.imwrite(screenshot_filename, video_data)
        print(f"Screenshot saved as {screenshot_filename}")
        screenshot_counter += 1

# Release the camera and close all windows
video_cap.release()
cv2.destroyAllWindows()
