import cv2

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, video_data = video_cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Display the video with face detection
    cv2.imshow("video_live", video_data)
    
    # Press 'a' to exit
    if cv2.waitKey(10) == ord('a'):
        break

# Release the camera and close all windows
video_cap.release()
cv2.destroyAllWindows()
