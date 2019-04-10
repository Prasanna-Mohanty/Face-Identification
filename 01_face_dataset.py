import cv2
import os

#cam = cv2.VideoCapture(0)
vid_cam = cv2.VideoCapture(0)
vid_cam.set(3, 640) # set video width
vid_cam.set(4, 480) # set video height

#face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detector = cv2.CascadeClassifier('C:/Users/Prasanna Mohanty/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):
    #ret, img = cam.read()
    _, image_frame = vid_cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 100, stop taking video
    elif count>100:
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()