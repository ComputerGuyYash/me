import face_recognition
import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')
i=0
end = 10
rep = 0
st = 1
while(cap.isOpened()):
    i+=1
    ret, img = cap.read()
    scale_percent=50
    try:
        img.all()
    except:
        break
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)    
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("image",resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #face_landmarks_list = face_recognition.face_landmarks(resized)
    if(i==st):
        default = face_recognition.face_encodings(resized)[0]
    elif(i==end):
        try:
            second = face_recognition.face_encodings(resized)[0]
            
            results = face_recognition.face_distance([second], default)
            print(results)
        except:
            
            rep+=1
            if(rep>10):
                rep=0
                st+=1
            end+=1
            print("Face Math Error")
            continue
    elif(i>end):
        break


cap.release()
cv2.destroyAllWindows()
