""" from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

course=input("Enter Course Code: ")

video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/mat_no.pkl', 'rb') as w:
    LABELS=pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground=cv2.imread("background.png")

COL_NAMES = ['MATRIC NUMBER', 'TIME',  'DATE', 'COURSE']

last_attendance_time = {}

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    for (x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance=[str(output[0]), str(timestamp), str(date), str(course)]
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if k==ord('o'):
        if (output[0], course) in last_attendance_time:
            last_time = last_attendance_time[(output[0], course)]
            time_diff = ts - last_time
            if time_diff <= 300:
                print("Attendance taken for ", str(output[0]))
                time.sleep(1)
                if exist:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer=csv.writer(csvfile)
                        writer.writerow(attendance)
                    csvfile.close()
                else:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer=csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                    csvfile.close()
            else:
                print("Attendance already marked for ", str(output[0]))
        else:
            print("Attendance taken for ", str(output[0]))
            time.sleep(1)
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(attendance)
                csvfile.close()
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                csvfile.close()
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows() """

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

course = input("Enter Course Code: ")

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/mat_no.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE']

# Dictionary to store the last attendance time for each face ID and course
last_attendance_time = {}

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray, 1.3 , 5)
    for (x, y, w, h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1, -1)
        output=knn.predict(resized_img)
        ts=time.time()
        date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        attendance = [str(output[0]), timestamp, date, course]
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", frame)
    k=cv2.waitKey(1)
    if k==ord('m'):
        # Same attendance marking logic as before (without needing another cv2.waitKey())
        if (str(output[0]), course) in last_attendance_time:
            last_time = last_attendance_time[(str(output[0]), course)]
            time_diff = ts - last_time
            if time_diff >= 300:  # 5 minutes = 300 seconds
                time.sleep(1)
                if exist:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer=csv.writer(csvfile)
                        writer.writerow(attendance)
                    csvfile.close()
                else:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer=csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                    csvfile.close()
                print("Attendance taken for ", str(output[0]))
            else:
                print(f"Already marked attendance for {str(output[0])}")
        else:
            time.sleep(1)
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(attendance)
                csvfile.close()
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer=csv.writer(csvfile)
                    writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                csvfile.close()
            print("Attendance taken for", str(output[0]))
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


