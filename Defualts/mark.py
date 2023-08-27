import sys
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import pickle
import csv
import time
from datetime import datetime
from Attendance.testcsv import real
import numpy as np
import tkinter as tk
from tkinter import messagebox


def add_std(name):
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')

    faces_data = []

    i = 0

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            i = i + 1
            cv2.putText(frame, str(len(faces_data)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q') or len(faces_data) == 100:
            break
    video.release()
    cv2.destroyAllWindows()

    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    if 'mat_no.pkl' not in os.listdir('../data/'):
        names = [name] * 100
        with open('../data/mat_no.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('../data/mat_no.pkl', 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open('../data/mat_no.pkl', 'wb') as f:
            pickle.dump(names, f)

    if 'faces_data.pkl' not in os.listdir('../data/'):
        with open('../data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
            print("student added to base")
    else:
        with open('../data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
            print("student added to base")
        faces = np.append(faces, faces_data, axis=0)
        with open('../data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
            print("student added to base")


def mark_attendance(course):
    video = cv2.VideoCapture(0)
    facedetect = (cv2.CascadeClassifier
                  ('../data/haarcascade_frontalface_default.xml'))

    with open('../data/mat_no.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('../data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    print('Shape of Faces matrix --> ', FACES.shape)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)


    col_names = ['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE']

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img,
                                     (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y-15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y+h), (50, 50, 255), 1)
            attendance = [str(output[0]), timestamp, date, course]
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('m'):
            # Same attendance marking logic as before (without needing another cv2.waitKey())
            num = real(output[0], course)
            if num == -2:
                time.sleep(2)
                with open("../Attendance/Attendance_23-07-2023.csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col_names)
                    writer.writerow(attendance)
                csvfile.close()
                messagebox.showinfo(title="Attendance",
                                    message="Attendance taken for " + attendance[0])
                print("Attendance taken for", attendance[0])
            elif num == 0:
                time.sleep(2)
                if os.path.isfile("../Attendance/Attendance_23-07-2023.csv"):
                    with open("../Attendance/Attendance_23-07-2023.csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                    csvfile.close()
                    messagebox.showinfo(title="Attendance",
                                        message="Attendance taken for " + attendance[0])
                    print("Attendance taken for", attendance[0])
            elif num == 1:
                messagebox.showwarning(title="Attendance",
                                       message="Already marked attendance for" + attendance[0])
                print(f"Already marked attendance for "
                      f"{attendance[0]}")
            else:
                sys.stderr.write("")
                messagebox.showerror(title="Attendance",
                                     message="Incorrect device date")
                print("Incorrect device date")
        if k == ord('q'):
            print("Exiting . . .")
            break
    video.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.title("Attendance System")
window.geometry("700x500")
window.configure(bg='#333333')
frame = tk.Frame(bg='#333333')
label = tk.Label(frame, text="Add student", bg='#333333',
                 fg="#FF3399", font=("Arial", 30))
label.pack()
mat_no = tk.Entry(frame, font=("Arial", 16))


def ad_std():
    if mat_no.get():
        add_std(mat_no.get())
    else:
        messagebox.showwarning(title="ADD Attendance",
                               message="NO MATRIC NUMBER INPUTED")


button = tk.Button(frame, text="ADD Student", bg="#FF3399",
                   fg="#FFFFFF", font=("Arial", 16), command=ad_std)
mat_no.pack()
button.pack()
# ------------------------------------------------------
# ------------------------------------------------------
label_m = tk.Label(frame, text="Mark Attendance", bg='#333333',
                   fg="#FF3399", font=("Arial", 30))
label_m.pack()
course = tk.Entry(frame, font=("Arial", 16))


def mk_atd():
    if course.get():
        mark_attendance(course.get())
    else:
        messagebox.showwarning(title="MARK Attendance", message="NO COUSRE INPUTED")


button_m = tk.Button(frame, text="MARK ATTENDANCE", bg="#FF3399",
                     fg="#FFFFFF", font=("Arial", 16), command=mk_atd)
course.pack()
button_m.pack()

frame.pack()
window.mainloop()

