import sys
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import pickle
import csv
import time
from datetime import datetime
from Attendance.onyedibia import real, check_course, check_mat_no
import numpy as np
import tkinter as tk
from tkinter import messagebox
import subprocess
from pymongo import MongoClient

MONGO_URI = "YOUR_MONGODB_ATLAS_URI"
DB_NAME = "face"
COLLECTION_NAME = "face_ats"

class Application:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System By OBI DORIS NKEMAKOLA")

        self.login_frame = tk.Frame(self.root)
        self.login_frame.pack(padx=20, pady=20)

        self.username_label = tk.Label(self.login_frame, text="Username:")
        self.username_label.grid(row=0, column=0, sticky="w")
        self.username_entry = tk.Entry(self.login_frame)
        self.username_entry.grid(row=0, column=1, pady=10)

        self.password_label = tk.Label(self.login_frame, text="Password:")
        self.password_label.grid(row=1, column=0, sticky="w")
        self.password_entry = tk.Entry(self.login_frame, show="*")
        self.password_entry.grid(row=1, column=1, pady=10)

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login)
        self.login_button.grid(row=5, columnspan=3, pady=10)       

        self.main_frame = tk.Frame(self.root)
        
        self.message_label = tk.Label(self.main_frame, font=("Arial", 20), text="Nkemakola Attendance System")
        self.message_label.pack(pady=100)

         # starts a subprocess for the attendance to be displayed
        subprocess.Popen(["streamlit", "run", "web.py"])

        self.final_frame = tk.Frame(self.root)

        self.entry_label1 = tk.Label(self.final_frame, text="Matric Number: ")
        self.entry_label1.grid(row=0, column=0, sticky="w")
        self.entry1 = tk.Entry(self.final_frame)
        self.entry1.grid(row=0, column=1, pady=10)

        self.entry_label2 = tk.Label(self.final_frame, text="Course Code: ")
        self.entry_label2.grid(row=1, column=0, sticky="w")
        self.entry2 = tk.Entry(self.final_frame)
        self.entry2.grid(row=1, column=1, pady=10)

        self.button1 = tk.Button(self.final_frame, text="Add Student", command=self.add_std)
        self.button1.grid(row=0, column=3, pady=10)

        self.button2 = tk.Button(self.final_frame, text="Mark Attendance", command=self.mark_attendance)
        self.button2.grid(row=1, column=3, pady=8)

    def login(self):
        """login
            Handles the login logic for the Application
        """
        if self.username_entry.get() == "okeoma" and self.password_entry.get() == "okeoma":
            self.login_frame.destroy()
            self.main_frame.pack(padx=20, pady=20)
            self.root.after(2000, self.show_final_frame)
        else:
            self.username_entry.insert(0, "")
            self.password_entry.insert(0, "")
            messagebox.showerror("Login Failed", "Invalid credentials. Please try again.")

    def show_final_frame(self):
        self.main_frame.destroy()
        self.final_frame.pack(padx=50, pady=90)

    def add_std(self):
        
        name = self.entry1.get()
        if check_mat_no(str(name)):
            video = cv2.VideoCapture(0)
            facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

            faces_data = []

            i = 0

            while True:
                ret, frame = video.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y + h, x:x + w, :]
                    resized_img = cv2.resize(crop_img, (50, 50))
                    if len(faces_data) <= 10 and i % 10 == 0:
                        faces_data.append(resized_img)
                    i = i + 1
                    cv2.putText(frame, str(len(faces_data)), (50, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1, (50, 50, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
                cv2.imshow("Frame", frame)
                k = cv2.waitKey(1)
                if k == ord('q') or len(faces_data) == 10:
                    break
            video.release()
            cv2.destroyAllWindows()

            faces_data = np.asarray(faces_data)
            faces_data = faces_data.reshape(10, -1)

            if 'mat_no.pkl' not in os.listdir('data/'):
                names = [name] * 10
                with open('data/mat_no.pkl', 'wb') as f:
                    pickle.dump(names, f)
            else:
                with open('data/mat_no.pkl', 'rb') as f:
                    names = pickle.load(f)
                names = names + [name] * 10
                with open('data/mat_no.pkl', 'wb') as f:
                    pickle.dump(names, f)

            if 'faces_data.pkl' not in os.listdir('data/'):
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces_data, f)
                    print("student added to base")
            else:
                with open('data/faces_data.pkl', 'rb') as f:
                    faces = pickle.load(f)
                    print("student added to base")
                faces = np.append(faces, faces_data, axis=0)
                with open('data/faces_data.pkl', 'wb') as f:
                    pickle.dump(faces, f)
                    print("student added to base")
        else:
            messagebox.showerror("Add Student", F"Invalid Matric number '{name}'\n"
                                                    F"Please follow the format 'D/1234/12/123'\n")

    def mark_attendance(self):
        course = self.entry2.get()
        if check_course(str(course)):
            video = cv2.VideoCapture(0)
            facedetect = (cv2.CascadeClassifier
                          ('data/haarcascade_frontalface_default.xml'))

            with open('data/mat_no.pkl', 'rb') as w:
                self.labels = pickle.load(w)
            with open('data/faces_data.pkl', 'rb') as f:
                self.faces = pickle.load(f)

            print('Shape of Faces matrix --> ', self.faces.shape)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(self.faces, self.labels)

            col_names = ['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE']

            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            collection = db[COLLECTION_NAME]
            
            while True:
                ret, frame = video.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    crop_img = frame[y:y + h, x:x + w, :]
                    resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                    self.output = knn.predict(resized_img)
                    ts = time.time()
                    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                    cv2.putText(frame, str(self.output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
                    self.attendance = [str(self.output[0]), str(timestamp), date, course]
                cv2.imshow("Frame", frame)
                k = cv2.waitKey(1)
                if k == ord('m'):
                    # Same attendance marking logic as before (without needing another cv2.waitKey())
                    num = real(self.output[0], course)
                    if num == -2:
                        time.sleep(2)
                        with open("Attendance/Attendance_23-07-2023.csv", "+a") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(col_names)
                            writer.writerow(self.attendance)
                        csvfile.close()
                        attendance_data = {
                            "MATRIC NUMBER": str(self.output[0]),
                            "TIME": timestamp,
                            "DATE": date,
                            "COURSE": course
                        }
                        collection.insert_one(attendance_data)
                        
                        print("Attendance taken for", self.attendance[0])
                    elif num == 0:
                        time.sleep(2)
                        if os.path.isfile("Attendance/Attendance_23-07-2023.csv"):
                            with open("Attendance/Attendance_23-07-2023.csv", "+a") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(self.attendance)
                            csvfile.close()
                            attendance_data = {
                                "MATRIC NUMBER": str(self.output[0]),
                                "TIME": timestamp,
                                "DATE": date,
                                "COURSE": course
                            }
                            collection.insert_one(attendance_data)
                            print("Attendance taken for", self.attendance[0])
                    elif num == 1:
                        print(f"Already marked attendance for "
                              f"{self.attendance[0]}")
                    else:
                        sys.stderr.write("")
                        print("Incorrect device date")
                if k == ord('q'):
                    print("Exiting . . .")
                    break
            video.release()
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Mark Attendance", F"Unknown course format '{course}'\n"
                                                    F"Please follow the format 'ABC123'\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
