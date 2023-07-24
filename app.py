""" import streamlit as st
import pandas as pd
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime


# Function for Face Data Adding
def add_face_data():
	video=cv2.VideoCapture(0)
	facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

	faces_data=[]

	i=0

	name=input("Enter Matric Number: ")

	while True:
		ret,frame=video.read()
		gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces=facedetect.detectMultiScale(gray, 1.3 ,5)
		for (x,y,w,h) in faces:
			crop_img=frame[y:y+h, x:x+w, :]
			resized_img=cv2.resize(crop_img, (50,50))
			if len(faces_data)<=10 and i%10==0:
				faces_data.append(resized_img)
			i=i+1
			cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
			cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
		cv2.imshow("Frame",frame)
		k=cv2.waitKey(1)
		if k==ord('q') or len(faces_data)==10:
			break
	video.release()
	cv2.destroyAllWindows()

	faces_data=np.asarray(faces_data)
	faces_data=faces_data.reshape(10, -1)

	if 'mat_no.pkl' not in os.listdir('data/'):
		names=[name]*10
		with open('data/mat_no.pkl', 'wb') as f:
			pickle.dump(names, f)
	else:
		with open('data/mat_no.pkl', 'rb') as f:
			names=pickle.load(f)
		names=names+[name]*10
		with open('data/mat_no.pkl', 'wb') as f:
			pickle.dump(names, f)

	if 'faces_data.pkl' not in os.listdir('data/'):
		with open('data/faces_data.pkl', 'wb') as f:
			pickle.dump(faces_data, f)
			print("student added to base")
	else:
		with open('data/faces_data.pkl', 'rb') as f:
			faces=pickle.load(f)
			print("student added to base")
		faces=np.append(faces, faces_data, axis=0)
		with open('data/faces_data.pkl', 'wb') as f:
			pickle.dump(faces, f)
			print("student added to base")

# Function for Face Recognition
def face_recognition():
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

# Function for Displaying Attendance
def display_attendance():
	ts = time.time()
	date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
	timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

	from streamlit_autorefresh import st_autorefresh

	count = st_autorefresh(interval=2000, limit=100)

	st.write("Attendance system by Doris Nkemakola Onyedibia")

	df = pd.read_csv("Attendance/Attendance_" + date + ".csv")

	# Check if the 'course' column is present in the DataFrame
	if 'COURSE' not in df.columns:
		st.error("The 'COURSE' column is not present in the DataFrame.")
	else:
		# Get the list of unique courses
		unique_courses = df["COURSE"].unique()

		# Create a dropdown widget to select the course
		selected_course = st.selectbox("Select Course", unique_courses)

		# Filter the DataFrame based on the selected course
		course_df = df[df["COURSE"] == selected_course]

		# Display attendance for the selected course
		st.write(f"Attendance for Course: {selected_course}")
		st.dataframe(course_df.style.highlight_max(axis=0))



# Function to Save Face Data
def save_face_data(names, faces_data):
    with open('data/mat_no.pkl', 'wb') as f:
        pickle.dump(names, f)

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)

# Function to Load Face Data
def load_face_data():
    with open('data/mat_no.pkl', 'rb') as f:
        names = pickle.load(f)

    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)

    return names, faces_data

# Function to Save Attendance Records
def save_attendance(attendance):
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    filename = f"Attendance/Attendance_{date}.csv"
    exist = os.path.isfile(filename)

    with open(filename, '+a') as csvfile:
        writer = csv.writer(csvfile)
        if not exist:
            writer.writerow(['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE'])
        writer.writerow(attendance)

# main funtion
def main():
    st.write("# Attendance System by Doris Nkemakola Onyedibia")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose an option", ["Add Face Data", "Face Recognition", "Display Attendance"])

    if app_mode == "Add Face Data":
        st.title("Add Face Data")
        add_face_data()

    elif app_mode == "Face Recognition":
        st.title("Face Recognition")
        face_recognition()

    elif app_mode == "Display Attendance":
        st.title("Display Attendance")
        display_attendance()

if __name__ == "__main__":
    main()

 """
 
import streamlit as st
import pandas as pd
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

# Function for Face Data Adding
def add_face_data():
    st.title("Add Face Data")
    name = st.text_input("Enter Matric Number:")

    # Initialize webcam video capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0

    st.write("Please capture 10 samples of your face by clicking the 'Capture' button:")
    if st.button("Capture"):
        while len(faces_data) < 10:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            # Display the frame with the face count
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Release video capture and destroy windows
        video.release()
        cv2.destroyAllWindows()

        # Convert faces_data to NumPy array and reshape
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
                st.write("Student added to database")
        else:
            with open('data/faces_data.pkl', 'rb') as f:
                faces = pickle.load(f)
                st.write("Student added to database")
            faces = np.append(faces, faces_data, axis=0)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces, f)
                st.write("Student added to database")


# Function for Face Recognition
def face_recognition():
    st.title("Face Recognition")
    course = st.text_input("Enter Course Code:")

    # Initialize webcam video capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

    with open('data/mat_no.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    st.write('Shape of Faces matrix --> ', FACES.shape)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(FACES, LABELS)

    COL_NAMES = ['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE']

    # Dictionary to store the last attendance time for each face ID and course
    last_attendance_time = {}

    st.write("Please wait for a face to be recognized...")
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            attendance = [str(output[0]), timestamp, date, course]
        # Display the frame with face recognition results
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        k = cv2.waitKey(1)
        if k == ord('m'):
            # Same attendance marking logic as before (without needing another cv2.waitKey())
            if (str(output[0]), course) in last_attendance_time:
                last_time = last_attendance_time[(str(output[0]), course)]
                time_diff = ts - last_time
                if time_diff >= 300:  # 5 minutes = 300 seconds
                    time.sleep(1)
                    if exist:
                        with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(attendance)
                        csvfile.close()
                    else:
                        with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(COL_NAMES)
                            writer.writerow(attendance)
                        csvfile.close()
                    st.write("Attendance taken for ", str(output[0]))
                else:
                    st.write(f"Already marked attendance for {str(output[0])}")
            else:
                time.sleep(1)
                if exist:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(attendance)
                    csvfile.close()
                else:
                    with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                    csvfile.close()
                st.write("Attendance taken for", str(output[0]))
        if k == ord('q'):
            break

    # Release video capture and destroy windows
    video.release()
    cv2.destroyAllWindows()


# Function for Displaying Attendance
def display_attendance():
    st.title("Display Attendance")
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

    df = pd.read_csv("Attendance/Attendance_" + date + ".csv")

    # Check if the 'COURSE' column is present in the DataFrame
    if 'COURSE' not in df.columns:
        st.error("The 'COURSE' column is not present in the DataFrame.")
    else:
        # Get the list of unique courses
        unique_courses = df["COURSE"].unique()

        # Create a dropdown widget to select the course
        selected_course = st.selectbox("Select Course", unique_courses)

        # Filter the DataFrame based on the selected course
        course_df = df[df["COURSE"] == selected_course]

        # Display attendance for the selected course
        st.write(f"Attendance for Course: {selected_course}")
        st.dataframe(course_df.style.highlight_max(axis=0))


# Function to Save Face Data
def save_face_data(names, faces_data):
    with open('data/mat_no.pkl', 'wb') as f:
        pickle.dump(names, f)

    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)


# Function to Load Face Data
def load_face_data():
    with open('data/mat_no.pkl', 'rb') as f:
        names = pickle.load(f)

    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)

    return names, faces_data


# Function to Save Attendance Records
def save_attendance(attendance):
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    filename = f"Attendance/Attendance_{date}.csv"
    exist = os.path.isfile(filename)

    with open(filename, '+a') as csvfile:
        writer = csv.writer(csvfile)
        if not exist:
            writer.writerow(['MATRIC NUMBER', 'TIME', 'DATE', 'COURSE'])
        writer.writerow(attendance)

def main():
    st.write("# Attendance System by Doris Nkemakola Onyedibia")

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose an option", ["Select one option", "Display Attendance", "Add Face Data", "Face Recognition"])

    if app_mode == "Display Attendance":
        display_attendance()

    elif app_mode == "Face Recognition":
        face_recognition()

    elif app_mode == "Add Face Data":
        add_face_data()

if __name__ == "__main__":
    main()
