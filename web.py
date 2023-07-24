import streamlit as st
import pandas as pd
import time
from datetime import datetime

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

