#!./venv/scripts/python
import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import os

count = st_autorefresh(interval=2000, limit=100)

st.write("Attendance system by Doris Nkemakola Onyedibia")

if not os.path.isfile("Attendance/Attendance_23-07-2023.csv"):
    st.error(f"No Attendance file present. "
             f"The site will refresh automatically once the file is present")
else:
    df = pd.read_csv("Attendance/Attendance_23-07-2023.csv")

    # Check if the column is present in the DataFrame
    if 'COURSE' not in df.columns:
        st.error("The 'COURSE' column is not present in the DataFrame.")
    elif 'MATRIC NUMBER' not in df.columns:
        st.error("The 'MATRIC NUMBER' column is not present in the DataFrame.")
    elif 'DATE' not in df.columns:
        st.error("The 'DATE' column is not present in the DataFrame.")
    else:
        # Get the list of unique rows
        unique_courses = df["COURSE"].unique()
        unique_student = df["MATRIC NUMBER"].unique()
        unique_date = df["DATE"].unique()

        selected_option = st.selectbox("Select An Option", ["None selected", "Course", "Student", "Date"])
        # Create a dropdown widget to select the course
        if selected_option == "Course":
            selected_course = st.selectbox("Select Course", unique_courses)

            # Filter the DataFrame based on the selected course
            course_df = df[df["COURSE"] == selected_course]
     
            # Display attendance for the selected course
            st.write(f"Attendance for Course: {selected_course}")
            st.dataframe(course_df.style.highlight_max(axis=0))
        elif selected_option == "Student":
            selected_std = st.selectbox("Select Student", unique_student)

            # Filter the DataFrame based on the selected course
            student_df = df[df["MATRIC NUMBER"] == selected_std]

            # Display attendance for the selected course
            st.write(f"Attendance for Student: {selected_std}")
            st.dataframe(student_df.style.highlight_max(axis=0))
        elif selected_option == "Date":
            selected_date = st.selectbox("Select Date", unique_date)

            # Filter the DataFrame based on the selected course
            date_df = df[df["DATE"] == selected_date]
            # Display attendance for the selected course
            st.write(f"Attendance for Date: {selected_date}")
            st.dataframe(date_df.style.highlight_max(axis=0))
        else:
            st.write(f"Nothing is selected\n"
                     f"Please select An Option\n")
