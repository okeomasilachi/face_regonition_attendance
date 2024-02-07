import tkinter as tk
import pandas as pd

# Create a window
window = tk.Tk()

# Read the attendance data from a CSV file
df = pd.read_csv("Attendance/Attendance_23-07-2023.csv")

# Get the list of unique courses, students, and dates
unique_courses = df["COURSE"].unique()
unique_students = df["MATRIC NUMBER"].unique()
unique_dates = df["DATE"].unique()

# Create a dropdown menu to select the option
option_menu = tk.Menubutton(window, text="Select An Option")
option_menu.menu = tk.Menu(option_menu, tearoff=0)

# Add options to the menu
option_menu.menu.add_command(label="Course", command=lambda: print("Course"))
option_menu.menu.add_command(label="Student", command=lambda: print("Student"))
option_menu.menu.add_command(label="Date", command=lambda: print("Date"))
option_menu.pack()

# Create a label to display the selected option
selected_option_label = tk.Label(window, text="Selected Option: None selected")
selected_option_label.pack()

# Create a function to update the selected option label
def update_selected_option_label(event):
    global selected_option_label
    selected_option = option_menu.menu.get_active()
    selected_option_label.config(text=f"Selected Option: {selected_option}")

# Bind the event to the dropdown menu
option_menu.menu.bind("<<MenuSelect>>", update_selected_option_label)

# Start the main loop
window.mainloop()
