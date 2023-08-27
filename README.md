# face regonition attendance

This README provides an overview of the Python code that implements an Attendance System using the K-Nearest Neighbors (KNN) algorithm and OpenCV. The system captures images, recognizes faces, and records attendance based on recognized faces. The code is organized into a class-based structure and uses the Tkinter library for the graphical user interface.

### Table of Contents
* **_Introduction_**
* **_Requirements_**
* **_Installation_**
* **_Usage_**
* **_Features_**
* **_Contributing_**
* **_License_**

## Introduction
The Attendance System is designed to automate the process of taking attendance using facial recognition technology. It captures images from a webcam, detects faces, and classifies them using the KNN algorithm. The system allows you to log in, add students' data, mark attendance, and visualize attendance records.

## Requirements
To run this code, you'll need the following dependencies:

* Python 3.x
* scikit-learn library (for KNN classifier)
* opencv-python library (for image processing)
* numpy library (for numerical operations)
* tkinter library (for GUI)
* streamlit library (for displaying attendance records)

You can install these dependencies using the following command:

`pip install scikit-learn opencv-python numpy tkinter streamlit`

**Clone the repository**

`git clone https://github.com/okeomasilchi/face_regonition_attendance.git`

`cd face_regonition_attendance`

Install the required dependencies as mentioned in the Requirements section.

## Usage
Run the main Python script to start the Attendance System:

`python3 gui.py`

The graphical user interface (GUI) will appear, prompting you to enter your username and password for login.

After successful login, you can interact with the system to add students and mark attendance.

## Features
* **Login:** Secure login functionality to access the attendance system.
* **Add Students:** Capture and save student images along with their matriculation numbers.
* **Mark Attendance:** Recognize faces in real-time using the webcam, mark attendance, and record timestamp and date.
* **Streamlit Integration:** The attendance records can be displayed using Streamlit by running the web.py script in a subprocess.

## Contributing
Contributions to this project are welcome. Feel free to open issues, submit pull requests, and suggest improvements.

* Fork the repository.
* Create a new branch.
* Implement your changes.
* Test thoroughly.
* Submit a pull request explaining your changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

### This README is a brief overview of the Attendance System code. For more detailed information and instructions, please refer to the actual code files and comments within the code.
