from datetime import datetime, date
import csv
import sys
import os
import re


# returns True if the course follows the specified format
def check_course(string):
    pattern = r'^[A-Z]{3}\d{3}$'
    mch = re.match(pattern, string)
    if mch:
        return True
    else:
        return False


# returns True if a matric number follows the specified format
def check_mat_no(string):
    pattern = r'^D/\d{4}/\d{2}/\d{3}$'
    mch = re.match(pattern, string)
    if mch:
        return True
    else:
        return False


# function to initialise default values for time manipulation
def default_time_date():
    tod = datetime.now()
    tod = tod.date()
    tod = date.strftime(tod, "%d-%m-%Y")
    date_list = datetime.strptime("01-01-2000", "%d-%m-%Y")
    date_list = date_list.date()
    date_list = date.strftime(date_list, "%d-%m-%Y")
    return tod, date_list


# function to convert string(date) into a date element formate
def ret_date_fom(d_st):
    ret = datetime.strptime(d_st, "%d-%m-%Y")
    ret = ret.date()
    ret = date.strftime(ret, "%d-%m-%Y")
    return ret


def files_read(file_path, mat_n, course):
    # get the default date and time
    today, default = default_time_date()
    # check if the csv file is present
    if not os.path.isfile(file_path):
        return -1, -1
    # open the file since the file exist
    with open(file_path, "rt") as f:
        csv_reader = csv.reader(f)
        # skip the header of the file
        next(csv_reader)
        # read the file one line at a time
        for lines in csv_reader:
            # getting the most recent attendance date for the student on the particular course
            if mat_n == lines[0] and course == lines[3]:
                high = ret_date_fom(lines[2])
                if default < high:
                    default = high
    # close the opened file
    f.close()
    # split the dates to get single data in day, month and year
    td, t_m, t_y = today.split("-")
    td, t_m, t_y = str(td), str(t_m), str(t_y)
    d_d, d_m, d_y = default.split("-")
    d_d, d_m, d_y = str(d_d), str(d_m), str(d_y)
    # change dates to the format YYYY/MM/DD
    cont_t = t_y + t_m + td
    cont_d = d_y + d_m + d_d
    # return int type for easier manipulation
    return int(cont_t), int(cont_d)


def real(mat_no, course):
    # checks if the argument are valid arguments
    if not mat_no or not course:
        sys.stderr.write("Arguments to real function empty")
        return False
    # gets default date
    t, d = files_read("Attendance/Attendance_23-07-2023.csv", mat_no, course)
    # if the file does not exist
    if t == -1 and d == -1:
        return -2
    # if date today == default
    if t == d:
        # print("the last attendance was marked today")
        return 1
    # if date today > default
    elif t > d:
        # print("attendance marked")
        return 0
    # if date today < default
    elif t < d:
        # print("last attendance marked in the future")
        return -1
