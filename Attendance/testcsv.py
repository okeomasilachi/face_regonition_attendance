from datetime import *
import csv


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


def real(time_t, time_d):
    d = int(time_d)
    t = int(time_t)
    # if year for today == default
    if t == d:
        # print("the last attendance was marked today")
        return False
    # if year for today > default
    elif t > d:
        # print("attendance marked")
        return True
    # if year for today < default
    elif t < d:
        # print("last attendance marked in the future")
        return False


math_n, course = "98790", "ENG555"
today, default = default_time_date()

with open("Attendance_23-07-2023.csv", "rt") as f:
    csv_reader = csv.reader(f)
    # skip the header of the file
    next(csv_reader)
    for lines in csv_reader:
        if math_n == lines[0] and course == lines[3]:
            high = ret_date_fom(lines[2])
            print(f"{high}============{default}")
            if default < high:
                default = high


t_d, t_m, t_y = today.split("-")
t_d, t_m, t_y = str(t_d), str(t_m), str(t_y)
d_d, d_m, d_y = default.split("-")
d_d, d_m, d_y = str(d_d), str(d_m), str(d_y)

cont_t = t_y + t_m + t_d
cont_d = d_y + d_m + d_d

if real(cont_t, cont_d):
    print(f"Attendance Marked for {math_n} for the course {course} on {today}")
