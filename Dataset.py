import pandas as pd
import random

data = []

for i in range(1000):

    attendance = random.randint(40,100)        # percentage
    study_hours = random.randint(1,10)         # hours per day
    previous_marks = random.randint(35,100)    # marks out of 100
    assignments = random.randint(0,5)          # assignments completed

    # Score out of 100
    score = (attendance/100)*20 + (study_hours/10)*20 + (previous_marks/100)*40 + (assignments/5)*20

    # Risk classification
    if score >= 75:
        risk = "Low"
    elif score >= 50:
        risk = "Medium"
    else:
        risk = "High"

    data.append([attendance,study_hours,previous_marks,assignments,score,risk])

df = pd.DataFrame(data, columns=[
    "Attendance",
    "Study_Hours",
    "Previous_Marks",
    "Assignments_Completed",
    "Score",
    "Risk_Level"
])

df.to_csv("student_dataset.csv",index=False)

print("Dataset Generated Successfully")
print(df.head())