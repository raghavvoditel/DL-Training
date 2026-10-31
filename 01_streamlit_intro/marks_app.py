# A small Streamlit app: enter marks for 5 subjects, see total, grade and a chart.
# Run with:  streamlit run marks_app.py

import pandas as pd
import streamlit as st

st.title("Student Marks Calculator")
st.write("Enter your marks (out of 100) and the app works out your result.")

name = st.text_input("Your name")

subjects = ["Maths", "Physics", "Chemistry", "English", "Computer"]
marks = []
for subject in subjects:
    mark = st.slider(subject, min_value=0, max_value=100, value=50)
    marks.append(mark)


def get_grade(percent):
    if percent >= 90:
        return "A+"
    elif percent >= 75:
        return "A"
    elif percent >= 60:
        return "B"
    elif percent >= 40:
        return "C"
    else:
        return "Fail"


if st.button("Calculate"):
    total = sum(marks)
    percent = total / len(subjects)
    grade = get_grade(percent)

    st.subheader(f"Result for {name or 'student'}")
    st.write(f"Total: {total} / {len(subjects) * 100}")
    st.write(f"Percentage: {percent:.1f}%")
    st.write(f"Grade: {grade}")

    # A table and a bar chart of the marks
    table = pd.DataFrame({"Subject": subjects, "Marks": marks})
    st.table(table)
    st.bar_chart(table, x="Subject", y="Marks")

    if grade == "Fail":
        st.error("You need at least 40% to pass.")
    else:
        st.success("Well done, you passed!")

# Optional: upload your own CSV and look at it
st.divider()
uploaded = st.file_uploader("Or upload a CSV file to view it", type="csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("First rows of your file:")
    st.dataframe(df.head())
