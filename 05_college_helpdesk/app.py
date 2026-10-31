# Web version of the helpdesk.
# Run with:  streamlit run app.py

import streamlit as st

from helpdesk import Helpdesk

st.title("College Helpdesk")
st.write("Ask about the library, exams, hostel or fees.")


@st.cache_resource  # build the TF-IDF index only once, not on every click
def get_helpdesk():
    return Helpdesk()


desk = get_helpdesk()

question = st.text_input("Your question", placeholder="e.g. What is the library fine?")
if question:
    result = desk.answer(question)
    if result["source"]:
        st.success(result["answer"])
        st.caption(f"Source: {result['source']}  |  match score: {result['score']:.2f}")
    else:
        st.warning(result["answer"])
