import streamlit as st

st.set_page_config(page_title="Streamlit Cloud Test")

st.title("Streamlit Deployment Test")

st.success("Application loaded successfully!")

option = st.radio(
    "Select an option",
    ["Logistic Regression", "Decision Tree", "SVM"]
)

st.write("Selected model:", option)

st.metric("Test Accuracy", "92.21%")