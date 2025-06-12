import streamlit as st
import helper

st.title("Image Digit classifier")
uploaded_file = st.file_uploader(label='upload image', type=["jpg", "jpeg", "png"])

if st.button("predict"):
    if uploaded_file is not None:
        filename = uploaded_file.name
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(filename)
        img_vec = helper.img_to_vec(filename)
        pred,prob = helper.predict_class(img_vec)
        st.write(f"Pred = {pred}, prob = {prob}")