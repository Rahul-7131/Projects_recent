import streamlit as st
from fastai.vision.all import *

def load_learner_(path):
    return load_learner(path)

def load_img(path):
    image = Image.open(path)
    w, h = image.size
    dim = (500, int((h*500)/w))
    return image.resize(dim)

learn = load_learner_('export.pkl')

st.markdown("# Ronaldo-Messi Classifier")
st.markdown("Upload an image and the classifier will tell you whether its a picture of Ronaldo OR Messi.")
st.markdown("If You upload any image other than the pictures of Ronaldo/Messi then it will you the similarity of your picture with Ronaldo's or Messi's Picture")

file_bytes = st.file_uploader("Upload a file", type=("png", "jpg", "jpeg", "jfif"))
if file_bytes:
    img = load_img(file_bytes)
    st.image(img)
    
    submit = st.button('Predict!')
    if submit:
        pred, pred_idx, probs = learn.predict(PILImage(img))
        st.markdown(f'Prediction: **{pred}**; Similarity: **{probs[pred_idx]:.04f}**')
