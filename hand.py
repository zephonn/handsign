
import cv2
import keras.models
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
import pickle as pk
import tensorflow as tf
import joblib
import keras
import dill
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

st.markdown("<div style='background-color: #333333;border: 3px solid #219C90; border-radius:100px;'><h1 style='text-align:center;color:white;'>Talking Hand</h1></div",unsafe_allow_html=True)
st.write(' ')
st.write(' ')
st.subheader('Hand Signs 👈')

ff = ['B', 'U', '9', 'W', 'E', 'T', '4', 'S', 'R', 'K', '1', 'D', 'Y', 'F', 'V', '_', 'M', '7', 'A', '8',
                  'C', 'N', 'P', '2', 'X', 'L', '0', 'H', 'I', '5', 'G', 'Z', '3', 'J', 'O', '6', 'Q']
tt=sorted(ff)
num=st.selectbox(label='-select any letter to see how the hand sign look like',options=['letters','numbers'],index=None)
if num=='letters':
    st.image('letter.jpg')
if num=='numbers':
    st.image('number.jpg')
st.subheader('Test by Your Own')
t=st.toggle('proceed')
cam=None

if t:
    cam=st.camera_input('take a photo of your hand sign')
    if cam is not None:
        st.write('image captured')
        image=Image.open(cam)
        img_array=np.array(image)
        ima_re=resize(img_array,(150,150,2))
        ima_ex=np.expand_dims(ima_re,axis=0)
        st.write("now let's see how good our model is")
        pre = st.button('predict hand gesture')
        if pre:
            model=load_model('model2.h5')
            k = model.predict(ima_ex)
            ar = np.argmax(k)
            ii=np.transpose(k)
            l=100*(ii[ar].item())
            va=str(tt[ar])
            st.header(va)
            st.subheader('-The model predicted the handsign captured will be %s with a chance of %d percentage'%(va,l))










