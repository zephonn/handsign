import streamlit as st
import tensorflow
from keras.models import load_model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from skimage.transform import resize

st.markdown("<div style='background-color: #333333;border: 3px solid #219C90; border-radius:100px;'><h1 style='text-align:center;color:white;'>Test Model</h1></div",unsafe_allow_html=True)
st.write(' ')
st.write(' ')
st.write('-we build a convolutional neural network and add a classifier on top of it, to recognize gesture handsigns.Convolutional neural networks are very powerful in image classification and recognition tasks. CNN models learn features of the training images.')
st.header('preview image and label')
model=load_model('model2.h5')
g=['9','R','B','S','1','D','C','4','O','W','V','Z','N','6','0','8','Y','7','U','M','Q', 'X','5','_','T','F','E','G','I','K','3','L','J','A','2','P','H']
yy=sorted(g)

o=None
raa=st.selectbox(label='select any file from my testing folder',options=g,index=None)
if raa=='S':
    o=Image.open('s.jpg')
elif raa=='A':
    o=Image.open('a.jpg')
elif raa=='B':
    o=Image.open('b.jpg')
elif raa=='C':
    o=Image.open('c.jpg')
elif raa=='D':
    o=Image.open('d.jpg')
elif raa=='E':
    o=Image.open('e.jpg')
elif raa=='F':
    o=Image.open('f.jpg')
elif raa=='G':
    o=Image.open('g.jpg')
elif raa=='H':
    o=Image.open('h.jpg')
elif raa=='I':
    o=Image.open('i.jpg')
elif raa=='j':
    o=Image.open('j.jpg')
elif raa=='K':
    o=Image.open('k.jpg')
elif raa=='L':
    o=Image.open('l.jpg')
elif raa=='M':
    o=Image.open('m.jpg')
elif raa=='N':
    o=Image.open('n.jpg')
elif raa=='O':
    o=Image.open('o.jpg')
elif raa=='P':
    o=Image.open('p.jpg')
elif raa=='Q':
    o=Image.open('q.jpg')
elif raa=='R':
    o=Image.open('r.jpg')
elif raa=='T':
    o=Image.open('t.jpg')
elif raa=='U':
    o=Image.open('u.jpg')
elif raa=='V':
    o=Image.open('v.jpg')
elif raa=='W':
    o=Image.open('w.jpg')
elif raa=='x':
    o=Image.open('x.jpg')
elif raa=='Y':
    o=Image.open('y.jpg')
elif raa=='Z':
    o=Image.open('z.jpg')
elif raa=='0':
    o=Image.open('0.jpg')
elif raa=='1':
    o=Image.open('1 1.jpg')
elif raa=='_':
    o=Image.open('space.jpg')
elif raa=='2':
    o=Image.open('2.jpg')
elif raa=='3':
    o=Image.open('3.jpg')
elif raa=='4':
    o=Image.open('4.jpg')
elif raa=='5':
    o=Image.open('5.jpg')
elif raa=='6':
    o=Image.open('6.jpg')
elif raa=='7':
    o=Image.open('7.jpg')
elif raa=='8':
    o=Image.open('8.jpg')
elif raa=='9':
    o=Image.open('9.jpg')


ff = ['B', 'U', '9', 'W', 'E', 'T', '4', 'S', 'R', 'K', '1', 'D', 'Y', 'F', 'V', '_', 'M', '7', 'A', '8',
                  'C', 'N', 'P', '2', 'X', 'L', '0', 'H', 'I', '5', 'G', 'Z', '3', 'J', 'O', '6', 'Q']

g=sorted(ff)
if o is not None:
    st.image(o,caption='u  selected %s handsign'%raa)
    img_array = np.array(o)
    ima_re = resize(img_array, (150, 150, 2))
    ima_ex = np.expand_dims(ima_re, axis=0)
pre = st.button('predict handsign ')
if pre:

    model = load_model('model2.h5')
    k = model.predict(ima_ex)
    ar = np.argmax(k)
    ii = np.transpose(k)
    l = 100 * (ii[ar].item())
    va = str(g[ar])
    if va==raa:
        st.snow()
        st.success('ACCURATE',icon='🎯')
    else:
        st.error('INACCURATE',icon='❌')
    st.header(va)
    st.subheader(
        '-The model predicted the handsign captured will be %s with a chance of %d percentage' % (va, l))

st.subheader('OR')
st.write('upload any imagee from your device to predict hand sign')
fil=st.file_uploader('choose any image file',type=['jpg','png','jpeg'])

if fil is not None:
    image=Image.open(fil)
    image_array=np.array(image)
    ima_re = resize(image_array, (150, 150, 2))
    ima_ex = np.expand_dims(ima_re, axis=0)
    qq=st.text_input('enter the hand sign label')
mm=st.button('predict handsign')
if mm:
    model = load_model('model2.h5')
    k = model.predict(ima_ex)
    ar = np.argmax(k)
    ii = np.transpose(k)
    l = 100 * (ii[ar].item())
    va = str(g[ar])
    st.header(va)
    st.subheader('-The model predicted the handsign imported will be %s with a chance of %d percentage' % (va, l))

    if qq==va:
        st.snow()
        st.success('ACCURATE',icon='🎯')
    else:
        st.error('INACCURATE',icon='❌')
