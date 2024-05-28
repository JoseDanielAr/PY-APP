import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
print("se leyo librerias")
import gdown
import os
import numpy as np

file_url = 'https://drive.google.com/uc?id=1locdBqp8abUEl_NB_x8Q7tLEYOXute92'
output_filename = 'model2.h5'

gdown.download(file_url, output_filename, quiet=False)

model = load_model(output_filename)

#model = load_model('model2.h5')
st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Modelo para Clasificación de Imagenes</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:26px; color: black;'>Este programa permite clasificar la imagen tomada en 10 categorias:</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 26px; color: black; text-align: center'>Cuchara, Cucharon, Espatula, Filtro, Encendedor, cuchillo, tijeras, banano, manzana y huevo</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:26px; color: black;'>Para realizar la predicción, tome la foto</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'>O</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 56px; color: black; text-align: center'></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file)
    print("existe picture")
    with open ('input.jpg','wb') as file:
          file.write(uploaded_file.getbuffer())
          print("se guardo foto")
    img = image.load_img('input.jpg', target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class=np.argmax(prediction, axis=1)
    print(predicted_class)

    if predicted_class==8:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Manzana</div>", unsafe_allow_html=True)
    if predicted_class==0:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Banano</div>", unsafe_allow_html=True)
    if predicted_class==1:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cuchara</div>", unsafe_allow_html=True)
    if predicted_class==2:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cucharon</div>", unsafe_allow_html=True)
    if predicted_class==3:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cuchillo</div>", unsafe_allow_html=True)
    if predicted_class==4:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Encendedor</div>", unsafe_allow_html=True)
    if predicted_class==5:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Espatula</div>", unsafe_allow_html=True)
    if predicted_class==6:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Filtro</div>", unsafe_allow_html=True)
    if predicted_class==7:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Huevo</div>", unsafe_allow_html=True)
    if predicted_class==9:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Tijeras</div>", unsafe_allow_html=True)
picture = st.camera_input("Por favor, tome una foto para utilizar el modelo.")
if picture:
    st.image(picture)
    print("existe picture")
    with open ('input.jpg','wb') as file:
          file.write(picture.getbuffer())
          print("se guardo foto")
    img = image.load_img('input.jpg', target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class=np.argmax(prediction, axis=1)
    print(predicted_class)
    
    if predicted_class==8:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Manzana</div>", unsafe_allow_html=True)
    if predicted_class==0:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Banano</div>", unsafe_allow_html=True)
    if predicted_class==1:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cuchara</div>", unsafe_allow_html=True)
    if predicted_class==2:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cucharon</div>", unsafe_allow_html=True)
    if predicted_class==3:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Cuchillo</div>", unsafe_allow_html=True)
    if predicted_class==4:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Encendedor</div>", unsafe_allow_html=True)
    if predicted_class==5:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Espatula</div>", unsafe_allow_html=True)
    if predicted_class==6:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Filtro</div>", unsafe_allow_html=True)
    if predicted_class==7:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Huevo</div>", unsafe_allow_html=True)
    if predicted_class==9:
         st.markdown("<div style='font-size: 56px; color: black; text-align: center'>Tijeras</div>", unsafe_allow_html=True)

    #1: BANANO, #2: CUCHARA, #3: CUCHARON, #4 CUCHILLO, #5 ENCENDEDOR, #6 ESPATULA
    #7 FILTRO #8 HUEVO, #9 MANZANA #10 TIJERAS
    
    
    
    
