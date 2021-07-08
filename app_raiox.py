import streamlit as st
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
from keras.preprocessing.image import img_to_array

uploader_file = st.file_uploader("Carregar Image Raio X", type=['png','jpg','jpeg'], accept_multiple_files=False)
if uploader_file is not None:
    input_img = Image.open(uploader_file)
    st.image(input_img)
    img = img_to_array(input_img)

    if img.shape[0] < 256 or img.shape[1] < 256:
        st.write('Favor entrar com uma imagem com resolução maior.')
        st.write('A imagem dever ter resolução superior a 256 x 256')
    else:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (256, 256))
            img = img / 255.0   
            img = np.expand_dims(img, axis=0)
        else:
            img = cv2.resize(img, (256, 256))
            img = img / 255.0   
            img = img.reshape(-1, 256, 256, 3)
        
        labels_names = {0: 'Covid-19', 1: 'Normal', 2: 'Pneunomia viral', 3: 'Pneunomia bacterial'}

        model = load_model('./weights.hdf5')
        predict = model(img)
        predict_class = np.argmax(predict)
        predict_proba = predict.numpy()
        predict_proba = pd.DataFrame(predict_proba, columns= ['Covid-19', 'Normal', 'Pneunomia viral', 'Pneunomia bacterial'])
        st.table(predict_proba*100)
    


    



