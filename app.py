import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
import copy 
 
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('old_model-009.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Image Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):
    
        size = (180,180)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        pred_class=decode_predictions(prediction,top=1)
        
        return pred_class
if file is None:
    st.text("Please upload an image file")
else:
    img = Image.open(file)
    img =np.array(img)
    img1=copy.deepcopy(img)
    #RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
    blur=cv2.medianBlur(gray,25)
    canyedge=cv2.Canny(blur,10,80)

    kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    close=cv2.morphologyEx(canyedge,cv2.MORPH_CLOSE,kernel,iterations=1)
    dilate=cv2.dilate(close,kernel,iterations=2)

    cnts=cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    withcontour=cv2.drawContours(img1,cnts,0,(255,0,0),5)
    total_area=0
    for i in range(len(cnts)):
        area=cv2.contourArea(cnts[0])
        total_area=total_area+area



    st.image(withcontour, use_column_width=True)
    img1=cv2.resize(img1, (228,228), cv2.INTER_AREA)
    images = np.expand_dims(img1, axis=0)

    prediction = model.predict(images)



    image_class = str(prediction)
    if prediction <= 0.5:
            st.write("The image is classified as good onion")
    else:
            st.write("The image is classified as bad onion")
    if total_area>50000:
        st.write("large onion")
    else:
        st.write("small onion")


