import streamlit as st 
import cv2
from detect import run, parse_opt
import numpy as np 
import sys 
from pathlib import Path 
import tempfile
import os 

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmpfile:
            tfile_name = tmpfile.name 
            tmpfile.write(uploaded_file.getvalue())
        return tfile_name 
    except Exception as e:
        return None

def run_detection(image):
    args ={
        'weights': './pretrained-weights/House/exp/weights/best.pt',
        'source': image,
        'imgsz': (1280, 1280),
        'project': './output',
        'name': 'HouseStreamlitTest',
        'save_txt': True
    }
    results = run(**args)
    save_dir = Path(args['project']) / args['name']
    result_image = list(save_dir.glob('*.jpg'))
    if not result_image:
        return None 
    result_image_path = str(result_image[0])
    return result_image_path


st.title('HTP 심리테스트')
uploaded_file = st.file_uploader('이미지를 업로드하세요.', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = save_uploaded_file(uploaded_file)
    if image: 
        if st.button('실행'):
            result_image_path = run_detection(image)
            cols = st.columns(2)
            with cols[0]:
                st.image(image, caption='Uploaded image', use_column_width=True)
            if result_image_path:
                result_image = cv2.imread(result_image_path)
                with cols[1]:
                    st.image(result_image, caption='Detected Image', use_column_width=True)
                    st.write('Detection completed.')
            else:
                st.error('No detected images were found.')
            os.unlink(image)
    else:
        st.error('Failed to save the uploaded image')
