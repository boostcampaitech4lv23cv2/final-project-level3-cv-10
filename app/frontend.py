import streamlit as st
import io

import time
from PIL import Image
import requests
import torch.nn as nn
from pyparsing import empty
import asyncio
from google.cloud import storage

def main():
    # Normal
    #----------------------------------------------------------------------------------------------------------------------
    st.set_page_config(layout="wide") # SETTING PAGE CONFIG TO WIDE MODE
    empty1,con1,empty2 = st.columns(3)
    empty1,con2,con3,empty2 = st.columns([0.3,0.5,0.5,0.3])
    empty1,con4,empty2 = st.columns([2.5,1,2])
    empty1,con5,empty2 = st.columns([0.3,1,0.3])

    with empty1 :
        empty()
    with empty2 :
        empty()

    with con1 :
        # st.title(":sunglasses:")
        st.markdown("<h1 style='text-align: center; color: pink;'>Virtual Try On</h1>", unsafe_allow_html=True)

    with con2 :
        st.markdown("<h3 style='text-align: center; color: white;'>Choose img</h3>", unsafe_allow_html=True)
        count = 0
        uploaded_md = st.file_uploader("",type=["jpg", "jpeg","png"], key=count)
        if uploaded_md :
            global md_g
            md_g = uploaded_md.name
            md_bytes = uploaded_md.getvalue()
            md = Image.open(io.BytesIO(md_bytes))
            st.image(md, caption='Uploaded Model Image')

    with con3 :
        st.markdown("<h3 style='text-align: center; color: white;'>Choose cloth</h3>", unsafe_allow_html=True)
        count = 1
        uploaded_cloth = st.file_uploader("",type=["jpg", "jpeg","png"], key=count)
        if uploaded_cloth :
            global cloth_g
            cloth_g = uploaded_cloth.name
            cloth_bytes = uploaded_cloth.getvalue()
            cloth = Image.open(io.BytesIO(cloth_bytes))
            st.image(cloth, caption='Uploaded Cloth Image')

    with con4 :
        if st.button('Inference') :
            if uploaded_cloth and uploaded_md :
                with st.spinner('Wait for it...'):
                    with con5 :
                        iter_start_time = time.time() 

                        # bucket_name = 'fm_save'    # 서비스 계정 생성한 bucket 이름 입력
                        # destination_blob_name = 'test.jpg'    # 업로드할 파일을 GCP에 저장할 때의 이름
                        # storage_client = storage.Client()
                        # bucket = storage_client.bucket(bucket_name)
                        # blob = bucket.blob(destination_blob_name)

                        # # file_obj = io.BytesIO(cloth_bytes)
                        # # file_obj.seek(0)

                        # blob.upload_from_filename("/opt/ml/input/VTO/app/Output/243f8ecb-da60-47ea-ab3a-a7ffa7b062e3.png")
                        
                        # st.success('success', icon="✅")
                        files = [
                            ('md_file', (uploaded_md.name, md_bytes, uploaded_md.type)),
                            ('cloth_file', (uploaded_cloth.name, cloth_bytes, uploaded_cloth.type))
                        ]

                        response = requests.post("http://localhost:30001/upload_images",  files=files) # 2.17초
                        id = response.json()['id']
                        # result = requests.get(f"http://localhost:30001/images/{id}") # 9.35초

                        # rimg = Image.open(io.BytesIO(result.content))
                        # st.image(rimg, caption='Result Image')
                        print(id)
                        st.success('success')
                        print(f"Test time {time.time() - iter_start_time}") # 9.39초
            else : 
                st.write("no image")
    
# front
main()