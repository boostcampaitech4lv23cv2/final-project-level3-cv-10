from fastapi.responses import FileResponse, Response, StreamingResponse

from fastapi import FastAPI, File, Form
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image 
import io 
import cv2, os, glob 
import numpy as np 
from uuid import UUID, uuid4
import subprocess

app = FastAPI()

@app.post('/densepose/')
async def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    os.makedirs("./data/model_image", exist_ok=True)
    os.makedirs("./data/densepose_output", exist_ok=True)
    
    id_2_str = str(uuid4())
    model_path = f"/opt/ml/Final_Project/connect/data/model_image/{id_2_str}.jpg"
    output_path = f"/opt/ml/Final_Project/connect/data/densepose_output/{id_2_str}.jpg"
    
    img.save(model_path)
    # img.save(output_path)
    
    output = subprocess.getoutput(f"python densepose/densepose_new/create_Densepose.py dump \
        --cfg '/opt/ml/Final_Project/connect/densepose/densepose_new/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml' \
        --model 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl' \
        --input {model_path} --output {output_path}")
    
    # print(output)
    
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    
    # os.remove(model_path)
    # os.remove(output_path)
    
    return Response(data, media_type="image/jpg")