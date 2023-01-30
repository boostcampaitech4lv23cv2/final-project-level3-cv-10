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

@app.post('/human-parse/')
def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    os.makedirs("./data/model_image", exist_ok=True)
    os.makedirs("./data/humanparse_output", exist_ok=True)
    
    id_2_str = str(uuid4())
    model_path = f"/opt/ml/Final_Project/connect/data/model_image/{id_2_str}.jpg"
    output_path = f"/opt/ml/Final_Project/connect/data/humanparse_output/{id_2_str}.jpg"
    
    model_folder = "/opt/ml/Final_Project/connect/data/model_image"
    output_folder = "/opt/ml/Final_Project/connect/data/humanparse_output"
    
    img.save(model_path)
    
    output = subprocess.getoutput(f"python create_hp/human_parse/inf_png.py \
        -i {model_folder} -o {output_folder}")
    
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    
    os.remove(model_path)
    os.remove(output_path)
    
    return Response(data, media_type="image/jpg")