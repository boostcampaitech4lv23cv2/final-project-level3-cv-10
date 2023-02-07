from fastapi.responses import FileResponse, Response, StreamingResponse

from fastapi import FastAPI, File, Form,UploadFile
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image 
import io 
import cv2, os, glob 
import numpy as np 
from uuid import UUID, uuid4
import subprocess, shutil

app = FastAPI()

@app.post('/human-parse/')
async def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    id_2_str = str(uuid4())    
    model_folder = os.path.join("/opt/ml/Final_Project/data/model_image", id_2_str)
    output_folder = os.path.join("/opt/ml/Final_Project/data/humanparse_output", id_2_str)
    os.makedirs(model_folder)
    os.makedirs(output_folder)
    
    model_path = os.path.join(model_folder, "model.jpg")
    output_path = os.path.join(output_folder, "model.png")
    
    img.save(model_path)
    print("human parse start")
    output = subprocess.getoutput(f"python create_hp/human_parse/inf_png.py \
        -i {model_folder} -o {output_folder}")
    
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".png", nparr_data)[1]
    data = data.tobytes()
    
    shutil.rmtree(model_folder, ignore_errors=True)
    shutil.rmtree(output_folder, ignore_errors=True)
    
    return Response(data, media_type="image/png")