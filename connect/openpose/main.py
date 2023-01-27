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
import uvicorn, json

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


app = FastAPI()

@app.post('/openpose-img/')
def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    os.makedirs("./data/model_image", exist_ok=True)
    os.makedirs("./data/openpose_output", exist_ok=True)
    
    id_2_str = str(uuid4())
    model_path = f"/opt/ml/Final_Project/connect/data/model_image/{id_2_str}.jpg"
    output_path = f"/opt/ml/Final_Project/connect/data/openpose_output/image/{id_2_str}_rendered.png"
    json_path = f"/opt/ml/Final_Project/connect/data/openpose_output/json/{id_2_str}_keypoints.json"
    
    model_folder = "/opt/ml/Final_Project/connect/data/model_image"
    output_folder = "/opt/ml/Final_Project/connect/data/openpose_output/image"
    json_folder = "/opt/ml/Final_Project/connect/data/openpose_output/json"
    img.save(model_path)
    
    output = subprocess.getoutput(f"./build/examples/openpose/openpose.bin \
                                    --image_dir {model_folder} \
                                    --write_images {output_folder} \
                                    --display 0 --write_json {json_folder} \
                                    --disable_blending --hand")
    
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    openpose_json = json.load(open(json_path))
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    
    os.remove(model_path)
    os.remove(output_path)
    os.remove(json_path)
    
    return Response(data, media_type="image/jpg")

@app.post('/openpose-json/')
def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    os.makedirs("./data/model_image", exist_ok=True)
    os.makedirs("./data/openpose_output", exist_ok=True)
    
    id_2_str = str(uuid4())
    model_path = f"/opt/ml/Final_Project/connect/data/model_image/{id_2_str}.jpg"
    output_path = f"/opt/ml/Final_Project/connect/data/openpose_output/image/{id_2_str}_rendered.png"
    json_path = f"/opt/ml/Final_Project/connect/data/openpose_output/json/{id_2_str}_keypoints.json"
    
    model_folder = "/opt/ml/Final_Project/connect/data/model_image"
    output_folder = "/opt/ml/Final_Project/connect/data/openpose_output/image"
    json_folder = "/opt/ml/Final_Project/connect/data/openpose_output/json"
    img.save(model_path)
    
    output = subprocess.getoutput(f"./build/examples/openpose/openpose.bin \
                                    --image_dir {model_folder} \
                                    --write_images {output_folder} \
                                    --display 0 --write_json {json_folder} \
                                    --disable_blending --hand")
    
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    openpose_json = json.load(open(json_path))
    # json_encoded = jsonable_encoder(openpose_json)
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    
    os.remove(model_path)
    os.remove(output_path)
    os.remove(json_path)
    
    return JSONResponse(content=openpose_json)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=30006, reload=True)