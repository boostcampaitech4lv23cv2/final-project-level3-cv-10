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
import shutil

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


app = FastAPI()

@app.post('/openpose-img/')
async def create_file(
    image: bytes = File(...),
):
    print("test-img")
    img = Image.open(io.BytesIO(image))
    if not os.path.exists("/opt/ml/Final_Project/data/model_image"):
        os.makedirs("/opt/ml/Final_Project/data/model_image")
    if not os.path.exists("/opt/ml/Final_Project/data/openpose_output"):
        os.makedirs("/opt/ml/Final_Project/data/openpose_output")
    id_2_str = str(uuid4())
    
    model_folder = f"/opt/ml/Final_Project/data/model_image/{id_2_str}"
    output_folder = f"/opt/ml/Final_Project/data/openpose_output/{id_2_str}"
    
    model_path = os.path.join(model_folder, "model.jpg")
    output_path = os.path.join(output_folder, "model_rendered.png")
    os.makedirs(model_folder)
    os.makedirs(output_folder)
    
    img.save(model_path)
    print(f"image saved to {model_path}")
    output = subprocess.getoutput(f"./build/examples/openpose/openpose.bin \
                                    --image_dir {model_folder} \
                                    --write_images {output_folder} \
                                    --display 0 \
                                    --disable_blending --hand")
    print(output)
    # get image from ./data/densepose_output
    densePose_img = Image.open(output_path)
    
    nparr_data = np.asarray(densePose_img)
    data = cv2.imencode(".png", nparr_data)[1]
    data = data.tobytes()
    
    shutil.rmtree(model_folder, ignore_errors=True)
    shutil.rmtree(output_folder, ignore_errors=True)
    
    return Response(data, media_type="image/png")

@app.post('/openpose-json/')
async def create_file(
    image: bytes = File(...),
):
    print("test-json")
    img = Image.open(io.BytesIO(image))
    if not os.path.exists("/opt/ml/Final_Project/data/model_image"):
        os.makedirs("/opt/ml/Final_Project/data/model_image")
    if not os.path.exists("/opt/ml/Final_Project/data/openpose_output"):
        os.makedirs("/opt/ml/Final_Project/data/openpose_output")
    
    id_2_str = str(uuid4())
    
    model_folder = f"/opt/ml/Final_Project/data/model_image/{id_2_str}"
    output_folder = f"/opt/ml/Final_Project/data/openpose_output/{id_2_str}"
    json_folder = os.path.join(output_folder, "json")
    img_folder = os.path.join(output_folder, "image")
    
    model_path = os.path.join(model_folder, "model.jpg")
    json_path = os.path.join(json_folder, "model_keypoints.json")
    
    os.makedirs(model_folder)
    os.makedirs(json_folder)
    os.makedirs(img_folder)
    
    img.save(model_path)
    print(f"image saved to {model_path}")
    output = subprocess.getoutput(f"./build/examples/openpose/openpose.bin \
                                    --image_dir {model_folder} \
                                    --write_images {img_folder} \
                                    --display 0 --write_json {json_folder} \
                                    --disable_blending --hand")
    print(output)
    # get json from json_path
    openpose_json = json.load(open(json_path))

    shutil.rmtree(model_folder, ignore_errors=True)
    shutil.rmtree(output_folder, ignore_errors=True)
    
    return JSONResponse(content=openpose_json)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=30006, reload=True)