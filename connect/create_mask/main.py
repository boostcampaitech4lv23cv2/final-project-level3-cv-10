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

from create_mask.refocus_Images import get_refocus_image_mask

app = FastAPI()

@app.post("/refocus/")
async def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    refocus_img = get_refocus_image_mask(img)[0]
    nparr_data = np.asarray(refocus_img)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    return Response(data, media_type="image/jpg")

@app.post("/cloth-mask/")
async def create_file(
    image: bytes = File(...),
):
    img = Image.open(io.BytesIO(image))
    cloth_mask = get_refocus_image_mask(img)[1]
    nparr_data = np.asarray(cloth_mask)
    data = cv2.imencode(".jpeg", nparr_data)[1]
    data = data.tobytes()
    return Response(data, media_type="image/jpg")
