from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field

from typing import List, Union, Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
import secrets
import argparse
import torch
import torch.nn as nn
import sys
sys.path.append('/opt/ml/input/VTO')
from HR_VITON.networks import ConditionGenerator, load_checkpoint, make_grid
from HR_VITON.network_generator import SPADEGenerator
from HR_VITON.cp_dataset_test import CPDatasetTest, CPDataLoader
from HR_VITON.get_parse_agnostic import get_im_parse_agnostic
from collections import OrderedDict
import os
from PIL import Image
import io
from fastapi.responses import FileResponse
from predict import get_prediction
# from .model import MyEfficientNet, get_model, get_config, predict_from_image_byte
import time
import asyncio
from urllib.request import urlopen
import json
import aiohttp
import subprocess
# yj
import cv2 
import requests
import numpy as np 

app = FastAPI()

@app.get("/")
def hello_world():
    return {"hello": "test"}

class Product(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    price: float

class Order(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    products: List[Product] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def bill(self):
        return sum([product.price for product in self.products])

    def add_product(self, product: Product):
        if product.id in [existing_product.id for existing_product in self.products]:
            return self

        self.products.append(product)
        self.updated_at = datetime.now()
        return self

orders = []

@app.get("/order", description="주문 리스트를 가져옵니다")
async def get_orders() -> List[Order]:
    return orders

@app.get("/order/{order_id}", description="Order 정보를 가져옵니다")
async def get_order(order_id: UUID) -> Union[Order, dict]:
    order = get_order_by_id(order_id=order_id)
    if not order:
        return {"message": "주문 정보를 찾을 수 없습니다"}
    return order

def get_order_by_id(order_id: UUID) -> Optional[Order]:
    return next((order for order in orders if order.id == order_id), None)

class InferenceImageProduct(Product):
    name: str = "inference_image_product"
    price: float = 100.0
    result: Optional[List]

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default='0')
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')
    # Cuda availability
    parser.add_argument('--cuda',default=True, help='cuda or cpu')

    parser.add_argument('--test_name', type=str, default='test', help='test name')
    parser.add_argument("--dataroot", default="/opt/ml/input/final/HR_VITON/data")
    parser.add_argument("--datamode", default="test")
    parser.add_argument("--data_list", default="test_pairs_v1.txt")
    parser.add_argument("--output_dir", type=str, default="./Output")
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=768)
    parser.add_argument("--fine_height", type=int, default=1024)

    parser.add_argument('--tensorboard_dir', type=str, default='./data/tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='/opt/ml/input/final/HR_VITON/eval_models/weights/v0.1/mtviton.pth', help='tocg checkpoint')
    parser.add_argument('--gen_checkpoint', type=str, default='/opt/ml/input/final/HR_VITON/eval_models/weights/v0.1/gen.pth', help='G checkpoint')

    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    parser.add_argument('--gen_semantic_nc', type=int, default=7, help='# of input label classes without unknown class')
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="relu")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        
    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")

    # generator
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance', help='instance normalization or batch normalization')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
    parser.add_argument('--num_upsampling_layers', choices=('normal', 'more', 'most'), default='most', # normal: 256, more: 512
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

    parser.add_argument('--md', default='03964_00.jpg')
    parser.add_argument('--cloth', default='13172_00.jpg')
    opt = parser.parse_args()
    return opt
def load_checkpoint_G(model, checkpoint_path,opt):
    if not os.path.exists(checkpoint_path):
        print("Invalid path!")
        return
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict.items()])
    new_state_dict._metadata = OrderedDict([(k.replace('ace', 'alias').replace('.Spade', ''), v) for (k, v) in state_dict._metadata.items()])
    model.load_state_dict(new_state_dict, strict=True)
    if opt.cuda :
        model.cuda()

# 클라이언트
# TODO : get - 서버에 저장되어있는 inference된 이미지 불러오기
@app.get('/images/{id}')
def get_image(id:str):
    # base
    opt = get_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids

    # opt.md = f"{id}_md.jpg"
    # opt.cloth = f"{id}_cloth.jpg"
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)

    # tocg
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = 16  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    
    # generator
    opt.semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    generator.print_network()

    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint,opt)
    load_checkpoint_G(generator, opt.gen_checkpoint,opt)

    get_prediction(opt ,test_loader, tocg, generator, id)

    return FileResponse(''.join(["./Output/",f"{id}.png"]))


# 클라이언트에게 image를 요청 (cloth, md)
@app.post('/upload_images', description="image 요청합니다")
async def upload_md(md_file: UploadFile = File(...),
                    cloth_file: UploadFile = File(...)):

    UPLOAD_DIR = "./image/"  # 이미지를 저장할 서버 경로
    id = str(uuid4())
    
    # md_content = await md_file.read()

    contents = await md_file.read()
    pil_image = Image.open(io.BytesIO(contents))

    # resize image to expected input shape
    pil_image = pil_image.resize((768, 1024))
    md_filename = f"{id}_md.jpg"  # uuid로 유니크한 파일명으로 변경
    pil_image.save(f"{UPLOAD_DIR}{id}_md.jpg")
    # with open(os.path.join(UPLOAD_DIR, md_filename), "wb") as fp:
    #     fp.write(pil_image)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    cloth_content = await cloth_file.read()
    cloth_filename = f"{id}_cloth.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, cloth_filename), "wb") as fp:
        fp.write(cloth_content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    start_time = time.time() 
    print("Start")
    # await main(cloth_filename, md_filename)
    # async with aiohttp.ClientSession() as sess:           
    await asyncio.gather(#original2refocus(cloth_filename),  
                        #  densepose(md_filename),
                        #  humanparse(md_filename),
                        openpose(md_filename),
                        # original2mask(cloth_filename),
                        )

    output = subprocess.getoutput("./HR_VITON/get_parse_agnostic.py")

    print(f"End time {time.time() - start_time}")
    
    return {"id": id,
            "md_filename": md_filename,
            "cloth_filename": cloth_filename}

async def main(cloth_filename, md_filename) :
    await asyncio.gather(original2refocus(cloth_filename),
                         densepose(md_filename),
                         humanparse(md_filename),
                         openpose(md_filename),
                         original2mask(cloth_filename),
                        )

    # await original2refocus(cloth_filename)
    # await original2mask(cloth_filename)

# [cloth_mask] Server
async def original2refocus(cloth_filename):
    print("Start : original2refocus")
    # await asyncio.sleep(5)

    image_path = f"./image/{cloth_filename}"
    image = cv2.imread(image_path)
    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    # data = open(image_path, 'rb')

    refocus_res = requests.post("http://49.50.163.219:30003/refocus/",
                        files=files)

    with open('data/test/cloth/cloth.jpg', 'wb') as f:
        f.write(refocus_res.content)
    print("End : original2refocus")

async def original2mask(cloth_filename):
    print("Start : original2mask")

    refocus_path = f"./image/{cloth_filename}"
    refocus_image = cv2.imread(refocus_path)
    refocus_data = cv2.imencode(".jpg", refocus_image)[1]

    files = {
        'image': ('a.jpg', refocus_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }
    mask_res  = requests.post("http://49.50.163.219:30003/cloth-mask/",
                    files=files)

    with open('data/test/cloth_mask/cloth_mask.jpg', 'wb') as f:
        f.write(mask_res.content)
    print("End : original2mask")

# [densepose] Server
async def densepose(md_filename):
    print("Start : densepose")
    image_path = f"./image/{md_filename}"
    image = cv2.imread(image_path)
    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    dp_res = requests.post("http://49.50.163.219:30004/densepose/",
                        files=files)

    dp_data = dp_res.content
    
    dp_nparr = np.frombuffer(dp_data, np.uint8)
    dp_nparr = cv2.imdecode(dp_nparr, cv2.IMREAD_COLOR)

    # save image to file
    dp_img = Image.fromarray(dp_nparr)

    dp_path = os.path.join('data/test/image-densepose', md_filename)
    dp_img.save(dp_path)
    
    print("End : densepose")

# [humanparse] Server
async def humanparse(md_filename):
    print("Start : humanparse")
    image_path = f"./image/{md_filename}"
    image = cv2.imread(image_path)
    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    hp_res = requests.post("http://49.50.163.219:30005/human-parse/",
                        files=files)

    with open('data/test/image-parse-v3/humanparse.jpg', 'wb') as f:
        f.write(hp_res.content)
    print("End : humanparse")

# [Openpose]
async def openpose(md_filename):
    print("Start : openpose")
    
    image_path = f"./image/{md_filename}"
    image = cv2.imread(image_path)
    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    dp_res = requests.post("http://49.50.163.219:30006/openpose-img/",
                        files=files)
    json_res = requests.post("http://49.50.163.219:30006/openpose-json/",
                        files=files)
    
    dp_data = dp_res.content
    json_data = json_res.content.decode("utf-8")
    
    decoded_data = json.loads(json_data)
    
    dp_nparr = np.frombuffer(dp_data, np.uint8)
    dp_nparr = cv2.imdecode(dp_nparr, cv2.IMREAD_COLOR)

    # save image to file
    dp_img = Image.fromarray(dp_nparr)
    dp_path = os.path.join('data/test/openpose_img', md_filename)
    dp_img.save(dp_path)
    
    json_path = os.path.join('data/test/openpose_json', md_filename.replace('.jpg', '.json'))
    with open(json_path, "w") as outfile:
        json.dump(decoded_data, outfile)
    
    print("End : openpose")

















# 다른 Server
# TODO : get - 서버에 저장되어있는 원본 이미지 불러오기
@app.get('/upload_clothmask', description="ClothMask")
async def upload_clothmask() :
                        #    cloth_file : UploadFile = File(...)) :

    CLOTH_DIR = "./data/test/cloth/85c43863-4efa-4f03-a180-51bc06320ac6_cloth.jpg"
    cloth = open(CLOTH_DIR, "rb")
    # CLOTH_MASK_DIR = "./data/test/cloth_mask"  # 이미지를 저장할 서버 경로

    # id = str(uuid4())

    # cloth_content = await cloth_file.read()
    # cloth_filename = f"{id}_cloth.jpg"  # uuid로 유니크한 파일명으로 변경
    # with open(os.path.join(CLOTH_DIR, cloth_filename), "wb") as fp:
    #     fp.write(cloth_content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)

    # cloth_mask_content = await maks_file.read()
    # cloth_mask_filename = f"{id}_cloth_mask.jpg"  # uuid로 유니크한 파일명으로 변경
    # with open(os.path.join(CLOTH_MASK_DIR, cloth_mask_filename), "wb") as fp:
    #     fp.write(cloth_mask_content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
    print("good")
    # with open("image.jpg", "wb") as f:
    #     f.write(await image.read())
    return FileResponse(CLOTH_DIR)

    # return {"id": id,
    #         "image": cloth_filename}

# TODO : post - Inference한 이미지 서버에 저장


# 사이즈 1024,768 model이미지 변경하고 전달 -> image 폴더안에