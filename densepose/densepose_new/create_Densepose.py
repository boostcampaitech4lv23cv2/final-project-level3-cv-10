import os
import pickle
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import Resize


# 먼저 DensePose위치 설정해주어야 함
import sys
sys.path.insert(0, '/opt/ml/Final_Project/densepose/densepose_new/DensePose')
from DensePose.apply_net import apply_Dump, create_argument_parser, DumpAction
from DensePose.color_map import PARULA_COLOR_MAP

# ** 사용 명령어 ** 
# python create_Densepose.py \
# dump \
# --cfg '/opt/ml/backend/densepose_new/DensePose/configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml' \
# --model 'https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl' \
# --input '/opt/ml/backend/densepose/input/image' \
# --output '/opt/ml/backend/densepose/input/densepose'

def get_args():
    parser = create_argument_parser()
    args = parser.parse_args()
    return args

def main():
    # 주의: input img_size와 output_img_size는 (1024, 768)로 맞출 것
    args = get_args()
    
    output_path = args.output
    
    input_img_size = (1024, 768)
    output_img_size = (1024, 768)
    
    pkl_data = apply_Dump(args)
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    for i in tqdm(range(len(pkl_data))):
        # 데이터 불러오기 (tensor)
        data_name = pkl_data[i]['file_name']
        data_score = pkl_data[i]['scores']
        data_bbox_point = pkl_data[i]['pred_boxes_XYXY']
        data_label = pkl_data[i]['pred_densepose'][0].labels
        
        # 데이터 형변환
        data_score = data_score.cpu().numpy().tolist()
        data_label = data_label.cpu().numpy().tolist()
        data_bbox_point = data_bbox_point.cpu().numpy().astype(int).tolist()

        # 이미지로 입력
        img = np.zeros(input_img_size, dtype=np.int8).tolist()
        min_j, min_i, max_j, max_i = data_bbox_point[0]
        
        for i in range(len(img)):
            for j in range(len(img[0])):
                if min_i < i < max_i and min_j < j < max_j and data_label[i-min_i-1][j-min_j-1]:
                    img[i][j] = PARULA_COLOR_MAP[data_label[i-min_i-1][j-min_j-1] * 10]
                else:
                    img[i][j] = (0, 0, 0)

        # pil로 변환 후 저장
        img = np.array(img) * 255
        img = img.astype(np.uint8)
        pil_img = Image.fromarray(img)
        pil_img = Resize(output_img_size)(pil_img)
        pil_img.save(output_path)

if __name__ == "__main__":
    main()

'''
Some body parts are split into 2 patches: 
1, 2 = Torso,
3 = Right Hand,
4 = Left Hand,
5 = Left Foot,
6 = Right Foot,
7, 9 = Upper Leg Right,
8, 10 = Upper Leg Left,
11, 13 = Lower Leg Right,
12, 14 = Lower Leg Left,
15, 17 = Upper Arm Left,
16, 18 = Upper Arm Right,
19, 21 = Lower Arm Left,
20, 22 = Lower Arm Right,
23, 24 = Head
'''
# apply_Dump사용법
# 1. DensePose 폴더를 나와 같이 준비 (Customize된 것으로)
# 2. sys import 후 Densepose경로 설정
# 3. apply_net의 apply_Dump와 create_argument_parser import
# 4. 사용할 densepose model의 cfg와 model_ckpt 경로 확인
# 5. get_args의 argument들 확인 후 apply_Dump에 넣어줌 (return값으로 폴더 안의 이미지들에 대한 list({결과}, {결과}, ...)값을 얻을 수 있음)
'''
import sys
sys.path.insert(0, '/opt/ml/DensePose')
from DensePose.apply_net import apply_Dump

input_img = '/opt/ml/CustomDataset/refocus/test/image'
dct = apply_Dump(input_img)
'''