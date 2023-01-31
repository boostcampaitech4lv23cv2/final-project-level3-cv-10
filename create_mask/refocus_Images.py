from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torchvision.transforms import Resize
from torchvision.transforms import functional as F
import torch
from carvekit.api.high import HiInterface
from argparse import ArgumentParser
# 대상 이미지의 테두리를 기준으로 padding만큼 자른 이미지를 만들어주는 함수

# **사용 명령어**
# python refocus_Images.py \
# --input_dir /opt/ml/CustomDataset/basic/train/image \
# --output_dir /opt/ml/test \
# --target_info model

# pad설정과 resize설정은 필요할 경우 코드에서 고침

def convert_to_mask(img:np):
    mask = img
    base = [130, 130, 130]
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j].tolist() == base:
                mask[i][j][0] = 0
                mask[i][j][1] = 0
                mask[i][j][2] = 0
            else:
                mask[i][j][0] = 255
                mask[i][j][1] = 255
                mask[i][j][2] = 255
    return mask

def min_max_img(img:np):
    min_h, min_w, max_h, max_w = 0, 0, 0, 0
    temp = img.transpose(2, 0, 1)
    for i in range(len(temp[0])):
        if any(temp[0][i]):
            min_h = i
            break
    for i in range(len(temp[0])-1, 0, -1):
        if any(temp[0][i]):
            max_h = i
            break
    temp = img.transpose(2, 1, 0)
    for j in range(len(temp[0])):
        if any(temp[0][j]):
            min_w = j
            break
    for j in range(len(temp[0])-1, 0, -1):
        if any(temp[0][j]):
            max_w = j
            break
    return min_h, min_w, max_h, max_w

def resize_img(img:np, mask:np, size:tuple, pad:tuple):
    img_h, img_w = len(img), len(img[0])
    min_h, min_w, max_h, max_w = min_max_img(mask)
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = pad
    
    # img resize
    img = Image.fromarray(img.astype(np.uint8))
    img = F.crop(img, 
                max(0, min_h-pad_h_top),
                max(0, min_w-pad_w_left),
                min(max_h-min_h+pad_h_bottom+pad_h_top, img_h-min_h+pad_h_top-1),
                min(max_w-min_w+pad_w_right+pad_w_left, img_w-min_w+pad_w_left-1))
    img = Resize(size)(img)
    return img

def resize_img_cloth(img:np, mask:np, size:tuple, pad:tuple):
    img_h, img_w = len(img), len(img[0])
    min_h, min_w, max_h, max_w = min_max_img(mask)
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = pad
    
    # img resize
    img = Image.fromarray(img.astype(np.uint8))
    img = F.crop(img, 
                max(0, min_h-pad_h_top),
                max(0, min_w-pad_w_left),
                min(max_h-min_h+pad_h_bottom+pad_h_top, img_h-min_h+pad_h_top-1),
                min(max_w-min_w+pad_w_right+pad_w_left, img_w-min_w+pad_w_left-1))
    img = Resize(size)(img)

    # mask resize
    mask = Image.fromarray(mask.astype(np.uint8))
    mask = F.crop(mask, 
                 max(0, min_h-pad_h_top),
                 max(0, min_w-pad_w_left),
                 min(max_h-min_h+pad_h_bottom+pad_h_top, img_h-min_h+pad_h_top-1),
                 min(max_w-min_w+pad_w_right+pad_w_left, img_w-min_w+pad_w_left-1))
    mask = Resize(size)(mask)

    return img, mask

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_info', type=str, required=True, choices=['model', 'cloth'])
    args = parser.parse_args()
    return args

def get_refocus_image_mask(input_img):
    pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = 50, 50, 50, 50
    resize_size = (1024, 768)
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
        batch_size_seg=5,
        batch_size_matting=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
        matting_mask_size=2048,
        trimap_prob_threshold=231,
        trimap_dilation=30,
        trimap_erosion_iters=5,
        fp16=False)
    
    target_img = input_img.convert('RGB')
    # type(images_without_background) == <class 'PIL.Image.Image'>
    img_without_background = interface([target_img])[0].convert('RGB')

    target_img = np.array(target_img)
    img_without_background = np.array(img_without_background)
    img_mask = convert_to_mask(img_without_background)
    
    refocus_img, refocus_mask = resize_img_cloth(target_img,
                                img_mask,
                                size=resize_size,
                                pad=(pad_h_top, pad_h_bottom, pad_w_left, pad_w_right))
            
    refocus_mask = refocus_mask.convert('L')
    return (refocus_img, refocus_mask)

def main():
    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.target_info == "cloth":
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = 50, 50, 50, 50
    elif args.target_info == "model":
        pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = 10, -200, 50, 50
    else:
        print("wrong target. please check target_info")
        exit()
        
    resize_size = (1024, 768)
    # Check doc strings for more information
    interface = HiInterface(object_type="hairs-like",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    
    for file in tqdm(os.listdir(input_dir)):
        target_img_path = os.path.join(input_dir, file)
        target_img = Image.open(target_img_path).convert('RGB')

        # type(images_without_background) == <class 'PIL.Image.Image'>
        img_without_background = interface([target_img])[0].convert('RGB')

        target_img = np.array(target_img)
        img_without_background = np.array(img_without_background)
        img_mask = convert_to_mask(img_without_background)
        
        refocus_img = resize_img(target_img,
                                img_mask,
                                size=resize_size,
                                pad=(pad_h_top, pad_h_bottom, pad_w_left, pad_w_right))
        refocus_img.save(os.path.join(output_dir, file))

if __name__ == "__main__":
    main()