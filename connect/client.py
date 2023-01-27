import requests
import cv2
import numpy as np 
import json
from PIL import Image
import os, argparse

def original2refocus(image_name):
    # original cloth to refocus cloth
    image_path = os.path.join('cloth', image_name)
    image = cv2.imread(image_path)

    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    refocus_res = requests.post("http://49.50.163.219:30003/refocus/",
                        files=files)

    refocus_data = refocus_res.content

    refocus_nparr = np.frombuffer(refocus_data, np.uint8)
    refocus_img = cv2.imdecode(refocus_nparr, cv2.IMREAD_COLOR)
    # save image to file
    refocus_img = Image.fromarray(refocus_img)

    refocus_path = os.path.join('refocus', image_name)
    refocus_img.save(refocus_path)

# refocus cloth to mask
def original2mask(image_name):
    refocus_path = os.path.join('cloth', image_name)
    refocus_image = cv2.imread(refocus_path)

    refocus_data = cv2.imencode(".jpg", refocus_image)[1]

    files = {
        'image': ('a.jpg', refocus_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }  
    mask_res = requests.post("http://49.50.163.219:30003/cloth-mask/",
                        files=files)

    mask_data = mask_res.content

    mask_nparr = np.frombuffer(mask_data, np.uint8)
    mask_img = cv2.imdecode(mask_nparr, cv2.IMREAD_COLOR)
    
    # save image to file
    mask_img_file = Image.fromarray(mask_img)

    mask_path = os.path.join('mask', image_name)
    mask_img_file.save(mask_path)
    
# model image to densepose 
def model2densepose(model_image_name):
    image_path = os.path.join('model', model_image_name)
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

    dp_path = os.path.join('densepose', model_image_name)
    dp_img.save(dp_path)
    
def model2humanparse(model_image_name):
    image_path = os.path.join('model', model_image_name)
    image = cv2.imread(image_path)

    img_data = cv2.imencode(".jpg", image)[1]

    files = {
        'image': ('a.jpg', img_data.tobytes(), 'image/jpg', {'Expires': '0'})
    }   

    dp_res = requests.post("http://49.50.163.219:30005/human-parse/",
                        files=files)

    dp_data = dp_res.content
    
    dp_nparr = np.frombuffer(dp_data, np.uint8)
    dp_nparr = cv2.imdecode(dp_nparr, cv2.IMREAD_COLOR)

    # save image to file
    dp_img = Image.fromarray(dp_nparr)

    dp_path = os.path.join('humanparse', model_image_name)
    dp_img.save(dp_path)
    
def model2openpose(model_image_name):
    image_path = os.path.join('model', model_image_name)
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

    dp_path = os.path.join('openpose', model_image_name)
    dp_img.save(dp_path)
    
    json_path = os.path.join('openpose-json', model_image_name.replace('.jpg', '.json'))
    with open(json_path, "w") as outfile:
        json.dump(decoded_data, outfile)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--image_name', type=str, default='1008026.jpg', help='image name')
    parser.add_argument('-m','--model_name', type=str, default='00006_00.jpg', help='model image name')
    args = parser.parse_args()
    
    image_name = args.image_name
    model_image_name = args.model_name
    
    # original2refocus(image_name)
    # original2mask(image_name)
    # model2densepose(model_image_name)
    # model2humanparse(model_image_name)
    model2openpose(model_image_name)