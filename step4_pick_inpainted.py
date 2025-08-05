import time
start1=time.time()
import argparse
import torch
import numpy as np
import os
from PIL import Image
import cv2
import sys
from openai import OpenAI
import json
import re
import base64
import requests
import jinja2
import shutil
import random

SEED=42

# put your api_key here
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED)

def ask_best_pix2gestalt(image_path_lists, furniture_name):
    template_pix2gestalt = jinja2.Template(
        r"""
    You are an expert  who specializes in evaluating inpainted results of an incomplete input image. If you do a good job, I can offer you $100 tips.
    I have an image of an incomplete {{furniture_name}}, and I used an inpainting method to complete it.
    The results are generated with different seeds, and I need your help with choosing the best result.
    I KNOW YOU CAN EVALUATE IMAGES, DO NOT REPLY THAT YOU CANNOT EVALUATE IMAGES. PLEASE COMPLETE THIS TASK.
    Please evaluate each result image in 3 aspects: (1) Realistic Object: how realistic it looks like {{furniture_name}},
    (2) Complete Appearance: how complete the geometry and appearance is. If an image still holes or large occlusions on the object, it should have a lower score.
    (3) Consistent Texture: how consistant the texture is. If the texture of one region is unrealistically inconsistent with its neighboring nextures, such as a black spot,
    that is probably resulted from a failed inpainting, and such that image should have a lower score. 
    I will provide you the 6 images below. For each, give a score from 1 to 5 to rate the 3 aspects above. The higher the score, the better the quality.
    Answer in the format: image index: (1) Realistic Object: score 1, (2) Complete Appearance: score 2, (3) Consistent Texture: score 3, sum of the 3 scores.
    For each image, start a new line.
    In the end, compare the summed score of the images and, in the last line of your reply, output the index of the image with the highest score in the format below: 
    (If there are multiple indices with the same highest score, randomly output only one of them.)
    ---BEGIN Index---
    Index:
    ---END Index---
    """.strip(),  # noqa: E501
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":template_pix2gestalt.render(furniture_name=furniture_name)
                },
            ],
        }
    ]
    for i in range(len(image_path_lists)):
        idx=int(image_path_lists[i].split('/')[-1].split('_')[0])
        print(idx,image_path_lists[i])
        messages[0]['content'].append({"type": "text", "text": "This is image index {i}".format(i=str(idx))})
        image=encode_image(image_path_lists[i])
        messages[0]['content'].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}",
                        "detail":"low",
                    },
                    
                })
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=500,
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

time1= time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_inpaint', action='store_true', help='if do not need inpaint, this script will simply perform a folder renaming.')
    parser.add_argument('--root_dir', type=str, help='the structure should be root_dir/folder1, folder2, ..., where each folder if of a scene')
    parser.add_argument('--folders', type=str, help='the folders you want to generate wallpapers for. if for all folders under the root_dir, use "all", otherwise separate the target folders by a comma, no space')
    args = parser.parse_args()

    folders=sorted(os.listdir(args.root_dir)) if args.folders=='all' else args.folders.split(',')
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        print('processing', s)
        img23d=os.path.join(args.root_dir, s, 'for_image_text23d')
        os.makedirs(img23d, exist_ok=True)
        os.makedirs(os.path.join(img23d, 'furnitures_merged_2_pix2gestalt'), exist_ok=True)
        os.makedirs(os.path.join(img23d, 'small_objects_merged_12'), exist_ok=True)
        scene_folder=os.path.join(args.root_dir, s)
        
        if args.no_inpaint:
            os.rename(os.path.join(scene_folder, 'furnitures_cropped_seg_rgb_together_merged_2'),\
                    os.path.join(img23d, 'furnitures_merged_2_pix2gestalt/images'))
        else:
            os.makedirs(os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2_pix2gestalt'), exist_ok=True)
            os.makedirs(os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2_pix2gestalt/images'), exist_ok=True)
            response_file=open(os.path.join(scene_folder,'GPT_selection.json'),'w')
            responses={}
            for furniture in os.listdir(os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2')):
                if '.png' not in furniture: continue
            
                image_paths=sorted([os.path.join(os.path.join(scene_folder, 'pix2gestalt_inpaint',i)) for i in \
                            os.listdir(os.path.join(scene_folder, 'pix2gestalt_inpaint')) if furniture in i])
                furniture_category=furniture.split('.png')[0].split('_')[-1]
                response=ask_best_pix2gestalt(image_paths, furniture_category)
                caption_match = re.search(r'---BEGIN Index---\s*(.*?)\s*---END Index---', response, re.DOTALL)
                if caption_match:
                    parsed_caption = caption_match.group(1).strip()
                    # Remove the "Caption:" prefix if present
                    index = parsed_caption.replace("Index:", "").strip()
                    shutil.copyfile(os.path.join(scene_folder,'pix2gestalt_inpaint', str(index)+'_'+furniture),\
                                    os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2_pix2gestalt/images', furniture))
                else:
                    parsed_caption = ""
                    shutil.copyfile(os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2', furniture),\
                            os.path.join(scene_folder,'furnitures_cropped_seg_rgb_together_merged_2_pix2gestalt/images', furniture))
            
                responses[furniture]=response
                print(furniture, image_paths, response)
            json.dump(responses,response_file)

            os.rename(os.path.join(scene_folder, 'furnitures_cropped_seg_rgb_together_merged_2_pix2gestalt/images'),\
                    os.path.join(img23d, 'furnitures_merged_2_pix2gestalt/images'))
        
        os.rename(os.path.join(scene_folder, 'furnitures_cropped_seg_rgb_together_merged_2.json'), \
                os.path.join(img23d, 'furnitures_merged_2_pix2gestalt', 'furnitures_cropped_seg_rgb_together_merged_2.json'))
        os.rename(os.path.join(scene_folder, 'small_objects_cropped_seg_rgb_together_merged_12'), \
                os.path.join(img23d, 'small_objects_merged_12/images'))
        os.rename(os.path.join(scene_folder, 'small_objects_cropped_seg_rgb_together_merged_12.json'),\
                os.path.join(img23d, 'small_objects_merged_12', 'small_objects_cropped_seg_rgb_together_merged_12.json'))

        time2=time.time()-start2
        with open(os.path.join(args.root_dir, s, 'step4_pick_inpainted_time.txt'), 'w') as f:
            f.write(str(time1+time2))
        print('time1', time1, 'time2', time2, time1+time2)
        start2=time2
    