import time
start1=time.time()
import argparse
import torch
import numpy as np
import os
from transformers import GroundingDinoForObjectDetection, AutoProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL
from PIL import Image
import cv2
import sys
sys.path.insert(0, '/home/zg45/Inpaint-Anything')
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img
from segment_anything import build_sam, SamPredictor 
from openai import OpenAI
import json
import re
import base64
import requests
import jinja2
import shutil
import random


SEED=42


def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(SEED)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def ask_scene_small_objects(scene):
    template_scene_small_objects = jinja2.Template(
        r"""
    You are an expert who specializes in listing out common small-sized objects in a scene. If you do a good job, I can offer you $100 tips.
    What are the top 20 most common small-sized objects in a {{scene}}?
    Answer by the category name, separated by a comma and then a space. Use singular form of the noun, NO plural.
    DO NOT include large furniture or machine names like chair, table, etc.
    Your reply format is: object_name_1, object_name_2, ..., object_name_20.
    """.strip(),  # noqa: E501
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_scene_small_objects.render(scene=scene)
                },
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=500,
        seed=SEED
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

    
def ask_scene_furnitures(scene):
    # For each answer, nouns with a single word are preferred over multiple words. For example, seating is preferred over home theater seating, as the latter is consist of three words while the first is consist of only one word. The two adjective words, home theater, do not change the meaning much. However, if you want to output pool table, do NOT shorten it to table, as pool table is very different from a table.
    template_scene_furnitures = jinja2.Template(
            r"""
        You are an expert who specializes in listing out common large-sized furnitures in a scene. If you do a good job, I can offer you $100 tips.
        What are the top 20 most common large-sized furnitures in a {{scene}}?
        Answer by the category name, separated by a comma and then a space. Use singular form of the noun, NO plural.
        DO NOT include small-sized objects like plant, box, etc.
        Your reply format is: object_name_1, object_name_2, ..., object_name_20.
        """.strip(),  # noqa: E501
        )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":template_scene_furnitures.render(scene=scene)
                },
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=500,
        seed=SEED
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

time1=time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_category', type=str, default='chinese bedroom', help='Used to ask GPT about extra furnitures and small objects')
    args = parser.parse_args()

    os.makedirs('common_objects_scenes', exist_ok=True)
    count={}
    for i in range(10):
        reply=ask_scene_furnitures(args.scene_category).lower()
        names=reply[:-1].split(', ')
        print(i, '\n',reply, '\n',names, len(names))
        for n in names:
            if n not in count:
                count[n]=0
            count[n]+=1
    sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
    print(sorted_count)
    furnitures=','.join(sorted_count)
    f=open(os.path.join('common_objects_scenes', args.scene_category+'_furnitures.txt'), 'w')
    f.write(furnitures)
    
    count={}
    for i in range(10):
        reply=ask_scene_small_objects(args.scene_category).lower()
        names=reply[:-1].split(', ')
        print(i, '\n',reply, '\n',names, len(names))
        for n in names:
            if n not in count:
                count[n]=0
            count[n]+=1
    sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
    print(sorted_count)
    small_objects=','.join(sorted_count)
    f=open(os.path.join('common_objects_scenes', args.scene_category+'_small_objects.txt'), 'w')
    f.write(small_objects)

    print('furnitures,',furnitures)
    print('small_objects',small_objects)
    time2=time.time()-start2
    with open(os.path.join('common_objects_scenes', args.scene_category+'_time.txt'), 'w') as f:
        f.write(str(time1+time2))
    print('time1', time1, 'time2', time2)
    start2=time2
