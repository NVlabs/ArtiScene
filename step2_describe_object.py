import time
start1=time.time()
import argparse
import shutil
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import torch
from openai import OpenAI
import base64
import requests
import matplotlib as mpl
import json
import re
import jinja2
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

template_rotate4 = jinja2.Template(
    r"""
You are an expert scene designer who specializes in describing objects in natural language. If you do a good job, I can offer you $100 tips.
Customer will use the description to reconstruct the 3D object from the provided text description, so the description should only about the object itself, not the surrounding. You are responsible for helping people create detailed, amazing 3D objects. The way to accomplish this is to take their short text prompts and make them extremely detailed and descriptive.
You should always focusing on shape, geometry, and color. You should not mention anything not related to the object shape, geometry, and color, including but not limited to viewpoint of the object, the size of the object in real dimension, the background, unrelated objects and the overall atmosphere, etc. Your description must contain enough detail and information about shape, geometry, and color to reconstruct the 3D object.
I will provide a CROPPED image of this subject. It is a segmentation result, so if there were objects occluding part of it, those object regions are excluded. Therefore there might be holes on the object needing description, or the object may appear incomplete.
Do not describe the holes and incompletions. The objects all appear on a black background, so do not describe the background.
For example, "A glass-fronted display case for pastries and snacks" is bad because it mentions pastries and snacks which are NOT display case. Instead, give "Rectangular, glass-fronted display case with LED lights and a brushed aluminum frame." which only describes the display case itself.
For example, "Red fire extinguishers mounted on walls throughout the storage area, each with an easy-to-read pressure gauge." is bad because it mentions physcial location that is not part of the object. Instead, give "Red fire extinguisher with a cylindrical body, black hose, and an easy-to-read pressure gauge." which only describes the fire extinguisher itself.
For example, "A wooden dining table surrounded by four chairs" is bad because it mentions the chairs which are NOT the table. Instead, give "Rectangular wooden dining table with a polished surface and tapered legs." which only describes the table itself.
You can elaborate more object details such as:
1. Texture and color.
2. Material that the object is composed of.
3. Object parts. If different parts of the object have different colors, textures, or materials, you should specify them for each part.
4. Customer wants to target the theme of "{{scene_description}}", tailor your descriptions to match the theme.
5. Give no more than 3 sentences
6. Generate a Pose for the object. Desribe the pose of the object by which other object it is facing to. If there is no object that it faces to, reply NONE. If there are multiple, reply them all.
Output format:
---BEGIN Object---
Object_name:
---END Object---
---BEGIN Caption---
Caption:
---END Caption---
---BEGIN Pose---
Pose:
---END Pose---
The right one is the CROPPED image you have to describe:
""".strip(),  # noqa: E501
)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_json_dict(json_dict):
    processed_dict = {}
    for segment_id, data in json_dict.items():
        response = data['response']
        parsed_valid=data['category']
        parsed_caption = "a "+data['category']
        parsed_pose = ""
        try:
            # Extract the text between "---BEGIN Caption---" and "---END Caption---"
            caption_match = re.search(r'---BEGIN Caption---\s*(.*?)\s*---END Caption---', response, re.DOTALL)
            if caption_match:
                parsed_caption = caption_match.group(1).strip()
                # Remove the "Caption:" prefix if present
                parsed_caption = parsed_caption.replace("Caption:", "").strip()
    
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            valid_match = re.search(r'---BEGIN Object---\s*(.*?)\s*---END Object---', response, re.DOTALL)
            if valid_match:
                parsed_valid = valid_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_valid = parsed_valid.replace("Object_name:", "").strip()
    
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            pose_match = re.search(r'---BEGIN Pose---\s*(.*?)\s*---END Pose---', response, re.DOTALL)
            if pose_match:
                parsed_pose = pose_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_pose = parsed_pose.replace("Pose:", "").strip()

        except Exception as e:
            print(e)

        processed_dict[segment_id] = {
            'caption': parsed_caption,
            'object_name': parsed_valid,
        }
        if parsed_pose != "":
            processed_dict[segment_id]['pose']= parsed_pose
    return processed_dict

time1=time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_rgb_folder_stem', type=str, \
                        help='only folder name, does not need full path; should include the underscore')
    parser.add_argument('--image_path', type=str, help='need full path')
    parser.add_argument('--seg_rgb_folder_suffix', type=str, default='1,2', help='should not include the first underscore')
    args = parser.parse_args()


    base64_base_image = encode_image(args.image_path)
    save_dir=os.path.dirname(args.image_path)
    seg_rgb_folders=args.seg_rgb_folder_suffix.split(',')
    segments_folder=os.path.join(save_dir, args.seg_rgb_folder_stem+'merged_'+''.join(seg_rgb_folders))
    os.makedirs(segments_folder, exist_ok=True)
    
    
    for stem in ['small_objects_cropped_seg_rgb_together_', \
                 'small_objects_cropped_seg_together_', \
                 'furnitures_cropped_seg_rgb_together_', \
                 'furnitures_cropped_seg_together_', \
                 'small_objects_seg_together_', \
                 'furnitures_seg_together_', \
                ]:
        count=0
        merged_folder=os.path.join(save_dir,stem+'merged_'+''.join(seg_rgb_folders))
        os.makedirs(merged_folder, exist_ok=True)
        for suffix in seg_rgb_folders:
            for image_file in sorted(os.listdir(os.path.join(save_dir, stem+suffix))):
                if '.png' not in image_file: continue
                idx=('00'+str(count))[-3:]
                obj=image_file.split('.png')[0].split('_')[1]
                shutil.copyfile(os.path.join(save_dir, stem+suffix, image_file), \
                                os.path.join(merged_folder, idx+'_'+obj+'.png'))
                count += 1
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Describe the image and list all objects in the image',
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_base_image}",
                        "detail":"low"
                    },
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
    
    scene_description = response.choices[0].message.content
    print('scene descriptor', scene_description)
    
    results = {}
    segment_files = sorted(os.listdir(segments_folder))
    for segment_file in segment_files:
        segment_path = os.path.join(segments_folder, segment_file)
        segment_name = os.path.splitext(segment_file)[0].split('_')[1]
        segment_id=os.path.splitext(segment_file)[0]
    
    
        try:
            base64_base_image = encode_image(args.image_path)
            base64_cropped_image = encode_image(segment_path)
            prompt = template_rotate4.render(scene=scene_description)
    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_base_image}",
                                "detail":"low"
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Here is the CROPPED image to show the {segment_name} that needs a caption:",
                            "detail":"low"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_cropped_image}",
                                "detail":"low"
                            },
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
    
            results[segment_id] = {
                "category": segment_name,
                "prompt": prompt,
                "response": response.choices[0].message.content,
            }
            print('results[segment_id]["response"]', segment_id, '\n', results[segment_id]["response"])
        except Exception:
            print(segment_name)

    filtered_results = process_json_dict(results)
    output_path=segments_folder+'.json'
    with open(output_path, "w") as f:
        json.dump(filtered_results, f, indent=4)

    time2=time.time()-start2
    with open(os.path.join(save_dir, 'step2_describe_object_time_'+args.seg_rgb_folder_suffix+'.txt'), 'w') as f:
        f.write(str(time1+time2))
    print('time1', time1, 'time2', time2, time1+time2)
