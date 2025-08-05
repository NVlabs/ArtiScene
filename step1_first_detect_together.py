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
sys.path.insert(0, '../Inpaint-Anything')
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GroundingDinoForObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny')
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model.to(device)

sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device)
sam_predictor = SamPredictor(sam)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.cpu().reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))
    
def app_fn(
    image,
    labels,
    box_threshold,
    text_threhsold,
):
    labels = labels.split("\n")
    labels = [label if label.endswith(".") else label + "." for label in labels]
    labels = " ".join(labels)
    inputs = processor(images=image, text=labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threhsold,
        target_sizes=[image.size[::-1]]
    )[0]

    # convert tensor of [x0,y0,x1,y1] to list of [x0,y0,x1,y1] (int)
    boxes = result["boxes"].int().cpu().tolist()
    pred_labels = result["labels"]
    annot = [(tuple(box), pred_label) for box, pred_label in zip(boxes, pred_labels)]
    return annot

def filter_masks(masks, outs):
    h, w = masks[0].shape[-2:]
    mask_images = sorted([mask.cpu().reshape(1, h, w) for mask in masks], key=lambda x:x.int().sum())#[::-1]
    outs = [x for _, x in sorted(zip(masks, outs), key=lambda pair: pair[0].int().sum())]

    filtered_masks, filtered_outs=[], []
    for i in range(len(mask_images)):
        curr=mask_images[i]
        cover_curr=False
        coord, label=outs[i]
        left,up,right,down=coord
        box_w=right-left
        box_h=down-up
        if box_h*box_w>0.65*h*w:
            cover_curr=True
            print(i,'bbox larger than 65%')
            continue
        for j in range(i+1, len(mask_images)):
            # print(outs[i][1], outs[j][1])
            comp=mask_images[j] #comp should always be the larger one
            intersection = curr.logical_and(comp)
            intersection_curr = intersection.logical_and(curr)
            # if the overlap between the smaller one and the larger one is greater than 80% of 
            # the smaller one, omit the smaller one
            # the overlap is calculated by seg, not bbox!
            if intersection_curr.int().sum()/curr.int().sum()>0.8:
                print('test',i, outs[i][1],'covered by', j, outs[j][1])
                coord_comp =outs[j][0]
                left_comp,up_comp,right_comp,down_comp=coord_comp
                box_w_comp=right_comp-left_comp
                box_h_comp=down_comp-up_comp
                if box_h_comp*box_w_comp<=0.65*h*w:
                    cover_curr=True
                    print(i, outs[i][1],'covered by', j, outs[j][1])
                    break
        if not cover_curr:
            filtered_masks.append(mask_images[i])
            filtered_outs.append(outs[i])
    return torch.stack(filtered_masks), filtered_outs

def get_response_list_furnitures(base_image_path, scene):
    base64_base_image = encode_image(base_image_path)
    #  For each answer, nouns with a single word are preferred over multiple words. For example, seating is preferred over home theater seating, as the latter is consist of three words while the first is consist of only one word. The two adjective words, Home theater, do not change the meaning much. However, if you want to output pool table, do NOT shorten it to table, as pool table is very different from a table.
    template_scene_furnitures = jinja2.Template(
            r"""
        You are an expert who specializes in detecting furnitures, machines and equipments in a scene. 
        If you do a good job, I can offer you $100 tips.
        List all furnitures in the provided image of a {{scene}}.
        Answer by the category name, separated by a comma and then a space. Use singular form of the noun, NO plural.
        DO NOT include decorative objects like detergent, plant, box, etc, even though they may appear large in the image, such as a carpet.
        A furniture, machine or equipment often is large, but if it appears small in the image, you should still list it. For example, a stool, a lamp or a chair.
        There are some common objects that can be either type. To clarify, stool, lamp and chair are furnitures; carpet, rug, box are decorative objects.
        Your reply format is: object_name_1, object_name_2, ..., object_name_n. List ALL furnitures, machines and equipments you detected WITHOUT repetition. End the answer with a period.
        """.strip(),  # noqa: E501
        )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_scene_furnitures.render(scene=scene)
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
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_response_list_small_objects(base_image_path, scene):
    base64_base_image = encode_image(base_image_path)
    
    template_scene_small_objects = jinja2.Template(
        r"""
    You are an expert who specializes in detecting decorative or small-sized objects in a scene. If you do a good job, I can offer you $100 tips.
    List all decorative, small-sized objects in the provided image of a {{scene}}.
    Answer by the category name, separated by a comma and then a space. Use singular form of the noun, NO plural.
    DO NOT include furniture, equipment or machine names, even if they may appear small in the image. If an object is decorative but appears large in the image, such as a carpet or a box, you should still list it.
    There are some common objects that can be either type. To clarify, stool, lamp and chair are furnitures; carpet, rug, box are decorative objects.
    Your reply format is: object_name_1, object_name_2, ..., object_name_n. List ALL decorative or small-sized objects you detected WITHOUT repetition. End the answer with a period.
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
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


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
                    "text":template_scene_small_objects.render(scene=scene)
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

def ask_object_furniture_or_small_objects_novis(object_name):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'You are an expert who specializes in telling if a thing is a decorative or small-sized object or a '+
                    'furniture, machine, or equipment, with common sense. If you do a good job, I can offer you $100 tips. '+
                    'Which type does '+object_name+' belong to, (1) a furniture, machine, or equipment, '+\
                    'or (2) a decorative or small-sized object? Answer by (1) or (2).'+\
                    'There are some common objects that can be either type. To clarify, stool, lamp and chair are type (1) furnitures; carpet, rug, box are type (2) decorative objects.'
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


def ask_furniture_or_object(base_image_path, base_crop_image_path, scene_category):
    template_ask_furniture_or_object = jinja2.Template(
        r"""
    You are an expert who specializes in differentiating small-sized objects from large-sized furnitures. If you do a good job, I can offer you $100 tips.
    Here is an image of a  {{scene}}. I will send a second image that is CROPPED from this complete on, but with black background. 
    The second image contains an object. 
    Please first recognize what object it is, in a concise noun. Then if it is (1) a furniture, equipment or machine, 
    or (2) a decorative or small-sized object such as detergent, plants, etc. A type (1) object often is large, but if it appears small on the image, you should still answer (1). For example, a stool, a lamp, or a chair. A type (2) object often is small, but if it appears large on the image, you should still answer (2). For example, a carpet or a box.
    There are some common objects that can be either type. To clarify, stool, lamp and chair are type (1) furnitures; carpet, rug, box are type (2) decorative objects.
    Your answer format is:
    
    ---BEGIN Object---
    Object:
    ---END Object---
    ---BEGIN Choice---
    (1) or (2):
    ---END Choice---
    The right one is the CROPPED image you have to describe:
    """.strip(),  # noqa: E501
    )
    base64_base_image = encode_image(base_image_path)
    base64_crop_base_image = encode_image(base_crop_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_furniture_or_object.render(scene=scene_category)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_base_image}",
                        "detail":"low"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_crop_base_image}",
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
    # print(response.choices[0].message.content)
    return response.choices[0].message.content
    
def detect(args, mode_list):
    image=Image.open(args.image_path)
    H,W=image.size
    save_dir=os.path.dirname(args.image_path)
    image_np=np.array(image)
    suffix=args.suffix
    save=True

    all_objects_list=[]
    all_multi_words_list=[]
    for mode, (objects_list, multi_words_list) in mode_list.items():
        all_objects_list.extend(objects_list)
        all_multi_words_list.extend(multi_words_list)
    TEXT_PROMPT = "\n".join(all_objects_list)
    out=app_fn(image,TEXT_PROMPT,0.3,0.3)
    sam_predictor.set_image(image_np)
    boxes_xyxy=torch.tensor([np.array(i[0]) for i in out])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_np.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    filtered_masks, filtered_outs=filter_masks(masks, out)
    print('len(masks), len(filtered_masks)', len(masks), len(filtered_masks))
    masks, out=filtered_masks,filtered_outs
    
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(image)
    
    os.makedirs(os.path.join(save_dir,'all_bbox_'+suffix),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'all_cropped_'+suffix),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'all_cropped_seg_'+suffix),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'all_seg_'+suffix),exist_ok=True)
    for mode in mode_list:
        os.makedirs(os.path.join(save_dir,mode+'_bbox_'+suffix),exist_ok=True)
        os.makedirs(os.path.join(save_dir,mode+'_cropped_'+suffix),exist_ok=True)
    
    for i in range(len(out)):
        coord, label=out[i]
        idx=('00'+str(i))[-3:]
        blank=np.zeros((H,W)).astype('uint8')
        
        left,up,right,down=coord
        w=right-left
        h=down-up
        blank[up:down,left:right]=255
        cropped_image=Image.fromarray(image_np[up:down,left:right])
        rect = patches.Rectangle((left, up), w, h, linewidth=1, edgecolor='r', facecolor='none') #x first
        # Add the patch to the Axes
        ax.add_patch(rect)
        print('[bbox]', i, label, 'before parsing')
        # label=parse_label(label, all_multi_words_list)
        # print('[bbox]', i, label, 'after parsing')
        if save:
            # for mode, (objects_list, multi_words_list) in mode_list.items():
            #     if True: #label in objects_list:
            Image.fromarray(blank).save(os.path.join(save_dir, 'all_bbox_'+suffix, idx+'_'+label+'.png'))
            cropped_image.save(os.path.join(save_dir, 'all_cropped_'+suffix, idx+'_'+label+'.png'))
    
    fig.savefig(os.path.join(save_dir, 'together_detect_'+suffix+'.png'), dpi=90, bbox_inches='tight')
    
    for mode in mode_list:
        os.makedirs(os.path.join(save_dir,mode+'_cropped_seg_'+suffix),exist_ok=True)
        os.makedirs(os.path.join(save_dir,mode+'_cropped_seg_rgb_'+suffix),exist_ok=True)
        os.makedirs(os.path.join(save_dir,mode+'_seg_'+suffix),exist_ok=True)
        
    for i in range(len(out)):
        coord, label=out[i]
        print('seg', i)
        idx=('00'+str(i))[-3:]
        mask=masks[i][0].int().cpu().numpy()
    
        left,up,right,down=coord
        w=right-left
        h=down-up
        cropped_image=image_np[up:down,left:right]*mask[up:down,left:right][:,:,None]
        cropped_image=Image.fromarray(cropped_image.astype('uint8'))
        # label=parse_label(label, all_multi_words_list)
        if save:
            # for mode, (objects_list, multi_words_list) in mode_list.items():
            #     if True: #label in objects_list:
            Image.fromarray((mask*255).astype('uint8')).save(os.path.join(save_dir,'all_seg_'+suffix,idx+'_'+label+'.png'))
            Image.fromarray((mask[up:down,left:right]*255).astype('uint8')).save(\
                os.path.join(save_dir,'all_cropped_seg_'+suffix,idx+'_'+label+'.png'))
            cropped_image.save(os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix,idx+'_'+label+'.png'))


time1=time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_size', type=int, default=19)
    parser.add_argument('--inpaint', type=bool, default=True)
    parser.add_argument('--image_path', type=str, help='description for option1')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--scene_category', type=str, default=None, help='Used to ask GPT about extra furnitures and small objects')
    parser.add_argument('--lama_config', type=str, default='../Inpaint-Anything/lama/configs/prediction/default.yaml')
    parser.add_argument('--lama_ckpt', type=str, default='../Inpaint-Anything/pretrained_models/big-lama')
    args = parser.parse_args()

    save_dir=os.path.dirname(args.image_path)
    suffix=args.suffix
    
    if args.scene_category is None:
        f=open(os.path.join(save_dir, 'category.txt'),'r')
        args.scene_category=f.readline()
        args.scene_category=args.scene_category.split('style ')[-1]
    if args.scene_category+'_furnitures.txt' in os.listdir('common_objects_scenes'):
        f=open(os.path.join('common_objects_scenes', args.scene_category+'_furnitures.txt'), 'r')
        furnitures=f.readline()
        print('loaded furnitures', furnitures)
    else:
        count={}
        for i in range(10):
            reply=ask_scene_furnitures(args.scene_category).lower()
            print(i, reply)
            names=reply.replace('.','').split(', ')
            for n in names:
                if n not in count:
                    count[n]=0
                count[n]+=1
        sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
        furnitures=','.join(sorted_count)
        f=open(os.path.join('common_objects_scenes', args.scene_category+'_furnitures.txt'), 'w')
        f.write(furnitures)
        print('GPT asked furnitures', furnitures)
        
    if args.scene_category+'_small_objects.txt' in os.listdir('common_objects_scenes'):
        f=open(os.path.join('common_objects_scenes', args.scene_category+'_small_objects.txt'), 'r')
        small_objects=f.readline()
        print('loaded small_objects', small_objects)
    else:
        count={}
        for i in range(10):
            reply=ask_scene_small_objects(args.scene_category).lower()
            print(i, reply)
            names=reply.replace('.','').split(', ')
            for n in names:
                if n not in count:
                    count[n]=0
                count[n]+=1
        sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
        small_objects=','.join(sorted_count)
        f=open(os.path.join('common_objects_scenes', args.scene_category+'_small_objects.txt'), 'w')
        f.write(small_objects)
        print('GPT asked small_objects', small_objects)

    if 'GPT_detected_furnitures_'+suffix+'.txt' in os.listdir(save_dir):
        f=open(os.path.join(save_dir, 'GPT_detected_furnitures.txt'), 'r')
        extra_furnitures=f.readline()
        print('loaded extra_furnitures', extra_furnitures)
    else:
        count={}
        for i in range(5):
            reply=get_response_list_furnitures(args.image_path, args.scene_category).lower()
            print(i, reply)
            names=reply.replace('.','').split(', ')
            for n in names:
                if n not in count:
                    count[n]=0
                count[n]+=1
        sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
        extra_furnitures=','.join(sorted_count)
        f=open(os.path.join(save_dir, 'GPT_detected_extra_furnitures_'+suffix+'.txt'), 'w')
        f.write(extra_furnitures)
        print('GPT asked extra_furnitures', extra_furnitures)
        
    if 'GPT_detected_extra_small_objects_'+suffix+'.txt' in os.listdir(save_dir):
        f=open(os.path.join(save_dir, 'GPT_detected_extra_small_objects_'+suffix+'.txt'), 'r')
        extra_small_objects=f.readline()
        print('loaded extra_small_objects', extra_small_objects)
    else:
        count={}
        for i in range(5):
            reply=get_response_list_small_objects(args.image_path, args.scene_category).lower()
            print(i, reply)
            names=reply.replace('.','').split(', ')
            for n in names:
                if n not in count:
                    count[n]=0
                count[n]+=1
        sorted_count={k: v for k, v in sorted(count.items(), key=lambda item: -item[1])[:20]}
        extra_small_objects=','.join(sorted_count)
        f=open(os.path.join(save_dir, 'GPT_detected_extra_small_objects_'+suffix+'.txt'), 'w')
        f.write(extra_small_objects)
        print('GPT extra_small_objects', extra_small_objects)
    
    small_objects=small_objects.split(',')
    furnitures=furnitures.split(',')
    extra_small_objects=extra_small_objects.split(',')
    extra_furnitures=extra_furnitures.split(',')
    print('extra_furnitures', extra_furnitures)
    print('extra_small_objects', extra_small_objects)

    for extra_furniture in extra_furnitures:
        if extra_furniture not in furnitures:
            furnitures.append(extra_furniture)
    for extra_small_object in extra_small_objects:
        if (extra_small_object not in furnitures) and (extra_small_object not in small_objects):
            small_objects.append(extra_small_object)

    multi_words_small_objects, multi_words_furnitures=[],[] 
    print('final small_object',small_objects)
    print('final furnitures', furnitures)
    mode_dict={'small_objects': [small_objects, multi_words_small_objects],
               'furnitures':  [furnitures, multi_words_furnitures]} #[['elliptical_machine'],[]]}
    detect(args, mode_dict)    
    for i in os.listdir(os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix)):
        if 'png' not in i: continue
        response=ask_furniture_or_object(os.path.join(save_dir,'input.png'), os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix,i), args.scene_category)
        
        img=Image.open(os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix,i))
        idx=i.split('_')[0]
        h,w,_=np.asarray(img).shape
        print(i, response)
        # plt.imshow(img)
        # plt.show()
        
        # Extract the text between "---BEGIN Valid---" and "---END Valid---"
        name_match = re.search(r'---BEGIN Object---\s*(.*?)\s*---END Object---', response, re.DOTALL)
        if name_match:
            parsed_name = name_match.group(1).strip()
            # Remove the "Valid:" prefix if present
            parsed_name = parsed_name.replace("Object:", "").strip()
        else:
            parsed_name = i.split('.png')[0].split('_')[1]
    
        # Extract the text between "---BEGIN Valid---" and "---END Valid---"
        size_match = re.search(r'---BEGIN Choice---\s*(.*?)\s*---END Choice---', response, re.DOTALL)
        if size_match:
            parsed_size = size_match.group(1).strip()
            # Remove the "Valid:" prefix if present
            parsed_size = parsed_size.replace("(1) or (2):", "").strip()
        else:
            novis_answer=ask_object_furniture_or_small_objects_novis(parsed_name)
            parsed_size = novis_answer.strip()
            print('ask no vis!', parsed_name, parsed_size)
        
        if parsed_size == "(2)" or h*w<5000:
            dest='small_objects'
        elif parsed_size == "(1)":
            dest='furnitures'
        else:
            dest='small_objects'
        if h*w>35000:
            dest='furnitures'
        print('parsed_name',parsed_name,'parsed_size',parsed_size, 'size_match', size_match, 'dest', dest, 'crop_size',h,w,h*w)
        shutil.copyfile(os.path.join(save_dir,'all_cropped_seg_rgb_'+suffix, i),
                os.path.join(save_dir, dest+'_cropped_seg_rgb_'+suffix,idx+'_'+parsed_name+'.png'))
        shutil.copyfile(os.path.join(save_dir,'all_cropped_seg_'+suffix, i),
                os.path.join(save_dir, dest+'_cropped_seg_'+suffix,idx+'_'+parsed_name+'.png'))
        shutil.copyfile(os.path.join(save_dir,'all_seg_'+suffix, i),
                os.path.join(save_dir, dest+'_seg_'+suffix,idx+'_'+parsed_name+'.png'))
        shutil.copyfile(os.path.join(save_dir,'all_cropped_'+suffix, i),
                os.path.join(save_dir, dest+'_cropped_'+suffix,idx+'_'+parsed_name+'.png'))
        shutil.copyfile(os.path.join(save_dir,'all_bbox_'+suffix, i),
                os.path.join(save_dir, dest+'_bbox_'+suffix,idx+'_'+parsed_name+'.png'))                            

    if args.inpaint:
        image=Image.open(args.image_path)
        H,W=image.size
        blank=np.zeros((H,W))
        kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8) 
        for obj_seg_file in os.listdir(os.path.join(save_dir,'small_objects_seg_'+suffix)):
            if 'png' not in obj_seg_file: continue
            obj_seg=cv2.imread(os.path.join(save_dir,'small_objects_seg_'+suffix, obj_seg_file), 0)
            img_dilation = cv2.dilate(obj_seg, kernel, iterations=1)
            blank += img_dilation
        blank[blank>0]=255
        blank_mask=Image.fromarray(blank.astype('uint8'))
        inpaint_mask_path=os.path.join(save_dir,'small_objects_all_erode{kernel}_{suffix}.png'.format(kernel=args.kernel_size, suffix=suffix))
        blank_mask.save(inpaint_mask_path)
        
        
        inpaint_img = load_img_to_array(args.image_path)
        inpaint_mask = load_img_to_array(inpaint_mask_path)[:,:,0]
        inpainted_img_path = os.path.join(save_dir, 'inpainted_'+suffix+'.png')
        inpainted_img = inpaint_img_with_lama(
            inpaint_img, inpaint_mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(inpainted_img, inpainted_img_path)

    time2=time.time()-start2
    with open(os.path.join(save_dir, 'step1_detect_time_'+suffix+'.txt'), 'w') as f:
        f.write(str(time1+time2))
    print('time1', time1, 'time2', time2, time1+time2)
