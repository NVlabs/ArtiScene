import os
import numpy as np
from PIL import Image
from openai import OpenAI
import base64
import requests
import re
import jinja2
import shutil
import urllib
import random
import argparse

# put your api_key here
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

SEED=42

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED)

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def describe_wall(base_image_path, scene_category):
    template_ask_wall = jinja2.Template(
        r"""
    Here is an image of a {{scene}}. Our task is to describe the wall appearance in detail, such that I can generate a wall paper of a similar appearance.
    You should decribe only the texture, color, and material of the wall. If there are any objects attached to it, DO NOT describe them.
    DO NOT describe any windows or doors neither. DO NOT mention the {{scene}}. DO NOT mention that it is a wall. DO NOT mention any reflection or lighting effects.
    Your replyformat is:
    ---BEGIN Description---
    Description: 
    ---END Description---
    If the wall is mostly of a single color without obvious texture or pattern, reply an RGB tuple that could desccribe that color. Your reply format would
    then instead be: 
    ---BEGIN RGB---
    R, G, B: 
    ---END RGB---
    """.strip(),  # noqa: E501
    )
    base64_base_image = encode_image(base_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_wall.render(scene=scene_category)
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
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content



def describe_wall_color(base_image_path, scene_category):
    template_ask_wall_color = jinja2.Template(
        r"""
    Here is an image of a {{scene}}. Our task is to describe the wall color, such that I can generate a wall paper of a similar single color.
    If there are any objects attached to it, DO NOT consider them.
    DO NOT consider any windows or doors neither. DO NOT consider the {{scene}}. DO NOT consider any reflection or lighting effects.
    Reply a RGB tuple that best desccribes the wall color. Your reply format would be: 
    ---BEGIN RGB---
    R, G, B: 
    ---END RGB---
    """.strip(),  # noqa: E501
    )

    base64_base_image = encode_image(base_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_wall_color.render(scene=scene_category)
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
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content



def describe_floor_color(base_image_path, scene_category):
    template_ask_floor_color = jinja2.Template(
        r"""
    Here is an image of a {{scene}}. Our task is to describe the floor color, such that I can generate an image of a similar single color.
    If there are any objects on it, DO NOT consider them.
    DO NOT consider any windows or doors neither. DO NOT consider the {{scene}}. DO NOT consider any reflection or lighting effects.
    Reply a RGB tuple that best desccribes the floor color. Your reply format would be: 
    ---BEGIN RGB---
    R, G, B: 
    ---END RGB---
    """.strip(),  # noqa: E501
    )
    base64_base_image = encode_image(base_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_floor_color.render(scene=scene_category)
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
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def ask_best_wallpaper(image_path_lists):
    template_wallpaper = jinja2.Template(
        r"""
    Our task here is to find the image most suitable to be used as a wall paper. In the following list of images, an image is good if it (1) has pattern covering as much of the image as possible, and (2) the surface is flat WITHOUT distortions or rotations.
    An image is bad if (1) it contains one mid-sized sphere or sqaure at the center of the image, and (2) the texture on that sphere or saqure is different from the 
    rest of the image, because that texture does not fill the entire image anymore, and you should not choose it. If there
    are multiple good images, randomly select one. Reply by the index of your chose image. If all images are bad, reply -1 as the index.
    Your reply format is:
    ---BEGIN Index---
    Index:
    ---END Index---
    ---BEGIN Reason---
    Your reason:
    ---END Reason---
    """.strip(),  # noqa: E501
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":template_wallpaper.render()
                },
            ],
        }
    ]
    for i in range(len(image_path_lists)):
        messages[0]['content'].append({"type": "text", "text": "This is image index {i}".format(i=str(i))})
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
        seed=42
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def describe_wall_onlyTexture(base_image_path, scene_category):
    template_ask_wall_onlyTexture = jinja2.Template(
        r"""
    Here is an image of a {{scene}}. Our task is to describe the wall appearance in detail, such that I can generate a wall paper of a similar appearance.
    You should decribe only the pattern, color, and material of the wall. If there are any objects attached to it, DO NOT describe them.
    DO NOT describe any windows or doors neither. DO NOT mention the {{scene}}. DO NOT mention that it is a wall. .
    """.strip(),  # noqa: E501
    )
    base64_base_image = encode_image(base_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_wall_onlyTexture.render(scene=scene_category)
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
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content


def describe_floor(base_image_path, scene_category):
    template_ask_floor = jinja2.Template(
        r"""
    Here is an image of a {{scene}}. Our task is to describe the floor appearance in detail, such that I can generate a floor paper of a similar appearance.
    You should decribe only the texture, color, and material of the floor. If there are any objects on it, DO NOT describe them.
    DO NOT describe any windows or doors neither. DO NOT mention the {{scene}}. DO NOT mention that it is a floor. DO NOT mention any reflection or lighting effects.
    Your replyformat is:
    ---BEGIN Description---
    Description: 
    ---END Description---
    If the floor is mostly of a single color without obvious texture or pattern, reply an RGB tuple that could desccribe that color. Your reply format would
    then instead be: 
    ---BEGIN RGB---
    R, G, B: 
    ---END RGB---
    """.strip(),  # noqa: E501
    )
    base64_base_image = encode_image(base_image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_ask_floor.render(scene=scene_category)
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
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def gen_wallpaper(args):
    root_dir=args.root_dir
    folders=sorted(os.listdir(root_dir)) if args.folders=='all' else args.folders.split(',')
    
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        #ask wallpaper based on the image with small objects removed and inpainted
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        scene=args.parse_scene_category(s)
        reply=describe_wall_onlyTexture(os.path.join(root_dir, s, 'inpainted_together_2.png'), scene)
        print(scene, reply)
        for i in range(3):
            response = client.images.generate(
              model="dall-e-3",
              prompt="generate a texture image based on the following description: "+reply+'\n you should ONLY generate the texture on a flat plane, with potentially repetitive patten.'+\
                'The ENTIRE image should be FILLED by the texture EVERYWHERE. NO other objects, NO shadow should be present. DO NOT put a texture ball or square frame in the middle of the image.',
              size="1024x1024",
              quality="standard",
              n=1,
            )
            # The scene should be lit up by white light with NO other color.
            image_url = response.data[0].url
            print('image_url', image_url)
            urllib.request.urlretrieve(image_url, os.path.join(root_dir, s,f'wall_noball{i}.png'))
    
    #choose the best out of the candidates generated
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        #generate the scene prompt same way as the above loop
        scene=args.parse_scene_category(s)
        lists=[os.path.join(root_dir, s,f'wall_noball{i}.png') for i in range(3)]
        reply = ask_best_wallpaper(lists)
        # The scene should be lit up by white light with NO other color.
        print(s, reply)
        caption_match = re.search(r'---BEGIN Index---\s*(.*?)\s*---END Index---', reply, re.DOTALL)
        if caption_match:
            parsed_caption = caption_match.group(1).strip()
            # Remove the "Caption:" prefix if present
            parsed_caption = parsed_caption.replace("Index:", "").strip()
                           
            if parsed_caption !='-1':
                shutil.copyfile(os.path.join(root_dir, s,f'wall_noball{parsed_caption}.png'),\
                                os.path.join(root_dir, s,'chosen.png'))
    
    #if all candidates failed, add pure color as a possibility, and regenerate more candidates
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'chosen.png' in os.listdir(os.path.join(root_dir, s)): continue
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        #generate the scene prompt same way as the above loop
        scene=args.parse_scene_category(s)
        reply=describe_wall(os.path.join(root_dir, s, 'inpainted_together_2.png'), scene)
        print(s,reply)
        if 'RGB' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN RGB---\s*(.*?)\s*---END RGB---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("R, G, B:", "").strip()
                R,G,B=parsed_name.split(', ')
            else:
                R,G,B=255,255,255
            single_color=np.zeros((1024,1024,3)).astype('uint8')
            single_color[:,:,0]=int(R)
            single_color[:,:,1]=int(G)
            single_color[:,:,2]=int(B)
            Image.fromarray(single_color).save(os.path.join(root_dir, s, 'chosen1.png'))
        elif 'Description' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN Description---\s*(.*?)\s*---END Description---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("Description:", "").strip()
                
                for i in range(3):
                    response = client.images.generate(
                      model="dall-e-3",
                      prompt="generate a texture image based on the following description: "+parsed_name+'\n you should ONLY generate the texture on a flat plane, with potentially repetitive patten.'+\
                        'The ENTIRE image should be FILLED by the texture EVERYWHERE. NO other objects, NO shadow should be present. DO NOT put a texture ball or square frame in the middle of the image.',
                      size="1024x1024",
                      quality="standard",
                      n=1,
                    )
                    # The scene should be lit up by white light with NO other color.
                    image_url = response.data[0].url
                    urllib.request.urlretrieve(image_url, os.path.join(root_dir, s,f'wall_noball_again{i}.png'))
    
    
    # select again from the newly generated candidates
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'chosen.png' in os.listdir(os.path.join(root_dir, s)) or 'chosen1.png' in os.listdir(os.path.join(root_dir, s)): continue
        #generate the scene prompt same way as the above loop
        scene=args.parse_scene_category(s)
        lists=[os.path.join(root_dir, s,f'wall_noball_again{i}.png') for i in range(3)]
        reply = ask_best_wallpaper(lists)
        # The scene should be lit up by white light with NO other color.
        print(s, reply)
        caption_match = re.search(r'---BEGIN Index---\s*(.*?)\s*---END Index---', reply, re.DOTALL)
        if caption_match:
            parsed_caption = caption_match.group(1).strip()
            # Remove the "Caption:" prefix if present
            parsed_caption = parsed_caption.replace("Index:", "").strip()
            if parsed_caption !='-1':
                shutil.copyfile(os.path.join(root_dir, s,f'wall_noball_again{parsed_caption}.png'),\
                                os.path.join(root_dir, s,'chosen.png'))
    
    #for the remanining that get non qualified ones for two rounds, just use pure color.
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        if 'chosen.png' in os.listdir(os.path.join(root_dir, s)): continue
        #generate the scene prompt same way as the above loop
        scene=args.parse_scene_category(s)
        reply=describe_wall_color(os.path.join(root_dir, s, 'inpainted_together_2.png'), scene)
        print(s,reply)
        if 'RGB' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN RGB---\s*(.*?)\s*---END RGB---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("R, G, B:", "").strip()
                R,G,B=parsed_name.split(', ')
            else:
                R,G,B=255,255,255
            single_color=np.zeros((1024,1024,3)).astype('uint8')
            single_color[:,:,0]=int(R)
            single_color[:,:,1]=int(G)
            single_color[:,:,2]=int(B)
            Image.fromarray(single_color).save(os.path.join(root_dir, s, 'chosen.png'))


def gen_floor(args):
    root_dir=args.root_dir
    folders=sorted(os.listdir(root_dir)) if args.folders=='all' else args.folders.split(',')
    
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        scene=args.parse_scene_category(s)
        reply=describe_floor(os.path.join(root_dir, s, 'inpainted_together_2.png'), scene)
        print(s,reply)
        if 'RGB' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN RGB---\s*(.*?)\s*---END RGB---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("R, G, B:", "").strip()
                R,G,B=parsed_name.split(', ')
            else:
                R,G,B=255,255,255
            single_color=np.zeros((1024,1024,3)).astype('uint8')
            single_color[:,:,0]=int(R)
            single_color[:,:,1]=int(G)
            single_color[:,:,2]=int(B)
            Image.fromarray(single_color).save(os.path.join(root_dir, s, 'floor.png'))
        elif 'Description' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN Description---\s*(.*?)\s*---END Description---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("Description:", "").strip()
                
                for i in range(3):
                    response = client.images.generate(
                      model="dall-e-3",
                      prompt="generate a texture image based on the following description: "+parsed_name+'\n you should ONLY generate the texture on a flat plane, with potentially repetitive patten.'+\
                        'The ENTIRE image should be FILLED by the texture EVERYWHERE. NO other objects, NO shadow should be present. DO NOT put a texture ball or square frame in the middle of the image.',
                      size="1024x1024",
                      quality="standard",
                      n=1,
                    )
                    # The scene should be lit up by white light with NO other color.
                    image_url = response.data[0].url
                    urllib.request.urlretrieve(image_url, os.path.join(root_dir, s,f'floor_noball_again{i}.png'))

    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'floor.png' in os.listdir(os.path.join(root_dir, s)) : continue
        scene=args.parse_scene_category(s)
        lists=[os.path.join(root_dir, s,f'floor_noball_again{i}.png') for i in range(3)]
        try:
            reply = ask_best_wallpaper(lists)
            # The scene should be lit up by white light with NO other color.
            print(s, reply)
            caption_match = re.search(r'---BEGIN Index---\s*(.*?)\s*---END Index---', reply, re.DOTALL)
            if caption_match:
                parsed_caption = caption_match.group(1).strip()
                # Remove the "Caption:" prefix if present
                parsed_caption = parsed_caption.replace("Index:", "").strip()
                if parsed_caption !='-1':
                    shutil.copyfile(os.path.join(root_dir, s,f'floor_noball_again{parsed_caption}.png'),\
                                    os.path.join(root_dir, s,'floor.png'))
        except:
            continue


    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'floor.png' in os.listdir(os.path.join(root_dir, s)) : continue
        scene=args.parse_scene_category(s)
        lists=[os.path.join(root_dir, s,f'floor_noball_again{i}.png') for i in range(3)]
        try:
            reply = ask_best_wallpaper(lists)
            # The scene should be lit up by white light with NO other color.
            print(s, reply)
            caption_match = re.search(r'---BEGIN Index---\s*(.*?)\s*---END Index---', reply, re.DOTALL)
            if caption_match:
                parsed_caption = caption_match.group(1).strip()
                # Remove the "Caption:" prefix if present
                parsed_caption = parsed_caption.replace("Index:", "").strip()
                if parsed_caption !='-1':
                    shutil.copyfile(os.path.join(root_dir, s,f'floor_noball_again{parsed_caption}.png'),\
                                    os.path.join(root_dir, s,'floor.png'))
        except:
            continue

    #for the remanining that get non qualified ones for two rounds, just use pure color.
    for s in folders:
        if 'DS' in s or 'ipynb' in s: continue
        if 'inpainted_together_2.png' not in os.listdir(os.path.join(root_dir, s)): continue
        if 'floor.png' in os.listdir(os.path.join(root_dir, s)): continue
        scene=args.parse_scene_category(s)

        reply=describe_floor_color(os.path.join(root_dir, s, 'inpainted_together_2.png'), scene)
        print(s,reply)
        if 'RGB' in reply:
            # Extract the text between "---BEGIN Valid---" and "---END Valid---"
            name_match = re.search(r'---BEGIN RGB---\s*(.*?)\s*---END RGB---', reply, re.DOTALL)
            if name_match:
                parsed_name = name_match.group(1).strip()
                # Remove the "Valid:" prefix if present
                parsed_name = parsed_name.replace("R, G, B:", "").strip()
                R,G,B=parsed_name.split(', ')
            else:
                R,G,B=255,255,255
            single_color=np.zeros((1024,1024,3)).astype('uint8')
            single_color[:,:,0]=int(R)
            single_color[:,:,1]=int(G)
            single_color[:,:,2]=int(B)
            Image.fromarray(single_color).save(os.path.join(root_dir, s, 'floor.png'))

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='the structure should be root_dir/folder1, folder2, ..., where each folder if of a scene')
    parser.add_argument('--folders', type=str, help='the folders you want to generate wallpapers for. if for all folders under the root_dir, use "all", otherwise separate the target folders by a comma, no space')
    args = parser.parse_args()

    # the scene should be a prompt of format "a [style, optional] [scene category]", if the info is embedded in the folder name
    # such as 'chinese_bedroom', we could form the prompt by parsing the folder name as below. Otherwise, pls fill in the
    # prompt manually
    
    args.parse_scene_category=lambda folder_name:'a '+folder_name.split('_')[0]
    # another example: for 'styled_rooms' root_dir: scene=s.split('_')[1]+'-styled '+s.split('_')[2]
    
    gen_wallpaper(args)
    #Now all folders should have a wallpaper with file name chosen.png, prioritizing texture over pure color.
    gen_floor(args)
    #Now all folders should have a wallpaper with file name floor.png, prioritizing texture over pure color.

