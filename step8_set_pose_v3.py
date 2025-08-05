import os
import re
import math
import argparse
import numpy as np
import bpy
import mathutils
from mathutils import Euler, Vector
import json
from PIL import Image
import OpenEXR as exr
from math import radians
import Imath
import sys
from openai import OpenAI
import base64
import jinja2
import requests
import random

SEED=42


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
seed_everything(SEED)

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

cube = bpy.data.objects["Cube"]
bpy.data.objects.remove(cube, do_unlink=True)

light_data = bpy.data.lights.new(name="light1", type='AREA')
light_data.energy = 3000
# create new object with our light datablock
light_object = bpy.data.objects.new(name="light1", object_data=light_data)
# link light object
bpy.context.collection.objects.link(light_object)
light_object.location=Vector((-1.736,0.4,10))
light_object.rotation_mode = 'XYZ'
light_object.rotation_euler=Euler((radians(2.12), radians(-11), radians(-8)), 'XYZ')

light_data = bpy.data.lights.new(name="light2", type='AREA')
light_data.energy = 3000
# create new object with our light datablock
light_object = bpy.data.objects.new(name="light2", object_data=light_data)
# link light object
bpy.context.collection.objects.link(light_object)
light_object.location=Vector((11,-2,9.55))
light_object.rotation_mode = 'XYZ'
light_object.rotation_euler=Euler((radians(43.59), radians(27.21), radians(53.52)), 'XYZ')
    
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
scene = bpy.context.scene
links = tree.links
cam = scene.objects["Camera"]
cam.data.type='ORTHO'
cam.data.ortho_scale=1.
cam.rotation_euler=Euler((54.736*np.pi/180, 0.0, 45*np.pi/180), 'XYZ')
cam.location = (0, 0, 0)
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024

for n in tree.nodes:
    tree.nodes.remove(n)
scene.view_layers["ViewLayer"].use_pass_z = True
rl = tree.nodes.new('CompositorNodeRLayers')
depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = "Depth Output"
depth_file_output.file_slots[0].use_node_format = True
depth_file_output.format.file_format = "OPEN_EXR"
depth_file_output.format.color_depth = "32"
links.new(rl.outputs["Depth"], depth_file_output.inputs[0])

rgb_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
rgb_file_output.label = "RGB Output"
rgb_file_output.file_slots[0].use_node_format = True
rgb_file_output.format.file_format = "PNG"
links.new(rl.outputs["Image"], rgb_file_output.inputs[0])

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def nothing_below(furniture_name, furnitures_dict):
    for name, info in furnitures_dict.items():
        if name==furniture_name:continue
        if 'carpet' in name.lower() or 'rug' in name.lower() or 'mat' in name.lower() or 'fence' in name.lower(): continue
        intersect_w, intersect_h=intersect_area(furnitures_dict[furniture_name]['bbox'], furnitures_dict[name]['bbox'])
        if intersect_w/(furnitures_dict[furniture_name]['bbox'][1]-furnitures_dict[furniture_name]['bbox'][0])>0.75 and\
        intersect_h/(furnitures_dict[furniture_name]['bbox'][3]-furnitures_dict[furniture_name]['bbox'][2])>0.75 and\
        furnitures_dict[name]['bbox'][4]< furnitures_dict[furniture_name]['bbox'][4] :
            print('two furnitures are stacked!', name, furniture_name)
            return False 
    print('nothing below', furniture_name)
    return True

def sort_fall_to_ground(furnitures_dict, small_objects_dict, floor_fall_threshold):
    room_x_min,room_x_max,room_y_min,room_y_max,room_z_min,room_z_max=np.inf, -np.inf,np.inf,-np.inf,np.inf,-np.inf
    all_dict={**furnitures_dict, **small_objects_dict}
    for name, info in all_dict.items():
        x_min,x_max,y_min,y_max,z_min,z_max=info['bbox']
        room_x_min=min(room_x_min, x_min)
        room_y_min=min(room_y_min, y_min)
        room_z_min=min(room_z_min, z_min)
        room_x_max=max(room_x_max, x_max)
        room_y_max=max(room_y_max, y_max)
        room_z_max=max(room_z_max, z_max)

    # xy_center_of_room=[(room_x_min+room_x_max)/2, (room_y_min+room_y_max)/2, room_z_min]
    xy_center_of_room=[room_x_min,room_y_max, room_z_min]

    furnitures_dist_name=[]
    for name, info in furnitures_dict.items():
        dist=np.linalg.norm(np.array(info['center'][:])-np.array(xy_center_of_room[:]))
        furnitures_dist_name.append((dist, name))

    furnitures_dist_name=sorted(furnitures_dist_name,key=lambda x:x[0]) #sort by distance to center
    small_objects_dist_name=[]
    for name, info in small_objects_dict.items():
        dist=np.linalg.norm(np.array(info['center'][:])-np.array(xy_center_of_room[:]))
        small_objects_dist_name.append((dist, name))

    small_objects_dist_name=sorted(small_objects_dist_name,key=lambda x:x[0]) #sort by distance to center
    print('sorted', furnitures_dist_name, small_objects_dist_name)
    for dist,name in furnitures_dist_name:
        if abs(furnitures_dict[name]['bbox'][4]-room_z_min)<floor_fall_threshold and nothing_below(name, furnitures_dict):
            disp_to_floor=room_z_min-furnitures_dict[name]['bbox'][4]
            furnitures_dict[name]['center'][2]+=disp_to_floor
            furnitures_dict[name]['bbox'][4]+=disp_to_floor
            furnitures_dict[name]['bbox'][5]+=disp_to_floor
            furnitures_dict[name]['on_floor']=True
            print('furniture should be on the floor',name, furnitures_dict[name]['asso_obj'], disp_to_floor)
        else:
            furnitures_dict[name]['on_floor']=False
        if 'asso_obj' in furnitures_dict[name]:
            for small_object in furnitures_dict[name]['asso_obj']:
                object_disp_to_furniture=furnitures_dict[name]['bbox'][5]-small_objects_dict[small_object]['bbox'][4]
                small_objects_dict[small_object]['center'][2]+=object_disp_to_furniture#disp_to_floor
                small_objects_dict[small_object]['bbox'][4]+=object_disp_to_furniture#disp_to_floor
                small_objects_dict[small_object]['bbox'][5]+=object_disp_to_furniture#disp_to_floor

    for dist,name in small_objects_dist_name:
        asso=False
        for _, info in furnitures_dict.items():
            if name in info['asso_obj']:
                asso=True
                break
        if abs(small_objects_dict[name]['bbox'][4]-room_z_min)<0.1 and asso==False:
            print('small object falls to the ground', name)
            disp_to_floor=room_z_min-small_objects_dict[name]['bbox'][4]
            small_objects_dict[name]['center'][2]+=disp_to_floor
            small_objects_dict[name]['bbox'][4]+=disp_to_floor
            small_objects_dict[name]['bbox'][5]+=disp_to_floor
            small_objects_dict[name]['on_floor']=True
        else:
            small_objects_dict[name]['on_floor']=False
    return furnitures_dict, small_objects_dict, furnitures_dist_name, small_objects_dist_name

def inverse_scale_obj(w,h,degree):
    if degree %180==0:
        return w,h
    elif degree %180==90:
        return h,w
    elif (degree %180==45) or (degree %180==135):
        min_side=min(w,h)
        return min_side/np.cos(45*np.pi/180), min_side/np.cos(45*np.pi/180)
    
def push_to_wall(furnitures_dict, small_objects_dict, wall_push_threshold):
    room_x_min,room_x_max,room_y_min,room_y_max,room_z_min,room_z_max=np.inf, -np.inf,np.inf,-np.inf,np.inf,-np.inf
    all_dict={**furnitures_dict, **small_objects_dict}
    # latest room limits
    for name, info in all_dict.items():
        x_min,x_max,y_min,y_max,z_min,z_max=info['bbox']
        room_x_min=min(room_x_min, x_min)
        room_y_min=min(room_y_min, y_min)
        room_z_min=min(room_z_min, z_min)
        room_x_max=max(room_x_max, x_max)
        room_y_max=max(room_y_max, y_max)
        room_z_max=max(room_z_max, z_max)

    # there are 2 walls: -x and +y
    for name in furnitures_dict:
        if abs(furnitures_dict[name]['bbox'][0]-room_x_min)<wall_push_threshold:
            disp_to_wall=room_x_min-furnitures_dict[name]['bbox'][0]
            furnitures_dict[name]['center'][0]+=disp_to_wall
            furnitures_dict[name]['bbox'][0]+=disp_to_wall
            furnitures_dict[name]['bbox'][1]+=disp_to_wall
            print('furniture should be on the left wall',name, furnitures_dict[name]['asso_obj'], disp_to_wall)
            furnitures_dict[name]['on_wall']='x'
            if 'asso_obj' in furnitures_dict[name]:
                for small_object in furnitures_dict[name]['asso_obj']:
                    small_objects_dict[small_object]['center'][0]+=disp_to_wall
                    small_objects_dict[small_object]['bbox'][0]+=disp_to_wall
                    small_objects_dict[small_object]['bbox'][1]+=disp_to_wall

    for name in furnitures_dict:
        if abs(furnitures_dict[name]['bbox'][3]-room_y_max)<wall_push_threshold:
            disp_to_wall=room_y_max-furnitures_dict[name]['bbox'][3]
            furnitures_dict[name]['center'][1]+=disp_to_wall
            furnitures_dict[name]['bbox'][2]+=disp_to_wall
            furnitures_dict[name]['bbox'][3]+=disp_to_wall
            print('furniture should be on the left wall',name, furnitures_dict[name]['asso_obj'], disp_to_wall)
            furnitures_dict[name]['on_wall']='y'
            if 'asso_obj' in furnitures_dict[name]:
                for small_object in furnitures_dict[name]['asso_obj']:
                    small_objects_dict[small_object]['center'][1]+=disp_to_wall
                    small_objects_dict[small_object]['bbox'][2]+=disp_to_wall
                    small_objects_dict[small_object]['bbox'][3]+=disp_to_wall

    for name in small_objects_dict:
        if abs(small_objects_dict[name]['bbox'][0]-room_x_min)<wall_push_threshold:
            print('small object should be on the left wall', name)
            disp_to_wall=room_x_min-small_objects_dict[name]['bbox'][0]
            small_objects_dict[name]['center'][0]+=disp_to_wall
            small_objects_dict[name]['bbox'][0]+=disp_to_wall
            small_objects_dict[name]['bbox'][1]+=disp_to_wall
            small_objects_dict[name]['on_wall']='x'

    for name in small_objects_dict:
        if abs(small_objects_dict[name]['bbox'][3]-room_y_max)<wall_push_threshold:
            print('small object should be on the right wall', name)
            disp_to_wall=room_y_max-small_objects_dict[name]['bbox'][3]
            small_objects_dict[name]['center'][1]+=disp_to_wall
            small_objects_dict[name]['bbox'][2]+=disp_to_wall
            small_objects_dict[name]['bbox'][3]+=disp_to_wall
            small_objects_dict[name]['on_wall']='y'
    return furnitures_dict, small_objects_dict

        
def check_intersect_and_move(inner_name, outer_name, source_dict, move_dict, small_objects_dict=None):
    """
    source_dict is to compare and if criterions are met, move the object in the move_dict
    """
    inner_bbox=source_dict[inner_name]['bbox']
    outer_bbox=move_dict[outer_name]['bbox']
    disp_x,disp_y,disp_z=np.inf,np.inf,np.inf
    if inner_bbox[0]<outer_bbox[1] and outer_bbox[0]<inner_bbox[1] and \
    inner_bbox[2]<outer_bbox[3] and outer_bbox[2]<inner_bbox[3] and \
    inner_bbox[4]<outer_bbox[5] and outer_bbox[4]<inner_bbox[5]:
        if move_dict[outer_name]['on_wall']!='x': #if on the x wall, can't move in x direction
            # if inner_bbox[0]>=outer_bbox[0]:
            #     disp_x=inner_bbox[0]-outer_bbox[1] #outer on the left of inner, move left
            #     print(outer_name, 'is into the screen than', inner_name, 'move inside',disp_x,outer_bbox[:2], inner_bbox[:2])
            # else:
            disp_x=inner_bbox[1]-outer_bbox[0] #outer on the right of inner, move right
            print(outer_name, 'is closer to the screen than', inner_name, 'move outside',disp_x,outer_bbox[:2], inner_bbox[:2])

        if move_dict[outer_name]['on_wall']!='y': #if on the y wall, can't move in y direction
            # if inner_bbox[2]>=outer_bbox[2]:
            disp_y=inner_bbox[2]-outer_bbox[3] #outer on the left of inner, move left
            print(outer_name, 'is on the left of', inner_name, 'move left',disp_y,outer_bbox[2:4], inner_bbox[2:4])
            # else:
            #     disp_y=inner_bbox[3]-outer_bbox[2] #outer on the right of inner, move right
            #     print(outer_name, 'is on the right of', inner_name, 'move right',disp_y,outer_bbox[2:4], inner_bbox[2:4])

        if move_dict[outer_name]['on_floor']==False:
            if inner_bbox[4]>=outer_bbox[4]:
                disp_z=inner_bbox[4]-outer_bbox[5] #outer on the left of inner, move left
                print(outer_name, 'is on the bottom of', inner_name, 'move down', disp_z,outer_bbox[4:], inner_bbox[4:])
            else:
                disp_z=inner_bbox[5]-outer_bbox[4] #outer on the right of inner, move right
                print(outer_name, 'is on the top of', inner_name, 'move up', disp_z,outer_bbox[4:], inner_bbox[4:])

    if disp_x!=np.inf or disp_y!=np.inf or disp_z!=np.inf:
        outer_w,outer_h,outer_l=move_dict[outer_name]['dim']
        print('finally, outer moves', outer_name)
        if abs(disp_x)/outer_w==min(abs(disp_x)/outer_w, abs(disp_y)/outer_h, abs(disp_z)/outer_l):
            print('in x', disp_x)
            move_dict[outer_name]['center'][0]+=disp_x 
            move_dict[outer_name]['bbox'][0]+=disp_x 
            move_dict[outer_name]['bbox'][1]+=disp_x 
            if 'asso_obj' in move_dict[outer_name] and small_objects_dict:
                for small_object in move_dict[outer_name]['asso_obj']:
                    small_objects_dict[small_object]['center'][0]+=disp_x 
                    small_objects_dict[small_object]['bbox'][0]+=disp_x 
                    small_objects_dict[small_object]['bbox'][1]+=disp_x 
        elif abs(disp_y)/outer_h==min(abs(disp_x)/outer_w, abs(disp_y)/outer_h, abs(disp_z)/outer_l):
            print('in y', disp_y)
            move_dict[outer_name]['center'][1]+=disp_y
            move_dict[outer_name]['bbox'][2]+=disp_y
            move_dict[outer_name]['bbox'][3]+=disp_y
            if 'asso_obj' in move_dict[outer_name] and small_objects_dict:
                for small_object in move_dict[outer_name]['asso_obj']:
                    print(small_object, small_objects_dict[small_object])
                    small_objects_dict[small_object]['center'][1]+=disp_y
                    small_objects_dict[small_object]['bbox'][2]+=disp_y
                    small_objects_dict[small_object]['bbox'][3]+=disp_y
        elif abs(disp_z)/outer_l==min(abs(disp_x)/outer_w, abs(disp_y)/outer_h, abs(disp_z)/outer_l):
            print('in z', disp_z)
            move_dict[outer_name]['center'][2]+=disp_z
            move_dict[outer_name]['bbox'][4]+=disp_z
            move_dict[outer_name]['bbox'][5]+=disp_z
            if 'asso_obj' in move_dict[outer_name] and small_objects_dict:
                for small_object in move_dict[outer_name]['asso_obj']:
                    small_objects_dict[small_object]['center'][2]+=disp_z
                    small_objects_dict[small_object]['bbox'][4]+=disp_z
                    small_objects_dict[small_object]['bbox'][5]+=disp_z
    

# def place_furnitures(args):
#     furnitures_info=json.load(open(os.path.join(args.root_dir, args.dict_file), 'r'))
#     glb_dir=os.path.join(args.root_dir, args.image_folder, "edify3D")
#     furnitures_info={k:v for k,v in furnitures_info.items() if k in os.listdir(glb_dir)}
#     for _, new_dict in furnitures_info.items():
#         new_dict['bbox']=[new_dict['center'][0]-new_dict['dim'][0]/2,new_dict['center'][0]+new_dict['dim'][0]/2,\
#                 new_dict['center'][1]-new_dict['dim'][1]/2,new_dict['center'][1]+new_dict['dim'][1]/2,\
#                 new_dict['center'][2]-new_dict['dim'][2]/2, new_dict['center'][2]+new_dict['dim'][2]/2]
    
#     remove_overlap(furnitures_info)
            
#     for target_name in os.listdir(glb_dir):
#         if target_name not in furnitures_info: continue
#         # Import the file
#         id_folder=[i for i in os.listdir(os.path.join(glb_dir, target_name)) if ('response' not in i) and ('.DS_Store' not in i)]
#         assert len(id_folder)==1
#         id_folder=id_folder[0]
#         bpy.ops.import_scene.gltf(filepath=os.path.join(glb_dir, target_name, id_folder, 'mesh_0.glb'))
#         for obj in bpy.context.selected_objects[:]:
#             obj.name = target_name
#             obj.hide_render = False
#         obj.rotation_mode = 'XYZ'

#         w,h,l=furnitures_info[target_name]['dim']
#         x_center,y_center,z_center=furnitures_info[target_name]['center']
#         z_rot_deg=furnitures_info[target_name]['pose']
#         obj.dimensions=Vector((w, h, l))
#         obj.location=Vector((x_center,y_center,z_center))
#         obj.rotation_euler=Euler((0.0, 0.0, z_rot_deg*np.pi/180), 'XYZ')
def intersect_area(bboxA, bboxB):
    if bboxA[1]<bboxB[0] or bboxA[0]>bboxB[1] or bboxA[3]<bboxB[2] or bboxA[2]>bboxB[3]:
        return 0,0
    width= bboxB[1]-bboxA[0] if bboxA[1]>bboxB[1] else bboxA[1]-bboxB[0]
    height=bboxB[3]-bboxA[2] if bboxA[3]>bboxB[3] else bboxA[3]-bboxB[2]
    return width, height
def associate_furnitures_objects(furnitures_info, small_objects_info, floor_fall_threshold):
    for furniture_name, furnitures_dict in furnitures_info.items():
        furnitures_dict['asso_obj']=[]
        for small_object_name, small_objects_dict in small_objects_info.items():
            #only consider a small object is on top of a furniture
            if abs(furnitures_dict['bbox'][5]-small_objects_dict['bbox'][4])<=floor_fall_threshold:
                # furnitures_dict['bbox'][5]>small_objects_dict['bbox'][5]) and\
                #    furnitures_dict['bbox'][4]<small_objects_dict['bbox'][4] :
                intersect_w, intersect_h=intersect_area(furnitures_dict['bbox'], small_objects_dict['bbox'])
                # print( small_object_name, furniture_name, intersect_w, intersect_h)
                if intersect_w/(small_objects_dict['bbox'][1]-small_objects_dict['bbox'][0])>0.75 and\
                intersect_h/(small_objects_dict['bbox'][3]-small_objects_dict['bbox'][2])>0.75:
                    print('associate', small_object_name, 'with', furniture_name)
                    if small_objects_dict['asso_furniture']==None:
                        furnitures_dict['asso_obj'].append(small_object_name)
                        small_objects_dict['asso_furniture']=furniture_name
                    else:
                        #associated to the tallest
                        if furnitures_dict['bbox'][5]>furnitures_info[small_objects_dict['asso_furniture']]['bbox'][5]:
                            furnitures_info[small_objects_dict['asso_furniture']]['asso_obj'].remove(small_object_name)
                            furnitures_dict['asso_obj'].append(small_object_name)
                            small_objects_dict['asso_furniture']=furniture_name

def ask_floor_color(base_image_path):
    template_color = jinja2.Template(
        r"""
    You are an expert  who specializes in describing colors of floors in natural language. If you do a good job, I can offer you $100 tips.
    Below is an image of a scene. What is the color of the floor in this image? Your estimation should get rid of the effects of lighting.
    Reply by a description of the color, then a single RGB tuple in the format of (R,G,B). If there are multiple colors, reply only the most dominant one.
    Output format:
    ---BEGIN Color Name---
    Color Name:
    ---END Color Name---
    ---BEGIN RGB Tuple---
    (R,G,B):
    ---END RGB Tuple---
    """.strip(),  # noqa: E501
    )

    
    base64_base_image = encode_image(base_image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_color.render()
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
    print('ask_floor_color original response', response.choices[0].message.content)
    response= response.choices[0].message.content
    valid_color = re.search(r'---BEGIN Color Name---\s*(.*?)\s*---END Color Name---', response, re.DOTALL)
    if valid_color:
        parsed_color = valid_color.group(1).strip()
        # Remove the "Valid:" prefix if present
        parsed_color = parsed_color.replace("Color Name:", "").strip()
    else:
        parsed_color = ""
    valid_rgb = re.search(r'---BEGIN RGB Tuple---\s*(.*?)\s*---END RGB Tuple---', response, re.DOTALL)
    if valid_rgb:
        parsed_rgb = valid_rgb.group(1).strip()
        # Remove the "Valid:" prefix if present
        parsed_rgb = parsed_rgb.replace("(R,G,B):", "").strip()
    else:
        parsed_rgb = ""
    return parsed_color, parsed_rgb

def ask_wall_color(base_image_path):
    template_color = jinja2.Template(
        r"""
    You are an expert  who specializes in describing colors of walls in natural language. If you do a good job, I can offer you $100 tips.
    Below is an image of a scene. What is the color of the walls in this image? Your estimation should get rid of the effects of lighting.
    Reply by a description of the color, then a single RGB tuple in the format of (R,G,B). If there are multiple colors, reply only the most dominant one.
    Output format:
    ---BEGIN Color Name---
    Color Name:
    ---END Color Name---
    ---BEGIN RGB Tuple---
    (R,G,B):
    ---END RGB Tuple---
    """.strip(),  # noqa: E501
    )

    
    base64_base_image = encode_image(base_image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": template_color.render()
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
    print('ask_wall_color original response', response.choices[0].message.content)
    response= response.choices[0].message.content
    valid_color = re.search(r'---BEGIN Color Name---\s*(.*?)\s*---END Color Name---', response, re.DOTALL)
    if valid_color:
        parsed_color = valid_color.group(1).strip()
        # Remove the "Valid:" prefix if present
        parsed_color = parsed_color.replace("Color Name:", "").strip()
    else:
        parsed_color = ""
    valid_rgb = re.search(r'---BEGIN RGB Tuple---\s*(.*?)\s*---END RGB Tuple---', response, re.DOTALL)
    if valid_rgb:
        parsed_rgb = valid_rgb.group(1).strip()
        # Remove the "Valid:" prefix if present
        parsed_rgb = parsed_rgb.replace("(R,G,B):", "").strip()
    else:
        parsed_rgb = ""
    return parsed_color, parsed_rgb

def place_all(args):
    repeat=None
    if 'repeat.json' in os.listdir(os.path.join(args.root_dir, args.furnitures_image_folder)):
        repeat=json.load(open(os.path.join(args.root_dir, args.furnitures_image_folder, 'repeat.json'), 'r'))
        print('repeat.json exists! loaded.')
    furnitures_info=json.load(open(os.path.join(args.root_dir, args.furnitures_dict_file), 'r'))
    ############################# Subject to chanage ###################################
    if args.show_list:
        no_show_keys=[]
        for obj,info in furnitures_info.items():
            phrase=obj.split('.png')[0].split('_')[1]
            show=False
            for s in args.show_list:
                if s in phrase.lower(): 
                    show=True
                    break
            if not show: no_show_keys.append(obj)
        furnitures_info={k:v for k,v in furnitures_info.items() if k not in no_show_keys}
    
    furnitures_glb_dir=os.path.join(args.root_dir, args.furnitures_image_folder, "edify3D")
    # furnitures_info={k:v for k,v in furnitures_info.items() if k in os.listdir(furnitures_glb_dir)}
    furnitures_info={k:v for k,v in furnitures_info.items() if 'window' not in k.lower() and 'door' not in k.lower()}
    ############################# Subject to chanage ###################################
    no_show_keys=[]
    for obj,info in furnitures_info.items():
        img=Image.open(os.path.join(args.root_dir, args.furnitures_image_folder, 'images', obj+'.png'))
        if 'carpet' in obj or 'rug' in obj or 'mat' in obj:
            info['dim'][-1]=0.01
        if img.size[0]>700:
            no_show_keys.append(obj)
    furnitures_info={k:v for k,v in furnitures_info.items() if k not in no_show_keys}
    for _, new_dict in furnitures_info.items():
        w,h,l=new_dict['dim']
        bbox_w,bbox_h=inverse_scale_obj(w,h,int(new_dict['pose']))
        new_dict['bbox']=[new_dict['center'][0]-bbox_w/2,new_dict['center'][0]+bbox_w/2,\
                new_dict['center'][1]-bbox_h/2,new_dict['center'][1]+bbox_h/2,\
                new_dict['center'][2]-l/2, new_dict['center'][2]+l/2]
        new_dict['on_floor']=None
        new_dict['on_wall']=None
    if args.small_objects_dict_file is not None:
        small_objects_info=json.load(open(os.path.join(args.root_dir, args.small_objects_dict_file), 'r'))
        small_objects_glb_dir=os.path.join(args.root_dir, args.small_objects_image_folder, "edify3D")
        small_objects_info={k:v for k,v in small_objects_info.items() if k in os.listdir(small_objects_glb_dir) and 'cushion' not in k.lower()}
        for _, new_dict in small_objects_info.items():
            new_dict['asso_furniture']=None
            if new_dict['pose']%180==90:
                new_dict['bbox']=[new_dict['center'][0]-new_dict['dim'][1]/2,new_dict['center'][0]+new_dict['dim'][1]/2,\
                    new_dict['center'][1]-new_dict['dim'][0]/2,new_dict['center'][1]+new_dict['dim'][0]/2,\
                    new_dict['center'][2]-new_dict['dim'][2]/2, new_dict['center'][2]+new_dict['dim'][2]/2]
            else:
                new_dict['bbox']=[new_dict['center'][0]-new_dict['dim'][0]/2,new_dict['center'][0]+new_dict['dim'][0]/2,\
                        new_dict['center'][1]-new_dict['dim'][1]/2,new_dict['center'][1]+new_dict['dim'][1]/2,\
                        new_dict['center'][2]-new_dict['dim'][2]/2, new_dict['center'][2]+new_dict['dim'][2]/2]
        new_dict['on_floor']=None
        new_dict['on_wall']=None
    else:
        small_objects_info={}
    associate_furnitures_objects(furnitures_info, small_objects_info, args.floor_fall_threshold)
    
    

    furnitures_info, small_objects_info, furnitures_dist_name, small_objects_dist_name=\
        sort_fall_to_ground(furnitures_info, small_objects_info, args.floor_fall_threshold)
    furnitures_info, small_objects_info=push_to_wall(furnitures_info, small_objects_info, args.wall_push_threshold)
    print(furnitures_dist_name)
    for name, info in furnitures_info.items():
        print(name, info['on_floor'], info['on_wall'])
    for i in range(len(furnitures_dist_name)-1):
        # certain overlaps (e.g. desk and chair) are fine.
        if 'carpet' in furnitures_dist_name[i][1].lower() or 'rug' in furnitures_dist_name[i][1].lower()\
            or 'mat' in furnitures_dist_name[i][1].lower() or 'fence' in furnitures_dist_name[i][1].lower()\
            or 'curtain' in furnitures_dist_name[i][1].lower(): continue
        for j in range(i+1, len(furnitures_dist_name)):
            if 'carpet' in furnitures_dist_name[j][1].lower() or 'rug' in furnitures_dist_name[j][1].lower()\
                or 'mat' in furnitures_dist_name[j][1].lower() or 'fence' in furnitures_dist_name[i][1].lower()\
                or 'curtain' in furnitures_dist_name[j][1].lower(): continue
            if ('desk' in furnitures_dist_name[i][1].lower() and 'chair' in furnitures_dist_name[j][1].lower()) or \
                ('desk' in furnitures_dist_name[j][1].lower() and 'chair' in furnitures_dist_name[i][1].lower()):
                continue
            if ('table' in furnitures_dist_name[i][1].lower() and 'chair' in furnitures_dist_name[j][1].lower()) or \
                ('table' in furnitures_dist_name[j][1].lower() and 'chair' in furnitures_dist_name[i][1].lower()):
                continue
            check_intersect_and_move(furnitures_dist_name[i][1], furnitures_dist_name[j][1], furnitures_info, furnitures_info, small_objects_dict=small_objects_info)

    # json.dump(furnitures_info, open(os.path.join(args.root_dir, 'intersectMoved_'+args.furnitures_dict_file),'w'))
    # if args.small_objects_dict_file is not None:
    #     json.dump(small_objects_info, open(os.path.join(args.root_dir, 'intersectMoved_'+args.small_objects_dict_file),'w'))

    # for i in range(len(small_objects_dist_name)-1):
    #     for j in range(i+1, len(small_objects_dist_name)):
    #         check_intersect_and_move(small_objects_dist_name[i][1], small_objects_dist_name[j][1], small_objects_info, small_objects_info)

    # for i in range(len(furnitures_dist_name)):
    #     for j in range(len(small_objects_dist_name)):
    #         check_intersect_and_move(furnitures_dist_name[i][1], small_objects_dist_name[j][1], furnitures_info, small_objects_info)

    for target_name in furnitures_info: #os.listdir(furnitures_glb_dir):
        repeat_target_name=target_name
        # import pdb;pdb.set_trace()
        
        if repeat and repeat[target_name] is not None:
            repeat_target_name=repeat[target_name]
        print(repeat_target_name, target_name)
        # if target_name not in furnitures_info: continue
        # Import the file
        id_folder=[i for i in os.listdir(os.path.join(furnitures_glb_dir, repeat_target_name)) if ('response' not in i) and ('.DS_Store' not in i)]
        assert len(id_folder)==1
        id_folder=id_folder[0]
        bpy.ops.import_scene.gltf(filepath=os.path.join(furnitures_glb_dir, repeat_target_name, id_folder, 'mesh_0.glb'))
        for obj in bpy.context.selected_objects[:]:
            obj.name = target_name
            obj.hide_render = False
        obj.rotation_mode = 'XYZ'

        w,h,l=furnitures_info[target_name]['dim']
        x_center,y_center,z_center=furnitures_info[target_name]['center']
        z_rot_deg=furnitures_info[target_name]['pose']
        obj.dimensions=Vector((w, h, l))
        obj.location=Vector((x_center,y_center,z_center))
        obj.rotation_euler=Euler((0.0, 0.0, z_rot_deg*np.pi/180), 'XYZ')
    if small_objects_info !={}:
        for target_name in os.listdir(small_objects_glb_dir):
            if target_name not in small_objects_info: continue
            # Import the file
            id_folder=[i for i in os.listdir(os.path.join(small_objects_glb_dir, target_name)) if ('response' not in i) and ('.DS_Store' not in i)]
            assert len(id_folder)==1
            id_folder=id_folder[0]
            bpy.ops.import_scene.gltf(filepath=os.path.join(small_objects_glb_dir, target_name, id_folder, 'mesh_0.glb'))
            for obj in bpy.context.selected_objects[:]:
                obj.name = target_name
                obj.hide_render = False
            obj.rotation_mode = 'XYZ'

            w,h,l=small_objects_info[target_name]['dim']
            x_center,y_center,z_center=small_objects_info[target_name]['center']
            z_rot_deg=small_objects_info[target_name]['pose']
            obj.dimensions=Vector((w, h, l))
            obj.location=Vector((x_center,y_center,z_center))
            obj.rotation_euler=Euler((0.0, 0.0, z_rot_deg*np.pi/180), 'XYZ')

    return furnitures_info, small_objects_info

def create_colored_plane(name, color, size, center, rotation):
    """
    Create a plane in the Blender scene with a specific color, size, center position, and rotation.

    :param name: Name of the plane object
    :param color: Color of the plane (RGB tuple, e.g., (1.0, 0.5, 0.3))
    :param size: Size of the plane (tuple, e.g., (x_scale, y_scale) for non-uniform scaling)
    :param center: Center position of the plane (3D tuple, e.g., (0, 0, 0))
    :param rotation: Rotation in degrees for the plane (3D tuple, e.g., (45, 0, 90) for X, Y, Z axes)
    """
    
    # Create the plane
    bpy.ops.mesh.primitive_plane_add(size=1, location=center)
    plane = bpy.context.active_object
    plane.name = name
    # Apply non-uniform scaling
    plane.scale.x = size[0]  # Scale along the X-axis
    plane.scale.y = size[1]  # Scale along the Y-axis
    # Convert rotation from degrees to radians and apply
    plane.rotation_euler = (math.radians(rotation[0]), 
                            math.radians(rotation[1]), 
                            math.radians(rotation[2]))

    # Create a new material with the given color
    material = bpy.data.materials.new(name=f"{name}_Material")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)  # RGB + Alpha

    # Assign the material to the plane
    plane.data.materials.append(material)


def create_textured_plane(name, image_path, size, center, rotation):
    # Create the plane
    bpy.ops.mesh.primitive_plane_add(size=1, location=center)
    plane = bpy.context.active_object
    plane.name = name
    # Apply non-uniform scaling
    plane.scale.x = size[0]  # Scale along the X-axis
    plane.scale.y = size[1]  # Scale along the Y-axis
    # Convert rotation from degrees to radians and apply
    plane.rotation_euler = (math.radians(rotation[0]), 
                            math.radians(rotation[1]), 
                            math.radians(rotation[2]))

    # Create a new material with the given color
    material = bpy.data.materials.new(name=f"{name}_Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes

    # Clear existing nodes TODO: cons
    for node in nodes:
        nodes.remove(node)

    # Create nodes for the texture
    node_tex_image = nodes.new(type="ShaderNodeTexImage")
    node_tex_image.image = bpy.data.images.load(image_path)
    node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    node_output = nodes.new(type="ShaderNodeOutputMaterial")

    # Connect nodes
    material.node_tree.links.new(node_tex_image.outputs["Color"], node_bsdf.inputs["Base Color"])
    material.node_tree.links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    # Assign the material to the object
    if plane.data.materials:
        # Replace the existing material
        plane.data.materials[0] = material
    else:
        # Add the new material
        plane.data.materials.append(material)


def add_wall_floor(furnitures_info, small_objects_info, args):
    room_x_min,room_x_max,room_y_min,room_y_max,room_z_min,room_z_max=np.inf, -np.inf,np.inf,-np.inf,np.inf,-np.inf
    all_dict={**furnitures_info, **small_objects_info}
    wall_extra_height=args.wall_extra_height
    floor_extra_width=args.floor_extra_width
    # latest room limits
    for name, info in all_dict.items():
        x_min,x_max,y_min,y_max,z_min,z_max=info['bbox']
        room_x_min=min(room_x_min, x_min)
        room_y_min=min(room_y_min, y_min)
        room_z_min=min(room_z_min, z_min)
        room_x_max=max(room_x_max, x_max)
        room_y_max=max(room_y_max, y_max)
        room_z_max=max(room_z_max, z_max)
    
    center_y=(room_x_min,(room_y_max+room_y_min)/2-floor_extra_width,(room_z_max+room_z_min)/2+wall_extra_height/2)
    size_y=(room_z_max-room_z_min+wall_extra_height, room_y_max-room_y_min+2*floor_extra_width)
    
    center_x=((room_x_max+room_x_min)/2+floor_extra_width,room_y_max,(room_z_max+room_z_min)/2+wall_extra_height/2)
    size_x=(room_z_max-room_z_min+wall_extra_height, room_x_max-room_x_min+2*floor_extra_width)
    center_z=((room_x_max+room_x_min)/2+floor_extra_width,(room_y_max+room_y_min)/2-floor_extra_width,room_z_min)
    size_z=(room_x_max-room_x_min+2*floor_extra_width, room_y_max-room_y_min+2*floor_extra_width)
    if args.color_floor:
        floor_color, floor_rgb='',''#ask_wall_color(args.image_path)
        print('floor_color, floor_rgb', floor_color, floor_rgb)
        if floor_rgb=='':
            floor_rgb='(255,255,255)'
        floor_rgb=floor_rgb[1:-1].split(',')
        create_colored_plane('floor', (float(i)/255 for i in floor_rgb), size_z, center_z, (0,0,0))
    else:
        if 'floor.png' in os.listdir(args.root_dir):
            floorpaper_file=os.path.abspath(os.path.join(args.root_dir, 'floor.png'))
        else:
            floorpaper_file=os.path.abspath(os.path.join(args.root_dir, 'floor_chosen1.png'))
        create_textured_plane('floor', floorpaper_file, size_z, center_z, (0,0,0))
    
    if args.no_wall:
        center_z=((room_x_max+room_x_min)/2+floor_extra_width,(room_y_max+room_y_min)/2-floor_extra_width,room_z_min)
        return room_x_max-room_x_min+2*floor_extra_width, room_y_max-room_y_min+2*floor_extra_width, room_z_max-room_z_min+wall_extra_height,\
          center_z
    
    if args.color_wall:
        wall_color, wall_rgb='',''#ask_wall_color(args.image_path)
        print('wall_color, wall_rgb',wall_color, wall_rgb)
        if wall_rgb=='':
            wall_rgb='(255,255,255)'
        wall_rgb=wall_rgb[1:-1].split(',')
        create_colored_plane('y_parallel_wall', (float(i)/255 for i in wall_rgb), size_y, center_y, (0,90,0))
        create_colored_plane('x_parallel_wall', (float(i)/255 for i in wall_rgb), size_x, center_x, (90,90,0))
    else:
        if 'chosen.png' in os.listdir(args.root_dir):
            wallpaper_file=os.path.abspath(os.path.join(args.root_dir, 'chosen.png'))
        else:
            wallpaper_file=os.path.abspath(os.path.join(args.root_dir, 'chosen1.png'))
        create_textured_plane('y_parallel_wall', wallpaper_file, size_y, center_y, (0,90,0))
        create_textured_plane('y_parallel_wall', wallpaper_file, size_x, center_x, (90,90,0))


    return room_x_max-room_x_min+2*floor_extra_width, room_y_max-room_y_min+2*floor_extra_width, room_z_max-room_z_min+wall_extra_height,\
          center_z


def add_rotating_camera(w, h, l, center, frames=100):
    """
    Add a camera to the scene at 1.5 times the longest diagonal of a bounding box (w, h, l).
    The camera will rotate 360 degrees around the center in the specified number of frames.

    :param w: Width of the bounding box along the x-axis
    :param h: Height of the bounding box along the y-axis
    :param l: Length of the bounding box along the z-axis
    :param center: Center position of the bounding box (tuple, e.g., (x, y, z))
    :param frames: Number of frames for the full rotation (default is 200)
    """
    # Calculate the longest diagonal of the bounding box
    diagonal_length = math.sqrt((w/2)**2 + (h/2)**2 + l**2)
    camera_distance = 3 * diagonal_length
    camera_height = center[2] + 2 * l
    print('w,h,l,center', w,h,l,center)
    print('camera_height', camera_height)

    # Add a new camera to the scene
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "RotatingCamera"
    # Set the initial position of the camera
    camera.location = (center[0], center[1] - camera_distance, camera_height)
    
    # Make the camera face the center point
    direction = camera.location - mathutils.Vector(center)
    rot_quat = direction.to_track_quat('Z', 'Y')  # Face toward the center
    camera.rotation_euler = rot_quat.to_euler()

    # Set up animation: 360-degree rotation around the Z-axis
    for frame in range(frames + 1):
        bpy.context.scene.frame_set(frame)
        
        # Calculate the rotation angle for the current frame
        angle = (frame / frames) * 2 * math.pi  # Full 360 degrees in radians
        # Rotate camera around the center in the X-Y plane
        x = center[0] + camera_distance * math.cos(angle)
        y = center[1] + camera_distance * math.sin(angle)
        camera.location = (x, y, camera_height)
        
        # Keep the camera pointed at the center
        direction = camera.location - mathutils.Vector(center)
        camera.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()
        
        # Insert keyframe for location and rotation
        camera.keyframe_insert(data_path="location", index=-1)
        camera.keyframe_insert(data_path="rotation_euler", index=-1)

def add_camera(theta, c, dx,dy,center,ortho=False):
    diagonal_length = math.sqrt((dx/2)**2 + (dy/2)**2)
    d = c * diagonal_length
    # Calculate the center of the scene
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    # z_center = z  # Center elevation as provided by the user
    x_center,y_center,z_center=center

    # Calculate the scene diagonal direction in 2D (x, y)
    # dx = x_max - x_min
    # dy = y_max - y_min
    angle_diagonal = math.atan2(dx, dy)  # Angle of the diagonal in the xy-plane
    # import pdb;pdb.set_trace()

    # Calculate the camera position based on distance d and angle theta
    x_cam = x_center + d * math.sin(angle_diagonal)# * math.cos(math.radians(theta))
    y_cam = y_center - d * math.cos(angle_diagonal)# * math.cos(math.radians(theta))
    z_cam = z_center + d * math.sin(math.radians(theta))

    # Add a new camera to the scene
    cam_data = bpy.data.cameras.new(name="RotatingCamera")
    cam_object = bpy.data.objects.new("RotatingCamera", cam_data)
    bpy.context.collection.objects.link(cam_object)

    # Set the camera's position
    cam_object.location = (x_cam, y_cam, z_cam)

    # Point the camera to the scene center
    cam_object.rotation_mode = 'XYZ'
    cam_object.rotation_euler[0] = math.atan2(z_cam - z_center, d * math.cos(math.radians(theta)))  # Rotation along x-axis (pitch)
    cam_object.rotation_euler[2] = angle_diagonal  # Rotation along z-axis (yaw)

    # Make this camera the active camera
    # bpy.context.scene.camera = cam_object
    if ortho:
        cam_object.data.type='ORTHO'
        cam_object.data.ortho_scale=0.9*max(dx,dy)/0.615


def add_vertical_camera(c, dx,dy,center):
    # Calculate the center of the scene in the x, y plane
    # x_center = (x_min + x_max) / 2
    # y_center = (y_min + y_max) / 2
    x_center,y_center,z_min=center
    
    # Calculate the scene diagonal length in the x, y plane
    scene_diagonal = math.sqrt((dx/2)**2 + (dy/2)**2)
    
    # Calculate the camera's distance above the scene
    z_cam = z_min + c * scene_diagonal

    # Add a new camera to the scene
    cam_data = bpy.data.cameras.new(name="VerticalCamera")
    cam_object = bpy.data.objects.new("VerticalCamera", cam_data)
    bpy.context.collection.objects.link(cam_object)

    # Set the camera's position directly above the scene's center
    cam_object.location = (x_center, y_center, z_cam)

    # Rotate the camera to face downward (along the negative z-axis)
    cam_object.rotation_mode = 'XYZ'
    cam_object.rotation_euler = (math.radians(0), 0, math.radians(90))  # 90 degrees around the x-axis

    # Set this camera as the active camera
    # bpy.context.scene.camera = cam_object


def create_scene_rotation_animation(center, frames=100, start_angle=-45):
    """
    Rotates the entire scene 360 degrees around a specified center in 100 frames with a constant angular speed.
    
    :param center: The center point (x, y, z) around which the scene will rotate.
    :param frames: The number of frames over which the rotation will occur (default is 100).
    """
    # Set up the rotation angle per frame in degrees
    angle_per_frame = 360 / frames  # 3.6 degrees per frame for 100 frames
    
    # Add an empty object at the center point to serve as the rotation pivot
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=center)
    empty = bpy.context.active_object
    empty.name = "RotationPivot"
    
    # Parent all objects in the scene to the empty object
    for obj in bpy.context.scene.objects:
        if obj != empty and 'light' not in obj.name and 'Camera' not in obj.name:  # Skip the empty itself
            obj.select_set(True)
            bpy.context.view_layer.objects.active = empty
            bpy.ops.object.parent_set(type='OBJECT')
            obj.select_set(False)

    # Set keyframes for rotating the empty object
    for frame in range(frames + 1):
        bpy.context.scene.frame_set(frame)
        
        # Calculate rotation in radians for the current frame
        rotation_angle = math.radians(start_angle+angle_per_frame * frame)
        
        # Apply rotation around the Z-axis
        empty.rotation_euler[2] = rotation_angle
        empty.keyframe_insert(data_path="rotation_euler", index=2)

def render(target_cam, save_dir):
    # Set the frame to the first frame of the animation
    bpy.context.scene.frame_set(0)

    # Get the camera object (assuming there's only one camera in the scene)
    camera = None
    for obj in bpy.data.objects:
        if obj.name==target_cam:
            camera = obj
            break

    if camera is None:
        raise ValueError(f"No {target_cam} found in the scene.")

    # Set the active camera
    bpy.context.scene.camera = camera

    # Set the file format and output path for the render
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # You can change this to 'JPEG', etc.
    bpy.context.scene.render.filepath = save_dir

    # Render the first frame and save it
    bpy.ops.render.render(write_still=True)

    print(f"Rendered first frame saved to {save_dir}")

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, help='description for option1')
    parser.add_argument('--save_name', type=str, help='description for option1')
    parser.add_argument('--furnitures_image_folder', type=str, default='for_image_text23d/furnitures_merged_2_pix2gestalt/')
    parser.add_argument('--furnitures_dict_file', type=str, default='feat_sd_dino_1.2_furnitures_pred.json')
    parser.add_argument('--small_objects_image_folder', type=str, default=None)
    parser.add_argument('--wall_push_threshold', type=float, default=0.08, help='the thresholding distance for pushing objects to wall')
    parser.add_argument('--floor_fall_threshold', type=float, default=0.08, help='the thresholding distance for pushing objects to the floor')
    parser.add_argument('--image_path', type=str, help='the original input scene image')
    parser.add_argument('--color_wall', action="store_true", help='where ask GPT for the walls color during runtime')
    parser.add_argument('--color_floor', action="store_true", help='where ask GPT for the floor color during runtime')
    # 'for_image_text23d/small_objects_cropped_seg_rgb/'
    parser.add_argument('--small_objects_dict_file', type=str, default=None) #'all_small_objects_pred.json'
    parser.add_argument('--show_only', type=str, default=None, help='limit what to show by a list of furniture names. For LayoutGPT comparison. Should be either livingroom or bedroom.')
    parser.add_argument('--no_wall',  action="store_true")
    parser.add_argument('--no_cam',  action="store_true")
    parser.add_argument('--floor_extra_width', type=float, default=0.02)
    parser.add_argument('--wall_extra_height', type=float, default=0.02)
    args = parser.parse_args()

    if args.show_only=='livingroom':
        args.show_list=[
        "armchair", "bookshelf", "cabinet", "sofa", "chair", "table","desk",  "shelf", "stool", "tv stand", "wardrobe", "ottomon","couch",\
        'shelves', "stand"]
        print('args.show_only==livingroom. Only show furnitures in', args.show_list)
    elif args.show_only=='bedroom':
        args.show_list=['bed','desk','table','chair','wardrobe','shelf','shelves','bookshelf','nightstand','ottoman','armchair','cabinet',\
            'sofa','dresser']
        print('args.show_only==bedroom. Only show furnitures in', args.show_list)
    else:
        args.show_list=[]

    furnitures_info, small_objects_info=place_all(args)
    f=open(os.path.join(args.root_dir, args.save_name+'_furnitures.json'), 'w')
    json.dump(furnitures_info, f)
    if args.small_objects_dict_file:
        f=open(os.path.join(args.root_dir, args.save_name+'_small_objects.json'), 'w')
        json.dump(small_objects_info, f)


    w,h,l,center_floor=add_wall_floor(furnitures_info, small_objects_info, args)
    create_scene_rotation_animation(center_floor, start_angle=0)
    if not args.no_cam:
        ########## If you want to add a fixed camera
        # Add a new camera to the scene
        # bpy.ops.object.camera_add()
        # camera = bpy.context.active_object
        # camera.name = "RotatingCamera"
        # camera.rotation_euler=Euler((66*np.pi/180, 0.0, 1.2*np.pi/180), 'XYZ')
        # camera.location = (-0.108, -1.25, 0.48)
        ##########

        # add_rotating_camera(w, h, l, center_floor, frames=-1)
        os.makedirs(os.path.join(args.root_dir,'render'), exist_ok=True)
        # add_camera(55, 2.8, w,h,center_floor)
        add_camera(54, 2.8, w,h,center_floor,ortho=True)
        add_vertical_camera(3.3, w, h, center_floor)
        
        render("RotatingCamera", os.path.join(args.root_dir,f'render/{args.save_name}_rotCam.png'))
        render("VerticalCamera", os.path.join(args.root_dir,f'render/{args.save_name}_vertCam.png'))
    
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.root_dir, args.save_name+ '.blend'))