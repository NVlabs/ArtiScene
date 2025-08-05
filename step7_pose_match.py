import time
start1=time.time()
import os
import argparse
import numpy as np
import bpy
from mathutils import Euler, Vector
import json
from PIL import Image
import OpenEXR as exr
from math import radians
import Imath
import sys
import torch
from transformers import AutoImageProcessor, AutoModel, Mask2FormerModel, Dinov2Model
import torch.nn as nn
sys.path.insert(0, '../sd-dino/')
from demo_vis_features import compute_pair_feature
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

def scale_obj(w,h,degree):
    if degree %180==0:
        return w,h
    elif degree %180==90:
        return h,w
    elif (degree %180==45) or (degree %180==135):
        min_side=min(w,h)
        return min_side*np.cos(45*np.pi/180), min_side*np.cos(45*np.pi/180)

light_data = bpy.data.lights.new(name="light1", type='AREA')
light_data.energy = 4500
# create new object with our light datablock
light_object = bpy.data.objects.new(name="light1", object_data=light_data)
# link light object
bpy.context.collection.objects.link(light_object)
light_object.location=Vector((-1.736,0.4,10))
light_object.rotation_mode = 'XYZ'
light_object.rotation_euler=Euler((radians(2.12), radians(-11), radians(-8)), 'XYZ')

light_data = bpy.data.lights.new(name="light2", type='AREA')
light_data.energy = 4500
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

def crop_by_mask(image, mask, crop_size):
    h,w=mask.shape
    mask_x_min, mask_x_max=np.nonzero(mask)[1].min(), np.nonzero(mask)[1].max()
    mask_y_min, mask_y_max=np.nonzero(mask)[0].min(), np.nonzero(mask)[0].max()
    mask_x_center, mask_y_center=(mask_x_min+ mask_x_max)/2, (mask_y_min+ mask_y_max)/2
    mask_bbox_w, mask_bbox_h=mask_x_max-mask_x_min, mask_y_max-mask_y_min
    bbox_x_min=int(max(mask_x_center-crop_size*mask_bbox_w//2, 0))
    bbox_y_min=int(max(mask_y_center-crop_size*mask_bbox_h//2, 0))
    bbox_x_max=int(min(mask_x_center+crop_size*mask_bbox_w//2, w))
    bbox_y_max=int(min(mask_y_center+crop_size*mask_bbox_h//2, h))

    mask_cropped=mask[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max].copy()
    image_cropped=image[bbox_y_min:bbox_y_max, bbox_x_min:bbox_x_max].copy()
    mult=mask_cropped[:,:,None]*image_cropped
    return mask_cropped,image_cropped, mult
    

def compare_closest(model, processor, est_path, mask_path, render_path, crop_size):
    os.makedirs(os.path.join(render_path, 'debug'), exist_ok=True)
    device=model.device
    gt_image = np.array(Image.open(est_path)).astype('float')
    mask=np.array(Image.open(mask_path)).astype('float')
    mask=mask/255
    if crop_size>0:
        mask_cropped,image_cropped, gt_image = crop_by_mask(gt_image, mask, crop_size)
        mask_cropped_image=(255*mask_cropped).astype('uint8')
        Image.fromarray(mask_cropped_image).save(os.path.join(render_path, 'debug', str(crop_size)+'_alpha_compare_cropped.png'))

    image1=Image.fromarray(gt_image.astype('uint8'))
    image1.save(os.path.join(render_path, 'debug', str(crop_size)+'_compare_cropped.png'))

    closest_match, closest_diff=None, -100000
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        if hasattr(outputs1, 'transformer_decoder_last_hidden_state'):
            image_features1 = outputs1.transformer_decoder_last_hidden_state
        elif hasattr(outputs1, 'last_hidden_state'):
            image_features1 = outputs1.last_hidden_state
        #image_features1 before torch.Size([1, 257, 768])
        image_features1 = image_features1.mean(dim=1)

    all_results=[]
    for i in sorted(os.listdir(render_path)):
        if '.png' not in i:continue
        # print(i)
        rendered_image = np.array(Image.open(os.path.join(render_path,i)).convert('RGB'))
        if crop_size>0:
            exrfile = exr.InputFile(os.path.join(render_path,i.replace('.png','.exr')))
            raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
            height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
            width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
            depth_map = np.reshape(depth_vector, (height, width))
            depth_map=np.array(depth_map)
            depth_map[depth_map>500]=0
            depth_map[depth_map!=0]=1.

            mask_cropped,image_cropped, rendered_image = crop_by_mask(rendered_image, depth_map, crop_size)
            mask_cropped_image=(255*mask_cropped).astype('uint8')
            Image.fromarray(mask_cropped_image.astype('uint8')).save(os.path.join(render_path, 'debug', str(crop_size)+'_alpha_cropped_'+i))
        
        image2=Image.fromarray(rendered_image.astype('uint8'))
        image2.save(os.path.join(render_path, 'debug',str(crop_size)+'_cropped_'+i))
        
        with torch.no_grad():
            inputs2 = processor(images=image2, return_tensors="pt").to(device)
            outputs2 = model(**inputs2)
            if hasattr(outputs2, 'transformer_decoder_last_hidden_state'):
                image_features2 = outputs2.transformer_decoder_last_hidden_state
            elif hasattr(outputs2, 'last_hidden_state'):
                image_features2 = outputs2.last_hidden_state
            image_features2 = image_features2.mean(dim=1)
        
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(image_features1[0],image_features2[0]).item()
        sim = (sim+1)/2
        print(i, 'Similarity:', sim)
        all_results.append((int(i.split('.png')[0]), -sim))
        if sim>closest_diff:
            closest_diff=sim
            closest_match=int(i.split('.png')[0])
    return closest_diff, closest_match, all_results

def match_one_furniture(args, target_name, repeat_name):
    root_dir=args.root_dir
    glb_dir=os.path.join(root_dir, args.image_folder, "edify3D")
    dim_center=json.load(open(os.path.join(root_dir, args.object_type+'_erode_5.json'), 'r'))
    if target_name+'.png' not in dim_center: return None
    w,h,l,x_center,y_center,z_center=dim_center[target_name+'.png']
    est_depth_file='est_depth_inpainted_together_2.png' if args.object_type=='furnitures' else 'est_depth_input.png'
    mask_folder=args.mask_folder #'furnitures_seg_cleaneddetect' if args.object_type=='furnitures' else 'small_objects_seg'
    # for furniture_folder in os.listdir(glb_dir):
        # Import the file
    id_folder=[i for i in os.listdir(os.path.join(glb_dir, repeat_name)) if ('response' not in i) and ('.DS_Store' not in i)]
    assert len(id_folder)==1
    id_folder=id_folder[0]
    # print(os.path.join(glb_dir, target_name, id_folder, 'mesh_0.glb'))
    bpy.ops.import_scene.gltf(filepath=os.path.join(glb_dir, repeat_name, id_folder, 'mesh_0.glb'))
    print(target_name, repeat_name, h, w)
    # Set the name of each imported object to the file name
    for obj in bpy.context.selected_objects[:]:
        obj.name = target_name

    for o in bpy.data.objects:
        if o.name==target_name:
            o.hide_render = False
            obj=bpy.data.objects[target_name]
        elif o.name in ['light1', 'light2']:
            o.hide_render = False
        else:
            o.hide_render = True
    
    if w>h:
        short_side, long_size=h, w
    else:
        short_side, long_size=w, h 
    if long_size/short_side <1.2: #still close to a square
        degrees=[0,45,90,135,180,225,270,315]
    else:
        degrees=[0,90,180,270]
    new_dict={}
    new_dict['center']=[x_center,y_center,z_center]

    obj.rotation_mode = 'XYZ'
    for rot_deg in degrees:
        scale_w, scale_h=scale_obj(w,h,rot_deg)
        obj.dimensions=Vector((scale_w, scale_h, l))
        obj.location=Vector((x_center,y_center,z_center))
        obj.rotation_euler=Euler((0.0, 0.0, rot_deg*np.pi/180), 'XYZ')
        depth_file_output.base_path =os.path.join(root_dir, str(rot_deg))
        rgb_file_output.base_path =os.path.join(root_dir, str(rot_deg))

        # redirect output to log file
        logfile = 'blender_render.log'
        open(logfile, 'a').close()
        old = os.dup(sys.stdout.fileno())
        sys.stdout.flush()
        os.close(sys.stdout.fileno())
        fd = os.open(logfile, os.O_WRONLY)

        # do the rendering
        bpy.ops.render.render(write_still=True)

        # disable output redirection
        os.close(fd)
        os.dup(old)
        os.close(old)

    exr_dir=os.path.join(root_dir, 'all_rots_depth', target_name)
    os.makedirs(exr_dir, exist_ok=True)
    for i in degrees:
        os.rename(os.path.join(root_dir, str(i), 'Image0001.exr'), os.path.join(exr_dir, str(i)+'.exr'))
        os.rename(os.path.join(root_dir, str(i), 'Image0001.png'), os.path.join(exr_dir, str(i)+'.png'))


    ######## match_by_depth
    if args.match_by =='depth':
        depth_scale=200
        depth_pred=np.array(Image.open(os.path.join(root_dir, est_depth_file))).astype('float')
        depth_pred=depth_pred/depth_scale
        mask=np.array(Image.open(os.path.join(root_dir, mask_folder, target_name+'.png'))).astype('float')
        mask=mask/255
        masked_depth=mask*depth_pred
        x_min, x_max=np.nonzero(masked_depth)[1].min(), np.nonzero(masked_depth)[1].max()
        y_min, y_max=np.nonzero(masked_depth)[0].min(), np.nonzero(masked_depth)[0].max()
        depth_bbox_w, depth_bbox_h=x_max-x_min, y_max-y_min
        
        closest_match, closest_diff=None, 100000
        all_results=[]
        for i in sorted(os.listdir(exr_dir)):
            if '.exr' not in i:continue
            exrfile = exr.InputFile(os.path.join(exr_dir,i))
            raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
            height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
            width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
            depth_map = np.reshape(depth_vector, (height, width))
            depth_map=np.array(depth_map)
            depth_map[depth_map>500]=0

            x_min, x_max=np.nonzero(depth_map)[1].min(), np.nonzero(depth_map)[1].max()
            y_min, y_max=np.nonzero(depth_map)[0].min(), np.nonzero(depth_map)[0].max()
            bbox_w, bbox_h=x_max-x_min, y_max-y_min
            ratio=bbox_w/depth_bbox_w
            bbox_center_x, bbox_center_y=(x_max+x_min)/2, (y_max+y_min)/2
            scaled_depth=np.zeros_like(depth_map)
            for k in range(y_min, y_max):
                for j in range(x_min, x_max):
                    source_coord_x=np.clip(int(bbox_center_x+ratio*(j-bbox_center_x)), 0, width)
                    source_coord_y=np.clip(int(bbox_center_y+ratio*(k-bbox_center_y)), 0, height)
                    scaled_depth[k,j]=depth_map[source_coord_y, source_coord_x]
            
            diff = abs(scaled_depth-masked_depth).sum()
            print(i, diff)
            all_results.append((int(i.split('.exr')[0]), diff))
            if diff<closest_diff:
                closest_diff=diff
                closest_match=int(i.split('.exr')[0])

    ########
    ######## match_by_feature
    elif args.match_by =='feat':
        device = torch.device('cuda:1' if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = Dinov2Model.from_pretrained('facebook/dinov2-base').to(device)

        est_path=args.image_path
        mask_path=os.path.join(root_dir, mask_folder, target_name+'.png')
        render_path=exr_dir
        closest_diff, closest_match, feat_all_results=compare_closest(model, processor, est_path, mask_path, render_path, args.crop_size)

        # sorted_feat_all_results=[i for i in sorted(feat_all_results, key=lambda x:x[1])]
        # sorted_all_results=[i for i in sorted(all_results, key=lambda x:x[1])]
        # rank={}
        # for r in range(len(sorted_feat_all_results)):
        #     rank[sorted_feat_all_results[r][0]]=r
        # for r in range(len(sorted_all_results)):
        #     rank[sorted_all_results[r][0]]+=1.2*r

        # closest_match=sorted(rank, key=lambda x: rank[x])
        
    elif args.match_by=='feat_mask2former':
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        processor = AutoImageProcessor.from_pretrained('facebook/mask2former-swin-small-coco-instance')
        model = Mask2FormerModel.from_pretrained("facebook/mask2former-swin-small-coco-instance").to(device)

        est_path=args.image_path
        mask_path=os.path.join(root_dir, mask_folder, target_name+'.png')
        render_path=exr_dir
        closest_diff, closest_match, _=compare_closest(model, processor, est_path, mask_path, render_path, args.crop_size)

    elif args.match_by=='feat_sd_dino':
        est_path=args.image_path
        mask_path=os.path.join(root_dir, mask_folder, target_name+'.png')
        render_path=exr_dir
        crop_size=args.crop_size
        
        os.makedirs(os.path.join(render_path, 'debug'), exist_ok=True)
        gt_image = np.array(Image.open(est_path)).astype('float')
        mask=np.array(Image.open(mask_path)).astype('float')
        mask=mask/255
        if crop_size>0:
            mask_cropped,image_cropped, gt_image = crop_by_mask(gt_image, mask, crop_size)
            mask_cropped_image=(255*mask_cropped).astype('uint8')
            Image.fromarray(mask_cropped_image).save(os.path.join(render_path, 'debug', str(crop_size)+'_alpha_compare_cropped.png'))
    
        image1=Image.fromarray(gt_image.astype('uint8'))
        image1.save(os.path.join(render_path, 'debug', str(crop_size)+'_compare_cropped.png'))
        # image2=Image.fromarray(image_cropped.astype('uint8'))
        # image2.save(os.path.join(render_path, 'debug', str(crop_size)+'_compare_nomask_cropped.png'))
    
        closest_match, closest_diff=None, 10000000
        all_results=[]
        for i in sorted(os.listdir(render_path)):
            if '.png' not in i: continue
            # print(i)
            rendered_image = np.array(Image.open(os.path.join(render_path,i)).convert('RGB'))
            if crop_size>0:
                exrfile = exr.InputFile(os.path.join(render_path,i.replace('.png','.exr')))
                raw_bytes = exrfile.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
                depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
                height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
                width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
                depth_map = np.reshape(depth_vector, (height, width))
                depth_map=np.array(depth_map)
                depth_map[depth_map>500]=0
                depth_map[depth_map!=0]=1.
    
                mask_cropped,image_cropped, rendered_image = crop_by_mask(rendered_image, depth_map, crop_size)
                mask_cropped_image=(255*mask_cropped).astype('uint8')
                Image.fromarray(mask_cropped_image.astype('uint8')).save(os.path.join(render_path, 'debug', str(crop_size)+'_alpha_cropped_'+i))
            
            image2=Image.fromarray(rendered_image.astype('uint8'))
            image2.save(os.path.join(render_path, 'debug',str(crop_size)+'_cropped_'+i))

            src_img_path =os.path.join(render_path, 'debug',str(crop_size)+'_cropped_'+i)
            trg_img_path = os.path.join(render_path, 'debug', str(crop_size)+'_compare_cropped.png')
            categories = [[''], ['']]
            files = [src_img_path, trg_img_path]
            # save_path = './results_vis' + f'/{trg_img_path.split("/")[-1].split(".")[0]}_{src_img_path.split("/")[-1].split(".")[0]}'
            result = compute_pair_feature(files, categories)
            diff= np.sqrt((result[0][0]-result[0][1])**2).sum().item()
            if diff<closest_diff:
                closest_diff=diff
                closest_match=int(i.split('.png')[0])
                
    ########
    closest_w, closest_h=scale_obj(w,h,closest_match)
    new_dict['pose']=closest_match
    new_dict['closest_diff']=closest_diff
    new_dict['dim']=[closest_w, closest_h, l]
    return new_dict

time1=time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='description for option1')
    parser.add_argument('--image_folder', type=str, default='for_image_text23d/furnitures_cropped_seg_rgb_cleaneddetect_pix2gestalt/')
    parser.add_argument('--mask_folder', type=str, default='small_objects_cropped_seg_rgb')
    parser.add_argument('--match_by', type=str, default='feat')
    parser.add_argument('--crop_size', type=float, default=1.) #-1 means no crop. Default to 1.2, as I found the background content influences feature matching score.
    args = parser.parse_args()

    args.object_type='furnitures' if 'furnitures' in args.image_folder else 'small_objects'
    args.root_dir=os.path.dirname(args.image_path)
    all_dict={}
    repeat=None
    if 'repeat.json' in os.listdir(os.path.join(args.root_dir, args.image_folder)):
        repeat=json.load(open(os.path.join(args.root_dir, args.image_folder, 'repeat.json'), 'r'))
        print('repeat.json exists! loaded')
    for furniture_image in sorted(os.listdir(os.path.join(args.root_dir, args.image_folder, 'images'))):
        if '.png' not in furniture_image: continue
        furniture_name=furniture_image.split('.png')[0]
        repeat_furniture_name=furniture_name
        if repeat and repeat[furniture_name] is not None:
            repeat_furniture_name=repeat[furniture_name]
            print(furniture_name, 'repeated! use', repeat_furniture_name)
        try:
            furniture_dict=match_one_furniture(args, furniture_name, repeat_furniture_name)
        except Exception as e:
            print('Exception', furniture_image, e)
            furniture_dict=None
        if furniture_dict is not None:
            all_dict[furniture_name]=furniture_dict
        print('furniture_image, furniture_dict', furniture_image, furniture_dict)

    with open(os.path.join(args.root_dir, '_'.join([args.match_by, str(args.crop_size), args.object_type+'_pred.json'])), 'w') as f:
        json.dump(all_dict, f, indent = 4)

    time2=time.time()-start2
    with open(os.path.join(args.root_dir,'step7_pose_match_time_'+args.object_type+'.txt'), 'w') as f:
        f.write(str(time1+time2))
    print('time1', time1, 'time2', time2, time1+time2)
