import time
start1=time.time()
from PIL import Image
import requests
import numpy as np
import json
import argparse
import sys
import os
import cv2
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random


SEED=42

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(SEED)

focalLength =1120
centerX = 512
centerY = 512
scalingFactor = 100 #doesnt influence the relative size between x,y,z
#cam.matrix_world.normalized()
extrinsics=np.array([[1.0, 0.0, 0.0, 0.0],\
                     [0.0, -0.7071068286895752, 0.7071067690849304, 1],\
                     [0.0, 0.7071068286895752, 0.7071067690849304, 1],\
                     [0.0, 0.0, 0.0, 1.0]]) 

x_angle=35.3644*np.pi/180
y_angle=45*np.pi/180
x_rot=np.array([[1,0,0],[0,np.cos(x_angle),-np.sin(x_angle)],[0,np.sin(x_angle),np.cos(x_angle)]])
y_rot=np.array([[np.cos(y_angle),0,np.sin(y_angle)],[0,1,0],[-np.sin(y_angle),0,np.cos(y_angle)]])
world_to_cam_rot=x_rot@y_rot
cam_to_world_rot=np.linalg.inv(world_to_cam_rot)

cam_K=np.array([[1120.0, 0.0, 512.0],\
                     [0.0, 1120.0, 512.0],\
                     [0.0, 0.0, 1.0]])

 
# The first parameter is the original image, 
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 


def generate_pointcloud_method4(rgb_file,depth_file,ply_file,masks,mask_colors, mask_files, z_scale):
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file).convert('L')
    # if rgb.size != depth.size:
    #     raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format",depth.mode)
    points = []
    points_for_dims={k:[] for k in range(len(masks))}
    data={}
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z=(depth.getpixel((u,v)))/z_scale
            if Z>100: continue
            if Z==0: continue
            X = (u - centerX)/ focalLength
            Y = (v - centerY)/ focalLength
            transformed=cam_to_world_rot@np.array([[X],[Y],[Z]])
            X,Y,Z=transformed[0][0],transformed[1][0],transformed[2][0]
            in_mask=False
            for j in range(len(masks)):
                mask_erosion=masks[j]
                if mask_erosion[v][u]==255: #h first w second
                    points_for_dims[j].append(np.array([X,Y,Z,color[0],color[1],color[2]]))
                    in_mask=True
                    break
            if not in_mask:
                points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))

    for i,points_for_dim in points_for_dims.items():
        try:
            points_for_dim=np.stack(points_for_dim)
            nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(points_for_dim[:,:3])
            distances, indices = nbrs.kneighbors(points_for_dim[:,:3])
            if points_for_dim.shape[0]>500:
                filtered_points_for_dim=points_for_dim[np.where(distances[:,-1]<0.0018)][:,:3]
            else:
                filtered_points_for_dim=points_for_dim.copy()
            print(i,'before filtering', points_for_dim.shape, 'filtered shape', filtered_points_for_dim.shape)
            count=0
            filtered_points_for_dim_dict={}
            for X,Y,Z in filtered_points_for_dim:
                filtered_points_for_dim_dict['_'.join([str(X),str(Y),str(Z)])]=1
            for X,Y,Z,r,g,b in points_for_dim:
                if '_'.join([str(X),str(Y),str(Z)]) in filtered_points_for_dim_dict:
                    points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
                    count+=1
                else:
                    points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,r,g,b))
            print(i,'count', count)
        
            x_max,y_max,z_max=filtered_points_for_dim[:,0].max(),filtered_points_for_dim[:,1].max(),filtered_points_for_dim[:,2].max()
            x_min,y_min,z_min=filtered_points_for_dim[:,0].min(),filtered_points_for_dim[:,1].min(),filtered_points_for_dim[:,2].min()
        except:
            continue
        for X in np.linspace(x_min,x_max,num=int((x_max-x_min)/0.01)):
            points.append("%f %f %f %d %d %d 0\n"%(X, y_min, z_min, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(X, y_min, z_max, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(X, y_max, z_min, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(X, y_max, z_max, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
        for Y in np.linspace(y_min, y_max, num=int((y_max-y_min)/0.01)):
            points.append("%f %f %f %d %d %d 0\n"%(x_min, Y, z_min, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_min, Y, z_max, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_max, Y, z_min, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_max, Y, z_max, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
        for Z in np.linspace(z_min, z_max, num=int((z_max-z_min)/0.01)):
            points.append("%f %f %f %d %d %d 0\n"%(x_min, y_min, Z, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_min, y_max, Z, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_max, y_min, Z, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
            points.append("%f %f %f %d %d %d 0\n"%(x_max, y_max, Z, mask_colors[i][0], mask_colors[i][1], mask_colors[i][2]))
        w,h,l=x_max-x_min, y_max-y_min, z_max-z_min
        x_center,y_center,z_center=(x_max+x_min)/2, (y_max+y_min)/2, (z_max+z_min)/2
        # print('before', x_center,y_center,z_center)
        # transformed=world_to_cam_rot@np.array([[x_center],[y_center],[z_center]])
        # x_center,y_center,z_center=transformed[0][0],transformed[1][0],transformed[2][0]
        # print('after', x_center,y_center,z_center)
        key= mask_files[i].rstrip(os.sep)
        data[os.path.basename(key)]=[w,l,h,x_center,z_center,-y_center]

    file = open(ply_file,"w")
    file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        property uchar alpha
        end_header
        %s
        '''%(len(points),"".join(points)))
    file.close()
    with open(ply_file.replace('.ply', '.json'),'w') as f:
        json.dump(data, f, indent=4)
    return points_for_dim

time1=time.time()-start1
if __name__=="__main__": 
    start2=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--erosion_kernel', type=int, default=3, help='kernel to erode the edges of point clouds when estimating 3D bbox')
    parser.add_argument('--input_image', type=str, help='description for option1')
    parser.add_argument('--output_name', type=str, help='description for option1')
    parser.add_argument('--mask_folders', type=str, help='description for option1')
    parser.add_argument('--scale', type=int, default=300, help='scale to divide z coordinate')
    args = parser.parse_args()
    
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",device='cuda:0')
    image = Image.open(args.input_image)
    depth = pipe(image)["depth"]
    depth=255-np.array(depth)

    input_dir=os.path.dirname(args.input_image)
    image_file=os.path.basename(args.input_image)
    depth_image=os.path.join(input_dir, 'est_depth_'+image_file)
    Image.fromarray(depth).save(depth_image)

    
    input_filename=os.path.basename(args.input_image).split('.')[0]
    mask_files=[]
    for mask_folder in args.mask_folders.split(','):
        mask_dir=os.path.join(input_dir, mask_folder) # input_filename+'_mask')
        mask_files+=sorted([os.path.join(mask_dir, i) for i in os.listdir(mask_dir) if 'png' in i])

    mask_colors=np.random.randint(0,high=255,size=(50,3))
    masks=[]
    if args.erosion_kernel>0:
        kernel = np.ones((args.erosion_kernel, args.erosion_kernel), np.uint8) 
    for i in range(len(mask_files)):
        print(mask_files[i])
        mask=cv2.imread(mask_files[i], 0)
        if args.erosion_kernel>0:
            mask_erosion = cv2.erode(mask, kernel, iterations=1) 
            masks.append(mask_erosion)
        else:
            masks.append(mask)

    generate_pointcloud_method4(args.input_image, depth_image, os.path.join(input_dir,args.output_name), masks, mask_colors, mask_files, args.scale)
    
    time2=time.time()-start2
    with open(os.path.join(input_dir,'step3_depth_bbox_time_'+args.mask_folders+'.txt'), 'w') as f:
        f.write(str(time1+time2))
    print('time1', time1, 'time2', time2, time1+time2)
