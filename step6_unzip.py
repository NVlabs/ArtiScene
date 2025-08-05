import os
import argparse
import zipfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='root folder that contains the generated 3D assests, should be $SCENE_NAME')
    args = parser.parse_args()
    
    edify_folder=os.path.join(args.folder, 'for_image_text23d/furnitures_merged_2_pix2gestalt/edify3D')
    os.makedirs(edify_folder+'_zipfile', exist_ok=True)
    for f in os.listdir(edify_folder):
        if 'zip' not in f: continue
        folder_name=f.split('.zip')[0]
        os.makedirs(os.path.join(edify_folder, folder_name), exist_ok=True)
        with zipfile.ZipFile(os.path.join(edify_folder,f), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(edify_folder, folder_name)) 
        os.rename(os.path.join(edify_folder,f),os.path.join(edify_folder+'_zipfile',f))
    
    edify_folder=os.path.join(args.folder, 'for_image_text23d/small_objects_merged_12/edify3D')
    os.makedirs(edify_folder+'_zipfile', exist_ok=True)
    for f in os.listdir(edify_folder):
        if 'zip' not in f: continue
        folder_name=f.split('.zip')[0]
        os.makedirs(os.path.join(edify_folder, folder_name), exist_ok=True)
        with zipfile.ZipFile(os.path.join(edify_folder,f), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(edify_folder, folder_name)) 
        os.rename(os.path.join(edify_folder,f),os.path.join(edify_folder+'_zipfile',f))