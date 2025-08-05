#use env One2345Again
import os
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, '/home/zg45/pix2gestalt/pix2gestalt')
from inference import run_inference, load_model_from_config, run_pix2gestalt
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

ckpt="../pix2gestalt/pix2gestalt/ckpt/epoch=000005.ckpt"
config="../pix2gestalt/pix2gestalt/configs/sd-finetune-pix2gestalt-c_concat-256.yaml"
device_idx =0

device = f"cuda:{device_idx}"
config = OmegaConf.load(config)


model = load_model_from_config(config, ckpt, device)

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='description for option1')
    args = parser.parse_args()

    image_dir = args.input_folder
    rgb_folder = 'furnitures_cropped_seg_rgb_together_merged_2'
    mask_folder = 'furnitures_cropped_seg_together_merged_2'
    os.makedirs(os.path.join(image_dir, 'pix2gestalt_inpaint'), exist_ok=True)
    for image_name in os.listdir(os.path.join(image_dir, rgb_folder)):
        if '.png' not in image_name: continue
        input_im=Image.open(os.path.join(image_dir, rgb_folder,image_name))
        w,h=input_im.size
        input_im=np.array(input_im.resize((256,256)))
        visible_mask=np.array(Image.open(os.path.join(image_dir,mask_folder,image_name)).convert('RGB').resize((256,256)))
        output=run_pix2gestalt(
            model,
            device,
            input_im,
            visible_mask)
        output_np=np.asarray(output).astype('float')
        mean_output= np.mean(np.asarray(output_np), axis=0)
        print(image_name, abs(output_np-mean_output).mean())
        for i in range(len(output)):
            out_image=Image.fromarray(output[i]).resize((w,h))
            out_image.save(os.path.join(image_dir, 'pix2gestalt_inpaint',str(i)+'_'+image_name))
            plt.imshow(out_image)
            plt.show()