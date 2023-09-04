# sudo apt-get install openslide-tools
# sudo apt-get install python-openslide
# pip install openslide-python
import argparse
import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

def main_worker(args):
    window_size = args.window_size
    gray_filter_tol = args.gray_filter_tol
    tissue_ratio_threshold = args.tissue_ratio_threshold
    target_mpp = args.target_mpp # for 20X, 0.25 for 40X

    output_path = args.output_path + '/' + args.hospital
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wsi_folder = args.wsi_folder + '/' + args.hospital
    wsi_files = os.listdir(wsi_folder)
    wsi_files.sort()
    
    for i, image_name in enumerate(wsi_files):
        output_subfolder = output_path + '/' + image_name[:-4]
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        wsi_file = wsi_folder + '/' + image_name
        print(wsi_file)
        img = openslide.OpenSlide(wsi_file)
        prop = img.properties
        try:
            width = int(prop['openslide.bounds-width'])
            height = int(prop['openslide.bounds-height'])
        except:
            width, height = img.level_dimensions[0]

        mpp_x = float(prop['openslide.mpp-x'])
        mpp_y = float(prop['openslide.mpp-y'])

        mpp_x_ratio = target_mpp / mpp_x
        mpp_y_ratio = target_mpp / mpp_y

        adjusted_pixel_x = int(window_size * mpp_x_ratio)
        adjusted_pixel_y = int(window_size * mpp_y_ratio)

        increment_x = int(np.ceil(width / adjusted_pixel_x))
        increment_y = int(np.ceil(height / adjusted_pixel_y))
        print("Process:", len(wsi_files), i, "start converting", image_name[:-4], "\t width: ", width, "\t height:", height)

        for incre_x in range(increment_x):
            for incre_y in range(increment_y):
                begin_x = adjusted_pixel_x * incre_x
                end_x = min(width, begin_x + adjusted_pixel_x)
                begin_y = adjusted_pixel_y * incre_y
                end_y = min(height, begin_y + adjusted_pixel_y)
                patch_width = end_x - begin_x
                patch_height = end_y - begin_y
                
                if patch_width==adjusted_pixel_x & patch_height==adjusted_pixel_y:
                    patch = img.read_region((begin_x, begin_y), 0, (patch_width, patch_height))
                    patch.load()
                    patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
                    patch_rgb.paste(patch, mask=patch.split()[3])
                    patch_rgb = patch_rgb.resize((window_size, window_size), Image.Resampling.BILINEAR)
                    patch_rgb_np = np.asarray(patch_rgb).astype(int)
                    patch_rgb_np_r_g = abs(np.subtract(patch_rgb_np[:,:,0], patch_rgb_np[:,:,1])) >= gray_filter_tol
                    patch_rgb_np_r_b = abs(np.subtract(patch_rgb_np[:,:,0], patch_rgb_np[:,:,2])) >= gray_filter_tol
                    patch_rgb_np_g_b = abs(np.subtract(patch_rgb_np[:,:,1], patch_rgb_np[:,:,2])) >= gray_filter_tol
                    patch_rgb_tissue = (patch_rgb_np_r_g + patch_rgb_np_r_b + patch_rgb_np_g_b) # or or or
                    patch_rgb_tissue_ratio = np.sum(patch_rgb_tissue)/(np.shape(patch_rgb_tissue)[0]*np.shape(patch_rgb_tissue)[1])*100
                    patch_rgb_tissue_ratio = np.round(patch_rgb_tissue_ratio,1)
                    
                    # save the image
                    save_data = Image.fromarray(patch_rgb_np.astype(np.uint8))
                    if (patch_rgb_tissue_ratio > tissue_ratio_threshold):
                        output_image_name1 = os.path.join(output_subfolder, image_name[:-4] + '_' + str(incre_x).zfill(3) + '_' + str(incre_y).zfill(3) + '.jpg')
                        save_data.save(output_image_name1)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', default=256, type=int, help='')
    parser.add_argument('--gray_filter_tol', default=8, type=int, help='')
    parser.add_argument('--tissue_ratio_threshold', default=25, type=int, help='')
    parser.add_argument('--target_mpp', default=0.50, type=float, help='') # for 20X, 0.25 for 40X
    
    parser.add_argument('--hospital', default='km', type=str, help='')
    parser.add_argument('--output_path', default='/media/data1/doohyun/wsi/data/svs_patch', type=str, help='')
    parser.add_argument('--wsi_folder', default='/media/data1/doohyun/wsi/data/svs/', type=str, help='')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main_worker(args)