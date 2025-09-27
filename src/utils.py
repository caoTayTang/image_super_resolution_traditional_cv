import os
import cv2
import shutil
from glob import glob

def load_hr_image(path, scale=4):
    hr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    hr = hr.astype("float32") / 255.0
    H, W = hr.shape
    Hc, Wc = H - (H % scale), W - (W % scale)
    return hr[:Hc, :Wc]

def img_classify(base_dir, output_dir):
    for dir in output_dir:
        os.makedirs(dir, exist_ok=True)
    srfs = [2,3,4]

    for srf in srfs:
        srf_folder = os.path.join(base_dir, f"image_SRF_{srf}")
        lr_out = os.path.join(output_dir[0], f"LR_x{srf}")
        hr_out = os.path.join(output_dir[1], f'HR_x{srf}')
        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)

        images = glob(os.path.join(srf_folder, '*.png'))
        for img_path in images:
            file_name = os.path.basename(img_path)
            if '_LR' in file_name:
                shutil.copy(img_path, os.path.join(lr_out, file_name))
            elif '_HR' in file_name:
                shutil.copy(img_path, os.path.join(hr_out, file_name))
            else:
                print(f'Warning: unknown type {file_name}')
        print(f'SRF_{srf} done: {len(images)} images processed.')
            