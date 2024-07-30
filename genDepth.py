import cv2
import torch
import time
import numpy as np
import os
from PIL import Image

model_type = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

if torch.backends.mps.is_available():
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

input_folder = 'inputs'

for filename in os.listdir('inputs'):
    start = time.time()
    filepath = os.path.join(input_folder, filename)
    img = Image.open(filepath)
    img = np.array(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_BONE)

        depth_output = f'depth_{filename}'
        color_output = f'color_{filename}'

        depth_filepath = os.path.join('outputs/depth', depth_output)
        color_filepath = os.path.join('outputs/color', color_output)

        print(depth_filepath)
        print(color_filepath)

        cv2.imwrite(depth_filepath, depth_map)
        cv2.imwrite(color_filepath, img)

        end = time.time()

        total_time = end - start
        total_time *= 1000

        print(f'Processed: {filename} in {total_time}ms')

print("Processing Complete")