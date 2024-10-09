# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

from PIL import Image
import numpy as np
np.random.seed(1)

import torch
from torchvision import transforms

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.visualizer import Visualizer
from utils.distributed import init_distributed

import cv2

# import os
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.eval import compute_results

logger = logging.getLogger(__name__)


def main(args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['base_path'] = absolute_user_dir
    opt = init_distributed(opt)

    # META DATA
    pretrained_pth = os.path.join(opt['RESUME_FROM'])
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    stuff_classes = ["Car", "Bus", "Motorcycle", "Bicycle", "Pedestrian", "Motorcyclist", "Bicyclist", "Cart", "Bench", "Umbrella", 
        "Box", "Pole", "Street_lamp", "Traffic_light", "Traffic_sign", "Car_stop", "Color_cone", "Sky", "Road", "Sidewalk", 
        "Curb", "Vegetation", "Terrain", "Building", "Ground"]
    # stuff_classes = [
    #     "Car: a motor vehicle with four wheels, used for transporting passengers",
    #     "Bus: a large motor vehicle carrying passengers by road",
    #     "Motorcycle: a two-wheeled motor vehicle",
    #     "Bicycle: a human-powered vehicle with two wheels",
    #     "Pedestrian: person walking",
    #     "Motorcyclist: person who rides a motorcycle",
    #     "Bicyclist: person who rides a bicycle",
    #     "Cart: a small vehicle pushed or pulled by hand",
    #     "Bench: a long seat for multiple people",
    #     "Umbrella: a device for protection against rain or sun",
    #     "Box: a container with flat sides and a lid",
    #     "Pole: a long, slender, rounded piece of wood or metal",
    #     "Street_lamp: a lamp that illuminates a street",
    #     "Traffic_light: a signaling device positioned at road intersections",
    #     "Traffic_sign: a sign providing information or instructions to road users",
    #     "Car_stop: designated area where cars are required to stop",
    #     "Color_cone: a conical marker used to direct traffic",
    #     "Sky: the region of the atmosphere and outer space seen from Earth",
    #     "Road: a wide way leading from one place to another, typically paved",
    #     "Sidewalk: a paved path for pedestrians at the side of a road",
    #     "Curb: the edge of a sidewalk or road",
    #     "Vegetation: plants in general or the plants in a particular area",
    #     "Terrain: a stretch of land, especially with regard to its physical features",
    #     "Building: a structure with a roof and walls",
    #     "Ground: the solid surface of the earth"
    # ]
    stuff_colors_dict = {
        # 0: (0, 0, 0),          # background (unlabeled)
        1: (0, 0, 142),        # Car
        2: (0, 60, 100),       # Bus
        3: (0, 0, 230),        # Motorcycle
        4: (119, 11, 32),      # Bicycle
        5: (255, 0, 0),        # Pedestrian
        6: (0, 139, 139),      # Motorcyclist
        7: (255, 165, 150),    # Bicyclist
        8: (192, 64, 0),       # Cart
        9: (211, 211, 211),    # Bench
        10: (100, 33, 128),    # Umbrella
        11: (117, 79, 86),     # Box
        12: (153, 153, 153),   # Pole
        13: (190, 122, 222),   # Street_lamp
        14: (250, 170, 30),    # Traffic_light
        15: (220, 220, 0),     # Traffic_sign
        16: (222, 142, 35),    # Car_stop
        17: (205, 155, 155),   # Color_cone
        18: (70, 130, 180),    # Sky
        19: (128, 64, 128),    # Road
        20: (244, 35, 232),    # Sidewalk
        21: (0, 0, 70),        # Curb
        22: (107, 142, 35),    # Vegetation
        23: (152, 251, 152),   # Terrain
        24: (70, 70, 70),      # Building
        25: (110, 80, 100)     # Ground
    }

    # model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)
    
    # List of methods
    methods = [
        "Visible",
        "Infrared",
        "2018_DenseFuse",
        "2019_FusionGAN",
        "2020_U2Fusion",
        "2020_DDcGAN",
        "2021_SDNet",
        "2021_RPNNest",
        "2022_SwinFusion",
        "2022_PIAFusion",
        "2022_TarDAL",
        "2023_LRRNet",
        "2023_DifFusion",
        "2023_DIVFusion",
        "2023_DLF",
        "2023_CDDFuse",
        "2023_MetaFusion",
        "2023_TGFuse",
        "2023_DDFM",
        "2024_SHIP",
        "2024_TCMoA",
        "2024_TextIF",
        "2024_DDBF",
        "2024_EMMA",
        "s2022_SeAFusion",
        "s2022_SuperFusion",
        "s2023_PSFusion",
        "s2023_SegMiF",
        "s2023_PAIF",
        "s2024_TIM",
        "s2024_SDCFusion",
        "s2024_MRFS"
    ]

    # For loop to process each method
    for method in methods:
            
        folder_pth = "VIF_Results_MVSeg/" + method
        output_root = os.path.join('./VIF_Results_MVSeg', opt['RESUME_FROM'].split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
        image_names = os.listdir(folder_pth)
        image_names = sorted(image_names)

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        for image_name in image_names:
            with torch.no_grad():
                image_pth = os.path.join(folder_pth,image_name)
                # print(image_pth)
                image_ori = Image.open(image_pth).convert("RGB")
                width = image_ori.size[0]
                height = image_ori.size[1]
                image = transform(image_ori)
                image = np.asarray(image)
                image_ori = np.asarray(image_ori)
                images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

                batch_inputs = [{'image': images, 'height': height, 'width': width}]
                outputs = model.forward(batch_inputs)

                sem_seg = outputs[-1]['sem_seg'].max(0)[1]
                sem_seg = sem_seg + 1
                sem_seg_color = Image.fromarray(sem_seg.cpu().numpy().astype(np.uint8), mode='P')
                palette = []
                for label in range(len(stuff_classes) + 1):
                    if label in stuff_colors_dict:
                        palette.extend(stuff_colors_dict[label])
                    else:
                        palette.extend([0, 0, 0])
                sem_seg_color.putpalette(palette)
                sem_seg_color.save(os.path.join(output_root, image_name))

        num_classes = len(stuff_classes)
        label_folder = "VIF_Results_MVSeg/Label"
        label_names = os.listdir(label_folder)
        label_names = sorted(label_names)

        pred_folder = output_root
        eval_classes = list(range(1,num_classes+1))
        num_class_eval = len(eval_classes)
        conf_total = np.zeros((num_class_eval, num_class_eval))

        for label_name in label_names:
            label_pth = os.path.join(label_folder, label_name)
            label = np.asarray(Image.open(label_pth))

            pred_pth = os.path.join(pred_folder, label_name)
            # pred = np.asarray(Image.open(pred_pth).convert('P'))
            pred_img = Image.open(pred_pth).convert('P')
            pred = np.asarray(pred_img)
            if label.shape != pred.shape:
                pred_resized = pred_img.resize(label.shape[::-1], Image.NEAREST)  # Resize to match label dimensions
                pred = np.asarray(pred_resized)

            conf = confusion_matrix(y_true=label.flatten(), y_pred=pred.flatten(), labels=eval_classes ) 
            precision_per_class, recall_per_class, iou_per_class = compute_results(conf)
            conf_total += conf

        precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)

        print(pred_folder)
        # print('iou_per_class:', np.round(iou_per_class*100, 1))
        stuff_classes = [cls.split(':')[0] for cls in stuff_classes]
        iou_dict = {stuff_classes[i]: np.round(iou * 100, 1) for i, iou in enumerate(iou_per_class)}
        print(iou_dict)
        print('miou', round(np.nanmean(iou_per_class*100), 1))


if __name__ == "__main__":
    main()
    sys.exit(0)