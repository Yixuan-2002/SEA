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

    stuff_classes = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
    # stuff_classes = [
    #     "road: a wide way leading from one place to another, typically paved",
    #     "sidewalk: a paved path for pedestrians at the side of a road",
    #     "building: a structure with a roof and walls",
    #     "lamp: a device that provides light",
    #     "sign: a board or placard providing information or instructions to road users",
    #     "vegetation: plants in general or the plants in a particular area",
    #     "sky: the region of the atmosphere and outer space seen from Earth",
    #     "person: a human being",
    #     "car: a motor vehicle with four wheels, used for transporting passengers",
    #     "truck: a large, heavy motor vehicle for transporting goods or materials",
    #     "bus: a large motor vehicle carrying passengers by road",
    #     "motorcycle: a two-wheeled motor vehicle",
    #     "bicycle: a human-powered vehicle with two wheels",
    #     "pole: a long, slender, rounded piece of wood or metal"
    # ]
    stuff_colors_dict = {
        0: (0, 0, 0),
        1: (179, 228, 228), # road
        2: (181, 57, 133), # sidewalk
        3: (67, 162, 177), # building
        4: (200, 178, 50), # lamp
        5: (132, 45, 199), # sign
        6: (66, 172, 84), # vegetation
        7: (179, 73, 79), # sky
        8: (76, 99, 166), # person
        9: (66, 121, 253), # car
        10: (137, 165, 91), # truck
        11: (155, 97, 152), # bus
        12: (105, 153, 140), # motorcycle
        13: (222, 215, 158), # bicycle
        14: (135, 113, 90), # pole
    }
    # stuff_colors = [stuff_colors_dict[i+1] for i in range(len(stuff_classes))]

    # stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    # MetadataCatalog.get("demo").set(
    #     stuff_colors=stuff_colors,
    #     stuff_classes=stuff_classes,
    #     stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    # )
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
            
        folder_pth = "VIF_Results_FMB/" + method
        output_root = os.path.join('./VIF_Results_FMB', opt['RESUME_FROM'].split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
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
                # visual = Visualizer(image_ori, metadata=metadata)

                sem_seg = outputs[-1]['sem_seg'].max(0)[1]
                # demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image
                # demo.save(os.path.join(output_root, image_name))

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

                # import pdb
                # pdb.set_trace()

                # print(1)

        label_folder = "VIF_Results_FMB/Label"
        label_names = os.listdir(label_folder)
        label_names = sorted(label_names)
        pred_folder = output_root

        eval_stuff_classes = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
        num_classes = len(stuff_classes)
        ### no background and bicycle
        eval_classes = list(range(1,num_classes+1))
        position_bicycle = eval_stuff_classes.index('bicycle')
        eval_classes.pop(position_bicycle)
        eval_stuff_classes.pop(position_bicycle)

        # conf_total = np.zeros((num_classes, num_classes))
        num_class_eval = len(eval_stuff_classes)
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
        iou_dict = {eval_stuff_classes[i]: np.round(iou * 100, 1) for i, iou in enumerate(iou_per_class)}
        print(iou_dict)
        print('miou', round(np.nanmean(iou_per_class*100), 1))


if __name__ == "__main__":
    main()
    sys.exit(0)