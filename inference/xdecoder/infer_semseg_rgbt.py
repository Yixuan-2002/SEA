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
    # output_root = './output_rgbt'
    output_root = os.path.join('./output_rgbt',opt['RESUME_FROM'].split('/')[-1].split('.')[0])
    # image_pth = 'inference/images/animals.png'
    folder_pth = 'inference/RGBT/[ICCV23]FMB/00089'
    image_names = os.listdir(folder_pth)


    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    # stuff_classes = ['zebra','antelope','giraffe','ostrich','sky','water','grass','sand','tree']
    stuff_classes = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
    # stuff_classes = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle']
    # stuff_colors = [random_color(rgb=True, maximum=255).astype(int).tolist() for _ in range(len(stuff_classes))]
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
    stuff_colors = [stuff_colors_dict[i+1] for i in range(len(stuff_classes))]

    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes + ["background"], is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    for image_name in image_names:
        with torch.no_grad():
            image_pth = os.path.join(folder_pth,image_name)
            print(image_pth)
            image_ori = Image.open(image_pth).convert("RGB")
            width = image_ori.size[0]
            height = image_ori.size[1]
            image = transform(image_ori)
            image = np.asarray(image)
            image_ori = np.asarray(image_ori)
            images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()

            batch_inputs = [{'image': images, 'height': height, 'width': width}]
            outputs = model.forward(batch_inputs)
            visual = Visualizer(image_ori, metadata=metadata)

            sem_seg = outputs[-1]['sem_seg'].max(0)[1]
            demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image

            if not os.path.exists(output_root):
                os.makedirs(output_root)
            # demo.save(os.path.join(output_root, 'sem.png'))
            demo.save(os.path.join(output_root, image_name))


if __name__ == "__main__":
    main()
    sys.exit(0)