import cv2
import numpy as np
import supervision as sv
import argparse
import os
import sys

import torch
import torchvision
from torchvision import transforms
from PIL import Image

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_results(conf_total):
    n_class =  conf_total.shape[0]
    consider_unlabeled = True  # must consider the unlabeled, please set it to True
    if consider_unlabeled is True:
        start_index = 0
    else:
        start_index = 1
    precision_per_class = np.zeros(n_class)
    recall_per_class = np.zeros(n_class)
    iou_per_class = np.zeros(n_class)
    for cid in range(start_index, n_class): # cid: class id
        if conf_total[start_index:, cid].sum() == 0:
            precision_per_class[cid] =  np.nan
        else:
            precision_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[start_index:, cid].sum()) # precision = TP/TP+FP
        if conf_total[cid, start_index:].sum() == 0:
            recall_per_class[cid] = np.nan
        else:
            recall_per_class[cid] = float(conf_total[cid, cid]) / float(conf_total[cid, start_index:].sum()) # recall = TP/TP+FN
        if (conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid]) == 0:
            iou_per_class[cid] = np.nan
        else:
            iou_per_class[cid] = float(conf_total[cid, cid]) / float((conf_total[cid, start_index:].sum() + conf_total[start_index:, cid].sum() - conf_total[cid, cid])) # IoU = TP/TP+FP+FN

    return precision_per_class, recall_per_class, iou_per_class

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def main(args):
    # folder_pth = args.SOURCE_IMAGE_PATH
    # output_root = os.path.join('./VIF_Results_FMB', args.SAM_CHECKPOINT_PATH.split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
    # image_names = os.listdir(folder_pth)
    # image_names = sorted(image_names)

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=args.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=args.GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[args.SAM_ENCODER_VERSION](checkpoint=args.SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    CLASSES = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
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
    # BOX_THRESHOLD = 0.25
    # TEXT_THRESHOLD = 0.25
    # NMS_THRESHOLD = 0.8
    BOX_THRESHOLD = args.BOX_THRESHOLD
    TEXT_THRESHOLD = args.TEXT_THRESHOLD
    NMS_THRESHOLD = args.NMS_THRESHOLD
    
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
        # output_root = os.path.join('./VIF_Results_FMB', args.SAM_CHECKPOINT_PATH.split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
        output_root = os.path.join('./VIF_Results_FMB', args.GROUNDING_DINO_CHECKPOINT_PATH.split('/')[-1].split('.')[0] + '-' + args.SAM_CHECKPOINT_PATH.split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
        image_names = os.listdir(folder_pth)
        image_names = sorted(image_names)
        
        if not os.path.exists(output_root):
            os.makedirs(output_root)
        for image_name in tqdm(image_names):
            with torch.no_grad():
                image_pth = os.path.join(folder_pth,image_name)
                image = cv2.imread(image_pth)

                # detect objects
                detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=CLASSES,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )

                # NMS post process
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    NMS_THRESHOLD
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]

                # convert detections to masks
                detections.mask = segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )

                labels = [
                    class_id
                    for _, _, confidence, class_id, _, _ 
                    in detections]

                sem_seg = np.zeros((detections.mask.shape[1], detections.mask.shape[2]), dtype=np.uint8)
                for box_idx, box_mask in enumerate(detections.mask):
                    sem_seg[box_mask] = labels[box_idx] + 1

                sem_seg_color = Image.fromarray(sem_seg, mode='P')
                palette = []
                for label in range(len(CLASSES) + 1):
                    if label in stuff_colors_dict:
                        palette.extend(stuff_colors_dict[label])
                    else:
                        palette.extend([0, 0, 0])
                sem_seg_color.putpalette(palette)
                sem_seg_color.save(os.path.join(output_root, image_name))

        CLASSES = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
        num_classes = len(CLASSES)
        label_folder = "VIF_Results_FMB/Label"
        label_names = os.listdir(label_folder)
        label_names = sorted(label_names)

        pred_folder = output_root
        # eval_classes = list(range(1,num_classes+1))
        ### no bicycle
        eval_classes = list(range(1,num_classes+1))
        position_bicycle = CLASSES.index('bicycle')
        eval_classes.pop(position_bicycle)
        CLASSES.pop(position_bicycle)

        # conf_total = np.zeros((num_classes, num_classes))
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
        iou_dict = {CLASSES[i]: np.round(iou * 100, 1) for i, iou in enumerate(iou_per_class)}
        print(iou_dict)
        print('miou', round(np.nanmean(iou_per_class*100), 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo")

    # Grounding Dino
    parser.add_argument(
        "--GROUNDING_DINO_CONFIG_PATH", type=str, required=False, help = "path to config file",
        default="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        "--GROUNDING_DINO_CHECKPOINT_PATH", type=str, required=False, help="path to checkpoint file",
        default="Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
    )

    # SAM
    parser.add_argument(
        "--SAM_ENCODER_VERSION", type=str, required=False, help="SAM Encoder Version",
        default="vit_h"
    )
    parser.add_argument(
        "--SAM_CHECKPOINT_PATH", type=str, required=False, help="path to checkpoint file",
        default="Grounded-Segment-Anything/sam_vit_h_4b8939.pth"
    )

    parser.add_argument(
        "--BOX_THRESHOLD", type=float, required=False, default=0.25
    )
    parser.add_argument(
        "--TEXT_THRESHOLD", type=float, required=False, default=0.25
    )
    parser.add_argument(
        "--NMS_THRESHOLD", type=float, required=False, default=0.8
    )

    # # image path
    # parser.add_argument(
    #     "--SOURCE_IMAGE_PATH", type=str, required=False, help="source image path",
    #     default="./VIF_Results_FMB/Visible"
    # )

    args = parser.parse_args()
    main(args)

    sys.exit(0)