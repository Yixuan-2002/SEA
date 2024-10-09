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

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=args.GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=args.GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[args.SAM_ENCODER_VERSION](checkpoint=args.SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    CLASSES = [
        # "background (unlabeled): background or unlabeled area",
        "Car: a motor vehicle with four wheels, used for transporting passengers",
        "Bus: a large motor vehicle carrying passengers by road",
        "Motorcycle: a two-wheeled motor vehicle",
        "Bicycle: a human-powered vehicle with two wheels",
        "Pedestrian: person walking",
        "Motorcyclist: person who rides a motorcycle",
        "Bicyclist: person who rides a bicycle",
        "Cart: a small vehicle pushed or pulled by hand",
        "Bench: a long seat for multiple people",
        "Umbrella: a device for protection against rain or sun",
        "Box: a container with flat sides and a lid",
        "Pole: a long, slender, rounded piece of wood or metal",
        "Street_lamp: a lamp that illuminates a street",
        "Traffic_light: a signaling device positioned at road intersections",
        "Traffic_sign: a sign providing information or instructions to road users",
        "Car_stop: designated area where cars are required to stop",
        "Color_cone: a conical marker used to direct traffic",
        "Sky: the region of the atmosphere and outer space seen from Earth",
        "Road: a wide way leading from one place to another, typically paved",
        "Sidewalk: a paved path for pedestrians at the side of a road",
        "Curb: the edge of a sidewalk or road",
        "Vegetation: plants in general or the plants in a particular area",
        "Terrain: a stretch of land, especially with regard to its physical features",
        "Building: a structure with a roof and walls",
        "Ground: the solid surface of the earth"
    ]
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

        folder_pth = "VIF_Results_MVSeg/" + method
        output_root = os.path.join('./VIF_Results_MVSeg', args.GROUNDING_DINO_CHECKPOINT_PATH.split('/')[-1].split('.')[0] + '-' + args.SAM_CHECKPOINT_PATH.split('/')[-1].split('.')[0], folder_pth.split('/')[-1])
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

                sem_seg = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
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

        num_classes = len(CLASSES)
        label_folder = "VIF_Results_MVSeg/Label"
        label_names = os.listdir(label_folder)
        label_names = sorted(label_names)

        pred_folder = output_root
        eval_classes = list(range(1,num_classes+1))

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
        CLASSES = [cls.split(':')[0] for cls in CLASSES]
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