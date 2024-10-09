import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
from torchvision import transforms
from PIL import Image

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/dongzhe/dayan/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/dongzhe/dayan/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/dongzhe/dayan/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Predict classes and hyper-param for GroundingDINO
SOURCE_IMAGE_PATH = "/home/dongzhe/dayan/datasets/FMB/test/Visible/00043.png"
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
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

image = cv2.imread(SOURCE_IMAGE_PATH)

# detect objects
detections = grounding_dino_model.predict_with_classes(
    image=image,
    classes=CLASSES,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# annotate image with detections
box_annotator = sv.BoxAnnotator()
labels = [
    f"{CLASSES[class_id]} {confidence:0.2f}" 
    for _, _, confidence, class_id, _, _ 
    in detections]
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

cv2.imwrite("inference/groundsam/groundingdino_annotated_image.jpg", annotated_frame)

# NMS post process
print(f"Before NMS: {len(detections.xyxy)} boxes")
nms_idx = torchvision.ops.nms(
    torch.from_numpy(detections.xyxy), 
    torch.from_numpy(detections.confidence), 
    NMS_THRESHOLD
).numpy().tolist()

detections.xyxy = detections.xyxy[nms_idx]
detections.confidence = detections.confidence[nms_idx]
detections.class_id = detections.class_id[nms_idx]

print(f"After NMS: {len(detections.xyxy)} boxes")

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
sem_seg_color.save('inference/groundsam/masks.png')