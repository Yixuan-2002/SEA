import os
import cv2
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
from utils.eval import compute_results

stuff_classes = ['road','sidewalk','building','lamp','sign','vegetation','sky','person','car','truck','bus','motorcycle','bicycle','pole']
num_classes = len(stuff_classes)
label_folder = "VIF_Results_FMB/Label"
label_names = os.listdir(label_folder)
label_names = sorted(label_names)

# results_folder = 'VIF_Results_FMB/xdecoder_focall_last/'
results_folder = 'VIF_Results_FMB/seem_focall_v1/'

###### Baseline ########
pred_folder = results_folder+"Visible"
# pred_folder = results_folder+"Infrared"
###### Baseline ########

###### SOTA ########
# pred_folder = results_folder+"2018_DenseFuse"
# pred_folder = results_folder+"2019_FusionGAN"
# pred_folder = results_folder+"2020_U2Fusion"
# pred_folder = results_folder+"2020_DDcGAN"
# pred_folder = results_folder+"2021_SDNet"
# pred_folder = results_folder+"2021_RPNNest"
# pred_folder = results_folder+"2022_SwinFusion"
# pred_folder = results_folder+"2022_PIAFusion"
# pred_folder = results_folder+"2022_TarDAL"
# pred_folder = results_folder+"2023_LRRNet"
# pred_folder = results_folder+"2023_DifFusion"
# pred_folder = results_folder+"2023_DIVFusion"
# pred_folder = results_folder+"2023_DLF"
# pred_folder = results_folder+"2023_CDDFuse"
# pred_folder = results_folder+"2023_MetaFusion"
# pred_folder = results_folder+"2023_TGFuse"
# pred_folder = results_folder+"2023_DDFM"
# pred_folder = results_folder+"2024_SHIP"
# pred_folder = results_folder+"2024_TCMoA"
# pred_folder = results_folder+"2024_TextIF"
# pred_folder = results_folder+"2024_DDBF"
# pred_folder = results_folder+"2024_EMMA"
###### SOTA ########

###### SOTA (Unified) ########
# pred_folder = results_folder+"s2022_SeAFusion"
# pred_folder = results_folder+"s2022_SuperFusion"
# pred_folder = results_folder+"s2023_PSFusion"
# pred_folder = results_folder+"s2023_SegMiF"
# pred_folder = results_folder+"s2023_PAIF"
# pred_folder = results_folder+"s2024_TIM"
# pred_folder = results_folder+"s2024_SDCFusion"
# pred_folder = results_folder+"s2024_MRFS"
###### SOTA (Unified) ########

# eval_classes = list(range(1,num_classes+1))
### no bicycle
eval_classes = list(range(1,num_classes+1))
position_bicycle = stuff_classes.index('bicycle')
eval_classes.pop(position_bicycle)
stuff_classes.pop(position_bicycle)

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
iou_dict = {stuff_classes[i]: np.round(iou * 100, 1) for i, iou in enumerate(iou_per_class)}
print(iou_dict)
print('miou', round(np.nanmean(iou_per_class*100), 1))


