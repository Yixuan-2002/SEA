import os
import cv2
import numpy as np

from VIFB_metrics.metricsQabf import metrics_qabf

print('<!-- a_VIFB_metrics_Qabf -->')

results_folder = 'VIF_Results_FMB/'
vis_folder = results_folder + 'Visible/'
ir_folder = results_folder + 'Infrared/'

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
            
    fusd_images_folder = results_folder + method
    image_names = os.listdir(fusd_images_folder)
    image_names = sorted(image_names)

    # image_names = image_names[:2]

    result_MI = 0
    for img_name in image_names:
        img1 = cv2.imread(os.path.join(vis_folder, img_name))
        img2 = cv2.imread(os.path.join(ir_folder, img_name))
        fused = cv2.imread(os.path.join(fusd_images_folder, img_name))
        if fused.shape[0] != img1.shape[0]:
            fused = cv2.resize(fused, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)

        result_MI += metrics_qabf(img1, img2, fused)

    result_MI = result_MI / len(image_names)    
    print(method + ': ' + str(np.round(result_MI, 3)))
