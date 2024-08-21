import torch
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint="./sam_vit_h_4b8939.pth")
sam.to(device=DEVICE)

import cv2
from segment_anything import SamAutomaticMaskGenerator
import supervision as sv

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread("/home/chkim/GraspNet1Billiion/test/test_novel/scene_0161/realsense/rgb/0063.png")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sam_result = mask_generator.generate(image_rgb)

mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=cv2.zero(image_bgr.shape).copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

cv2.imwrite('annotated_image.jpg', annotated_image)

