from match_anything.third_party.ROMA.roma.matchanything_roma_model import MatchAnything_Model

from PIL import Image

# Load images
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import cv2
from match_anything.config.default import _CN as cfg
import kornia as K
import kornia.feature as KF
from kornia_moons.viz import draw_LAF_matches
from match_anything.utils.model_utils import load_config

config = load_config("matchanything_roma")
img1_path = Path("/home/landson/RGBT-Scenes/Building/rgb/test/img_001.jpg")
img2_path = Path("/home/landson/RGBT-Scenes/Building/rgb/test/img_009.jpg")
img1 = K.io.load_image(img1_path, K.io.ImageLoadType.RGB32)[None, ...]
img2 = K.io.load_image(img2_path, K.io.ImageLoadType.RGB32)[None, ...]
print(f"Loaded images: {img1.shape}, {img2.shape}")

matcher = MatchAnything_Model(config, True)
#load weights
ckpt = torch.load("weights/matchanything_roma.ckpt", map_location='cpu')
matcher.load_state_dict(ckpt['state_dict'])

matcher.eval()

input_dict = {
    "image0_rgb": img1,
    "image1_rgb": img2,
}

with torch.inference_mode():
    batch = matcher(input_dict)

mkpts0 = batch['mkpts0_f'].cpu().numpy()
mkpts1 = batch['mkpts1_f'].cpu().numpy()
Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
inliers = inliers > 0

draw_LAF_matches(
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts0).view(1, -1, 2),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    ),
    KF.laf_from_center_scale_ori(
        torch.from_numpy(mkpts1).view(1, -1, 2),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
        torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    ),
    torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    K.tensor_to_image(img1),
    K.tensor_to_image(img2),
    inliers,
    draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
)

plt.show()
