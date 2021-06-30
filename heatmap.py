import torch
import numpy as np
import cv2
from scipy.special import expit

cv2.setNumThreads(1)

def overlay_cam(img, cam, weight=0.5, img_max=255.):
    """
    Red is the most important region
    Args:
        img: numpy array (h, w) or (h, w, 3)
    """

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()
        
    if len(img.shape) == 2:
        h, w = img.shape
        img = img.reshape(h, w, 1)
        img = np.repeat(img, 3, axis=2)

    h, w, c = img.shape

    # normalize the cam
    x = expit(cam)
    x = x - x.min()
    x = x / x.max()
    # resize the cam
    x = cv2.resize(x, (w, h))
    x = x - x.min()
    x = x / x.max()
    # coloring the cam
    x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
    x = np.float32(x) / 255.

    # overlay
    x = img / img_max + weight * x
    x = x / x.max()
    x = np.uint8(255 * x)
    return x