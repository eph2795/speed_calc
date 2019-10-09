import numpy as np
import matplotlib.pyplot as plt

from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90,
    Transpose, Compose, PadIfNeeded, Normalize,
    RandomScale, Rotate, Resize
)


def postprocess(image):
    x, y, z = np.where(image.astype(bool))
    r = np.max(x) 
    l = np.min(x)
    d = np.max(y) 
    u = np.min(y)    
    new_l = np.random.randint(1, image.shape[0] - (r - l) - 1, 1)[0] 
    new_u = np.random.randint(1, image.shape[1] - (d - u) - 1, 1)[0]
    new_image = np.zeros_like(image, dtype=np.float64)
    new_image[new_l : new_l+(r-l)+1, new_u : new_u+(d-u)+1] = image[l : r+1, u : d+1] 
    return new_image


def augment_and_show(aug, image):
    image = aug(image=image)['image']
    image = postprocess(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    return image


normalize = Normalize(max_pixel_value=1)
def augment(aug, image):
    image = aug(image=image)['image']
    image = postprocess(image)
    image = normalize(image=image)['image']
    return image
 
    
def augment_flips_color(h=None, w=None, interpolation=0):
    t = [
        RandomRotate90(always_apply=True),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Transpose(p=0.5),
        Rotate(limit=(-45, 45), interpolation=interpolation, border_mode=0, always_apply=True),
        RandomScale(scale_limit=(-0.2, 0.5), interpolation=interpolation, always_apply=True),
        PadIfNeeded(h, w, border_mode=0, always_apply=True),
        Resize(h, w, interpolation=interpolation, always_apply=True),
    ]
    return Compose(t, p=1)

def pad_img(h, w, interpolation=0):
    return Compose([
        PadIfNeeded(h, w, border_mode=0, always_apply=True),
        Resize(h, w, interpolation=interpolation, always_apply=True),
    ])