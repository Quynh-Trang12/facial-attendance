"""CPU tensor conversion shared by emotion loading and later model inference."""
import numpy as np
import torch
from PIL import Image


class GrayscaleTensor:
    """Deterministic grayscale conversion and /255 scaling; no augmentation."""
    def __init__(self,image_size=48):
        if image_size!=48:raise ValueError('Baseline image size is 48.')
        self.image_size=image_size

    def __call__(self,image):
        image=image.convert('L')
        if image.size!=(self.image_size,self.image_size):
            image=image.resize((self.image_size,self.image_size),Image.Resampling.BILINEAR)
        array=np.array(image,dtype=np.float32,copy=True)/np.float32(255)
        return torch.from_numpy(array).unsqueeze(0)
