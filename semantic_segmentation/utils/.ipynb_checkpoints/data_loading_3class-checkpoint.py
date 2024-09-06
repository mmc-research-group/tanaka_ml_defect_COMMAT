import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir_1: str, masks_dir_2: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir_1= Path(masks_dir_1)
        self.masks_dir_2= Path(masks_dir_2)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)

        if is_mask:
            img_ndarray = np.array(np.asarray(pil_img)[:,:],dtype = "float")

            img_ndarray = np.where(img_ndarray==0,1,0)
            a = img_ndarray*255
            

        if not is_mask:
            img_ndarray = np.asarray(pil_img)
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray/255
        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file_1 = list(self.masks_dir_1.glob(name + '.*'))
        mask_file_2 = list(self.masks_dir_2.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file_1) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file_1}'
        assert len(mask_file_2) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file_2}'
        mask_1 = self.load(mask_file_1[0])
        mask_2 = self.load(mask_file_2[0])
        img = self.load(img_file[0])
        assert img.size == mask_1.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask_1.size}'
        assert img.size == mask_2.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask_2.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask_1 = self.preprocess(mask_1, self.scale, is_mask=True)
        mask_2 = self.preprocess(mask_2, self.scale, is_mask=True)
        
        pil_img = Image.fromarray(mask_1.astype(np.uint8))
        pil_img.save("mask_1.png")
        
        pil_img = Image.fromarray(mask_2.astype(np.uint8))
        
        pre_mask_1 = np.where(mask_1==1,0,1)
        pil_img = Image.fromarray(pre_mask_1.astype(np.uint8))

        pre_mask_2 = np.where(mask_2==0,2,0)
        pil_img = Image.fromarray(pre_mask_2.astype(np.uint8))
        
        pre_mask = pre_mask_1 + pre_mask_2
        pil_img = Image.fromarray(pre_mask.astype(np.uint8))

        mask = np.where(pre_mask==3,1,pre_mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
