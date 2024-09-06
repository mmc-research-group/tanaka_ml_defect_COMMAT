import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.dice_score import dice_loss
from PIL import Image
import matplotlib.pyplot as plt




def val_loss(net, dataloader, device):
    criterion = nn.CrossEntropyLoss()
    num_val_batches = len(dataloader)
    val_loss = 0


    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)


        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score = criterion(mask_pred, mask_true) \
                            + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                        F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)

            else:
                
                # compute the Dice score, ignoring background  
                val_loss = criterion(mask_pred, mask_true) \
                            + dice_loss(F.softmax(mask_pred, dim=1).float(),
                                        F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)
        return val_loss







