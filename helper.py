import torch, yaml
import numpy as np
from statistics import mean
import torch.nn as nn
from einops import rearrange
import cv2

"""
    A few helper functions used throughout the codebase for various purposes.
"""

def distortion2lbl(distortion):
    distortion2nbr_dct = {
        "clean": 0,
        "snow": 1,
        "contrast_inc": 2,
        "compression": 3,
        "brightness": 4,
        "oversharpen": 5,
        "noise": 6,
        "blur": 7,
        "haze": 8,
        "rain": 9,
        "saturate_inc": 10,
        "saturate_dec": 11,
        "contrast_dec": 12,
        "darken": 13,
        "pixelate": 14
    }
    return distortion2nbr_dct[distortion]

def lbl2distortion(lbl):
    nbr2distortion_dct = {
        0: "clean",
        1: "snow",
        2: "contrast_inc",
        3: "compression",
        4: "brightness",
        5: "oversharpen",
        6: "noise",
        7: "blur",
        8: "haze",
        9: "rain",
        10: "saturate_inc",
        11: "saturate_dec",
        12: "contrast_dec",
        13: "darken",
        14: "pixelate"
    }
    return nbr2distortion_dct[lbl]

def comparison2lbl(comparison):
    com2nbr_dct = {
        "same": 0,
        "slightly_worse": 1,
        "significantly_worse": 2,
        "slightly_better": 3,
        "significantly_better": 4
    }
    return com2nbr_dct[comparison]

def lbl2comparison(lbl):
    nbr2comb_dct = {
        0: "same",
        1: "slightly_worse",
        2: "significantly_worse",
        3: "slightly_better",
        4: "significantly_better"
    }
    return nbr2comb_dct[lbl]

def sev2lbl(comparison):
    com2nbr_dct = {
        "clean": 0,
        "minor": 1,
        "moderate": 2,
        "severe": 3
    }
    return com2nbr_dct[comparison]

def lbl2sev(lbl):
    nbr2comb_dct = {
        0: "clean",
        1: "minor",
        2: "moderate",
        3: "severe"
    }
    return nbr2comb_dct[lbl]

def loadconfig(configpath):
    with open(configpath, "r") as fp:
        config = yaml.safe_load(fp)
    return config

class MetricMonitor:
    def __init__(self, lst_of_metrics_to_monitor):
        self.metrics = {}
        for i in lst_of_metrics_to_monitor:
            self.metrics[i] = []
    
    def set_metric(self, metric_name, val, reduced=False):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if reduced: assert len(self.metrics[metric_name]) == 0, f"Setting reduced {metric_name}, but it has prior values."
        self.metrics[metric_name].append(val)
    
    def reset_specific_metric(self, metric_name):
        if len(self.metrics[metric_name]) > 0:
            self.metrics[metric_name] = []
    
    def whatis_logged(self):
        return list(self.metrics.keys())

    def reset(self, mode=None):
        # reset for particular mode or entirely
        for key in self.metrics.keys():
            if mode is not None:
                if mode in key:
                    self.metrics[key] = []
            else:
                self.metrics[key] = []

    def flush_metrics(self):
        # this is for inference
        for i in range(len(self.metrics)):
            key = list(self.metrics.keys())[i]
            print(f"[Accuracy/MAE] {key}: {mean(self.metrics[key])}")

    def print_log(self, logger, epoch, mode="val"):
        logger.info(f"[Accuracy/MAE] for Epoch: {epoch+1}")        
        for i in range(len(self.metrics)):
            key = list(self.metrics.keys())[i]
            # ignore the train because we wrote it already
            if key != "total_loss":
                if mode in key:
                    logger.info(f"[Accuracy/MAE] {key}: {mean(self.metrics[key])}")
    
    def get_specific_metric(self, metric_name):
        return mean(self.metrics[metric_name])
    
    def get_all_thats_logged(self, mode):
        dct_of_vals = {}
        for i in range(len(self.metrics)):
            key = list(self.metrics.keys())[i]
            # ignore the train because we wrote to TB already
            if key != "total_loss":
                if mode in key:
                    value = mean(self.metrics[key])
                    dct_of_vals[key] = value
        return dct_of_vals
        
    def write_to_wandb(self, run, mode):
        for i in range(len(self.metrics)):
            key = list(self.metrics.keys())[i]
            if key != "total_loss":
                if mode in key:
                    value = mean(self.metrics[key])
                    run.log({key: value})
    
    def write_to_tensorboard(self, writer, epoch, mode):
        for i in range(len(self.metrics)):
            key = list(self.metrics.keys())[i]
            # ignore the train because we wrote to TB already
            if key != "total_loss":
                if mode in key:
                    value = mean(self.metrics[key])
                    writer.add_scalar(key, value, epoch+1)

def resize_mask(mask, max_height, max_width):
    return nn.functional.interpolate(mask.unsqueeze(0), size=(max_height, max_width), mode='nearest').squeeze(0)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class MultipleSequential(nn.Sequential):
    # taken from: https://github.com/pytorch/pytorch/issues/19808#issuecomment-487291323
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def one_hot_encode(labels, num_classes=3):
    b, r = labels.shape
    one_hot = torch.zeros(b, r, num_classes, dtype=torch.float32).to(labels.device)
    valid_mask = labels != -1 # -1 is the invalid region pad
    one_hot[valid_mask] = torch.eye(num_classes).to(labels.device)[labels[valid_mask]]
    return one_hot

def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)):
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)
    overlayed_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image

def get_valid_indices_from_padded(padded_mask):
    flat_padded_mask = padded_mask.view(padded_mask.size(0), -1)
    all_zeros_mask = (flat_padded_mask == 0).all(dim=1)
    valid_indices = (~all_zeros_mask).nonzero(as_tuple=True)[0]
    return valid_indices

def unpad_masks(padded_masks):
    # a helper utility to go back to the list
    batch_size, max_regions, H, W = padded_masks.shape
    unpadded_list = []
    for b in range(batch_size):
        masks = []
        for r in range(max_regions):
            mask = padded_masks[b, r]
            if torch.any(mask):
                if len(mask) > 0:
                    masks.append(mask.unsqueeze(0))
            else: break
        # make sure it is uint8
        masks = [mask.to(torch.uint8) for mask in masks]
        unpadded_list.append(torch.stack(masks))
    return unpadded_list

def pair(t):
    return t if isinstance(t, tuple) else (t, t)