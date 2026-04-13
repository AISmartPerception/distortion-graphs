import torch
from torch.utils.data import Dataset
import os, json
from helper import comparison2lbl, distortion2lbl, resize_mask
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import albumentations as A
from pycocotools import mask as cocomask
from itertools import permutations
from PIL import Image
from copy import deepcopy
from pandabench_idx import easy, medium

class PandaBenchLoader(Dataset):
    """
        The Distortion Graph dataset loading class for loading and processing image pairs with associated ground truth (GT) data. 
        The dataset consists of image pairs, where each image pair represents an "anchor" image and a "target" image, along with a corresponding ground truth 
        that compares the quality and characteristics of the images.

        Parameters:
            imgdir (str): Directory containing the image data, organized by degradation type and mode.
            dict_of_stats (dict): A dictionary where keys are degradation types and values are the paths to JSON files with image statistics.
            mode (str): The mode to load images in ('train', 'test', or 'val').
            inf_option (str): The test inference option from PandaBench ("easy", "medium", "hard")

        Methods:
            __len__(self): Returns the number of image pairs in the dataset.
            __getitem__(self, idx): Returns a tuple of a randomly selected anchor image, target image, and their corresponding ground truth.
            make_gt(self, anchor_stats, target_stats): Generates the ground truth comparison between the anchor and target images based on degradation severity, 
                                                       scene score, and region information.
    """
    def __init__(self, imgdir, dict_of_stats, 
                 resize_shape, mode, inf_option="hard"):
        self.resize_shape = resize_shape
        self.mode = mode
        self.imgdir = imgdir
        self.inf_option = inf_option

        if self.mode == "train":
            self.do_aug = True
        else:
            self.do_aug = False
        self.degradations = [x for x in os.listdir(imgdir) if os.path.isdir(imgdir+x)]
        # remove both the folders that are not degradations
        self.degradations.remove("stats")
        self.degradations.remove("depth")

        # replace for each mode as required.
        self.dict_of_stats = dict_of_stats.copy()
        for k, v in self.dict_of_stats.items():
            self.dict_of_stats[k] = v.replace("train", self.mode)

        self.set_of_degradations = self.degradations.copy()
        self.set_of_degradations.remove("gt")
        
        # if inf_option is hard, then just load mixed sets.
        if self.mode == "test" and self.inf_option == "hard":
            self.set_of_degradations = ["mixed", "mixed2"]

        self.img_tags = os.listdir(imgdir+"gt/"+self.mode+"/")
        
        self.pairs = []
        for img_tag in self.img_tags:
            for anchor_deg, target_deg in permutations(self.set_of_degradations, 2):
                self.pairs.append((anchor_deg, target_deg, img_tag))

        # maintain a cache for faster processing
        if self.mode == "test" and self.inf_option == "easy":
            self.pairs = easy # load the easy idx from pandabench_idx
        if self.mode == "test" and self.inf_option == "medium":
            self.pairs = medium # load the medium idx from pandabench_idx

        self.cached_stats = {}
        for deg in self.degradations:
            if deg != "gt":
                with open(self.dict_of_stats[deg], "r") as f:
                    if deg not in list(self.cached_stats.keys()):
                        self.cached_stats[deg] = json.load(f)
        
        # augmentation
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # dinov2 extracts better features like this.
        pixel_val = 1.0 # does not do proper norm then

        if self.do_aug:
            self.aug_transform = A.Compose([
                A.Resize(self.resize_shape, self.resize_shape),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean, std=std,
                            max_pixel_value=pixel_val)
            ],  seed=42,
                additional_targets={'target': 'image',
                                   'anchor_mask': 'masks',
                                   'target_mask': 'masks',
                                   'anchor_bbox': 'bboxes',
                                   'target_bbox': 'bboxes'},
                bbox_params=A.BboxParams(format='coco'))
        else:
            self.aug_transform = A.Compose([
                A.Resize(self.resize_shape, self.resize_shape),
                A.Normalize(mean=mean, std=std,
                            max_pixel_value=pixel_val)
            ],  seed=42,
                additional_targets={'target': 'image',
                                    'anchor_mask': 'masks',
                                    'target_mask': 'masks',
                                    'anchor_bbox': 'bboxes',
                                    'target_bbox': 'bboxes'},
                bbox_params=A.BboxParams(format='coco'))

    def check_gt_region_valid(self, anchor, target):
        if (anchor["segmentation_mask"] is None) or (target["segmentation_mask"] is None):
            return False
        if (anchor["bbox"] is None) or (target["bbox"] is None):
            return False
        return True

    def compare_scene_scores(self, anchor_score, target_score):
        diff = anchor_score - target_score
        abs_diff = abs(diff)
        if abs_diff < 0.1:
            return "same"
        elif 0.1 <= abs_diff < 0.3:
            return "slightly_better" if diff > 0 else "slightly_worse"
        else:  # abs_diff >= 0.3
            return "significantly_better" if diff > 0 else "significantly_worse"
    
    def make_gt(self, anchor_stats, target_stats):
        gts = {"names": [],
               "relations": [],
               "category_id": [],
               "description": [],
               "anchor_bbox": [], # (B, R_n, 4)
               "target_bbox": [], # (B, R_n, 4)
               "severity": [], # (B, R_n+1, 2) -- +1 because of whole image which is prepended
               "distortion": [], # (B, R_n, 2) 
               "comparison": [], # (B, R_n+1) -- +1 because of whole image which is prepended
               "scores": [], # (B, R_n, 2)
               "anchor_seg_masks": [], # (B, R_n, H, W)
               "target_seg_masks": [] # (B, R_n, H, W)
               }

        # scene comparison
        anchor_scene_score = anchor_stats['score_fr']
        target_scene_score = target_stats['score_fr']
        gts_comparison = comparison2lbl(self.compare_scene_scores(anchor_scene_score, target_scene_score))
        gts["comparison"].append(gts_comparison)
        gts["relations"].append(anchor_stats['relations'])
        # region info
        for i, region in enumerate(anchor_stats["regions"]):
            # skipping region is mask missing
            if not self.check_gt_region_valid(region, target_stats["regions"][i]):
                continue
            
            # segmentation masks
            anchor_segmentation_mask = decode_mask_to_binary(region["segmentation_mask"])
            target_segmentation_mask = decode_mask_to_binary(target_stats["regions"][i]["segmentation_mask"])
            # skipping regions with zero mask
            if check_empty_masks(anchor_segmentation_mask, target_segmentation_mask, self.resize_shape):
                continue

            # store masks
            gts["anchor_seg_masks"].append(anchor_segmentation_mask)
            gts["target_seg_masks"].append(target_segmentation_mask)
            # bounding boxes
            gts["anchor_bbox"].append(region["bbox"])
            gts["target_bbox"].append(target_stats["regions"][i]["bbox"])
            gts["names"].append(region["name"])
            gts["description"].append(region.get("description", "")) # does not exist for coco
            gts["category_id"].append(region.get("category_id", 0)) # does not exist for Seagull | so 0
            anchor_sev = region["severity"]
            target_sev = target_stats["regions"][i]["severity"]
            gts["severity"].append([int(anchor_sev), int(target_sev)])
            anchor_region_score = region['score_fr']
            target_region_score = target_stats["regions"][i]['score_fr']
            reg_comparison = comparison2lbl(self.compare_scene_scores(anchor_region_score, 
                                                                      target_region_score))
            gts["comparison"].append(reg_comparison)

            anchor_distortion = region['degradation']
            target_distortion = target_stats["regions"][i]['degradation']
            gts["distortion"].append([distortion2lbl(anchor_distortion), distortion2lbl(target_distortion)])
            gts["scores"].append([anchor_region_score, target_region_score])
        return gts

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # anchor is imageA, target is imageB
        # the GT compares imageA and imageB
        # the graph follows <imageA, C, imageB> triplets
        anchor_deg, target_deg, img_tag = self.pairs[idx]
        anchor_img = self.imgdir + anchor_deg + "/" + self.mode + "/" + img_tag
        target_img = self.imgdir + target_deg + "/" + self.mode + "/" + img_tag
        t_img_tag = img_tag

        # Getting anchor and target image
        anchor = np.array(Image.open(anchor_img).convert('RGB'))
        target = np.array(Image.open(target_img).convert('RGB'))

        # Getting scene and region gt data
        anchor_stats = self.cached_stats[anchor_deg][img_tag]
        target_stats = self.cached_stats[target_deg][t_img_tag]
        gt_stats = self.make_gt(anchor_stats, target_stats)
        
        # Converting lists of region info to np
        gt_stats["anchor_deg"] = anchor_deg
        gt_stats["target_deg"] = target_deg
        gt_stats["img_tag"] = img_tag
        gt_stats["anchor"] = anchor
        gt_stats["target"] = target
        gt_stats["anchor_seg_masks"] = np.array(gt_stats["anchor_seg_masks"])
        gt_stats["target_seg_masks"] = np.array(gt_stats["target_seg_masks"])
        gt_stats["anchor_bbox"] = np.array(gt_stats["anchor_bbox"])
        gt_stats["target_bbox"] = np.array(gt_stats["target_bbox"])

        # Add copies of original data
        gt_stats["orig_anchor"] = deepcopy(gt_stats["anchor"])
        gt_stats["orig_target"] = deepcopy(gt_stats["target"])
        gt_stats["orig_anchor_seg_masks"] = deepcopy(gt_stats["anchor_seg_masks"])
        gt_stats["orig_target_seg_masks"] = deepcopy(gt_stats["target_seg_masks"])
        gt_stats["orig_anchor_bbox"] = deepcopy(gt_stats["anchor_bbox"])
        gt_stats["orig_target_bbox"] = deepcopy(gt_stats["target_bbox"])

        # store data class
        anchor_is_mixed = anchor_deg in {"mixed", "mixed2"}
        target_is_mixed = target_deg in {"mixed", "mixed2"}

        if not anchor_is_mixed and not target_is_mixed:
            data_category = "dist2dist"
        elif anchor_is_mixed and target_is_mixed:
            data_category = "mixed2mixed"
        else:
            data_category = "dist2mixed"

        gt_stats["data_category"] = data_category

        if len(gt_stats["anchor_seg_masks"]) != 0 and len(gt_stats["target_seg_masks"]) != 0:
            augmented = self.aug_transform(image=gt_stats["anchor"], 
                                           target=gt_stats["target"],
                                           anchor_mask=gt_stats["anchor_seg_masks"],
                                           target_mask=gt_stats["target_seg_masks"],
                                           anchor_bbox=gt_stats["anchor_bbox"],
                                           target_bbox=gt_stats["target_bbox"])
            
            gt_stats["anchor"] = torch.from_numpy(augmented['image']).permute(2, 0, 1).float()
            gt_stats["target"] = torch.from_numpy(augmented['target']).permute(2, 0, 1).float()
            gt_stats["anchor_seg_masks"] = torch.from_numpy(augmented['anchor_mask']).float()
            gt_stats["target_seg_masks"] = torch.from_numpy(augmented['target_mask']).float()
            gt_stats["anchor_bbox"] = torch.from_numpy(augmented['anchor_bbox'])
            gt_stats["target_bbox"] = torch.from_numpy(augmented['target_bbox'])

        return gt_stats

def check_empty_masks(maskA, maskT, resize_shape):
    # checking empty masks after resize
    tempA = resize_mask(torch.from_numpy(maskA).unsqueeze(0), resize_shape, resize_shape).squeeze(0)
    tempT = resize_mask(torch.from_numpy(maskT).unsqueeze(0), resize_shape, resize_shape).squeeze(0)
    anchor_zero_mask_flags = (tempA == 0).all().item()  # shape: (n,)
    target_zero_mask_flags = (tempT == 0).all().item()  # shape: (n,)
    if anchor_zero_mask_flags > 0 or target_zero_mask_flags > 0:
        return True
    return False

def decode_mask_to_binary(rle_mask):
    compressed_rle = cocomask.frPyObjects(rle_mask, 
                                          rle_mask.get('size')[0], 
                                          rle_mask.get('size')[1])
    mask = cocomask.decode(compressed_rle)
    return mask

def pad_masks(mask_list, max_height, max_width):
    # pad the masks to make sure they are consistent and can be made tensors
    max_regions = max(len(masks) for masks in mask_list)
    padded_masks = []
    for masks in mask_list:
        if not isinstance(masks, (list, tuple)):
            masks = [masks]
        if len(masks[0]) == 0: # add a dummy mask
            dummy_mask = torch.zeros((1, max_height, max_width), dtype=torch.uint8).unsqueeze(0) 
            masks.extend([dummy_mask] * max_regions)
        else:
            pad_count = max_regions - masks[0].shape[0]
            if pad_count > 0:
                pad_masks = [torch.zeros(masks[0].shape[1], 
                                        masks[0].shape[2],
                                        masks[0].shape[3],
                                    dtype=torch.uint8).unsqueeze(0) 
                        for _ in range(pad_count)]
                masks.extend(pad_masks)
        padded_masks.append(torch.concat(masks))
    return torch.stack(padded_masks).squeeze(2)  # (B, R_n, H, W)

def pandabench_train_collate_fn(batch, h, w):
    """
    Processes a batch of images and returns a dict with various tensors, some are padded to maintain tensor shape.
    Returns:
        dict: A dictionary containing the processed batch with the following keys:
            - "anchor": A tensor of shape (B, C, H, W) representing the batch of anchor images.
            - "target": A tensor of shape (B, C, H, W) representing the batch of target images.
            - "anchor_bbox": A tensor of shape (B, R_n, 4) representing the resized bounding boxes for anchors.
            - "target_bbox": A tensor of shape (B, R_n, 4) representing the resized bounding boxes for targets.
            - "severity": A tensor of shape (B, R_n, 2) representing the severity of the regions in each image.
            - "distortion": A tensor of shape (B, R_n, 2) representing the distortion of the regions in each image.
            - "comparison": A tensor of shape (B, R_n) representing the comparison values for each region.
            - "anchor_mask": A tensor of shape (B, R_n, H, W) representing the batch of masks for anchor images.
            - "target_mask": A tensor of shape (B, R_n, H, W) representing the batch of masks for target images.
    """
    mask_valid_indices = [i for i,x in enumerate(batch) if len(x["anchor_seg_masks"]) != 0]
    bbox_valid_indices = [i for i,x in enumerate(batch) if len(x["anchor_bbox"]) != 0]
    assert mask_valid_indices == bbox_valid_indices, f"mask_valid_indices: {mask_valid_indices} | bbox_valid_indices: {bbox_valid_indices}"
    batch = [batch[i] for i in mask_valid_indices] 
    
    # Unpack all keys at once using list comprehensions
    anchor_imgs = torch.stack([x["anchor"] for x in batch])
    target_imgs = torch.stack([x["target"] for x in batch])
    r_severities = [torch.tensor(x["severity"]) for x in batch]
    r_anchor_bboxes = [x["anchor_bbox"] for x in batch]
    r_target_bboxes = [x["target_bbox"] for x in batch]
    r_comparisons = [torch.tensor(x["comparison"]) for x in batch]
    r_distortions = [torch.tensor(x["distortion"]) for x in batch]
    r_scores = [torch.tensor(x["scores"]) for x in batch]
    r_anchor_mask = [x["anchor_seg_masks"].unsqueeze(1) for x in batch]
    r_target_mask = [x["target_seg_masks"].unsqueeze(1) for x in batch]

    # tensorify with padding
    padded_severities = pad_sequence(r_severities, padding_value=-1, batch_first=True)
    padded_anchor_bboxes = pad_sequence(r_anchor_bboxes, padding_value=-1, batch_first=True)
    padded_target_bboxes = pad_sequence(r_target_bboxes, padding_value=-1, batch_first=True)
    padded_comparisons = pad_sequence(r_comparisons, padding_value=-1, batch_first=True)
    padded_distortions = pad_sequence(r_distortions, padding_value=-1, batch_first=True)
    padded_scores = pad_sequence(r_scores, padding_value=-1, batch_first=True)
    padded_anchor_masks = pad_masks(r_anchor_mask, h, w)
    padded_target_masks = pad_masks(r_target_mask, h, w)

    # finding valid indices: non-padded regions
    region_mask_flags = (padded_anchor_masks.abs().sum(dim=(2, 3)) > 0).reshape(-1) # (b*r,)

    return {
        "anchor": anchor_imgs, # (B, C, H, W)
        "target": target_imgs, # (B, C, H, W)
        "anchor_bbox": padded_anchor_bboxes, # (B, R_n, 4)
        "target_bbox": padded_target_bboxes, # (B, R_n, 4)
        "severity": padded_severities, # (B, R_n+1, 2) -- +1 because of whole image which is prepended
        "distortion": padded_distortions, # (B, R_n, 2) 
        "comparison": padded_comparisons, # (B, R_n+1) -- +1 because of whole image which is prepended
        "scores": padded_scores, # (B, R_n, 2)
        "anchor_seg_masks": padded_anchor_masks, # (B, R_n, H, W)
        "target_seg_masks": padded_target_masks, # (B, R_n, H, W)
        "region_mask_flags": region_mask_flags # (B*R_n, )
    }

def pandabench_test_collate_fn(batch, h, w):
    """
    Processes a batch of images of size 1 and returns a dict with various tensors, some are padded to maintain tensor shape.
    Returns:
        dict: A dictionary containing the processed batch with the following keys:
            - "anchor": A tensor of shape (B, C, H, W) representing the batch of anchor images.
            - "target": A tensor of shape (B, C, H, W) representing the batch of target images.
            - "anchor_bbox": A tensor of shape (B, R_n, 4) representing the resized bounding boxes for anchors.
            - "target_bbox": A tensor of shape (B, R_n, 4) representing the resized bounding boxes for targets.
            - "severity": A tensor of shape (B, R_n, 2) representing the severity of the regions in each image.
            - "distortion": A tensor of shape (B, R_n, 2) representing the distortion of the regions in each image.
            - "comparison": A tensor of shape (B, R_n) representing the comparison values for each region.
            - "anchor_mask": A tensor of shape (B, R_n, H, W) representing the batch of masks for anchor images.
            - "target_mask": A tensor of shape (B, R_n, H, W) representing the batch of masks for target images.
    """
    mask_valid_indices = [i for i,x in enumerate(batch) if len(x["anchor_seg_masks"]) != 0]
    bbox_valid_indices = [i for i,x in enumerate(batch) if len(x["anchor_bbox"]) != 0]
    assert mask_valid_indices == bbox_valid_indices, f"mask_valid_indices: {mask_valid_indices} | bbox_valid_indices: {bbox_valid_indices}"
    batch = [batch[i] for i in mask_valid_indices]
    
    # Unpack all keys at once using list comprehensions
    anchor_imgs = torch.stack([x["anchor"] for x in batch])
    target_imgs = torch.stack([x["target"] for x in batch])

    orig_anchor_imgs = torch.stack([torch.tensor(x["orig_anchor"]) for x in batch])
    orig_target_imgs = torch.stack([torch.tensor(x["orig_target"]) for x in batch])

    r_severities = [torch.tensor(x["severity"]) for x in batch]
    r_anchor_bboxes = [x["anchor_bbox"] for x in batch]
    r_target_bboxes = [x["target_bbox"] for x in batch]
    r_comparisons = [torch.tensor(x["comparison"]) for x in batch]
    r_distortions = [torch.tensor(x["distortion"]) for x in batch]
    r_scores = [torch.tensor(x["scores"]) for x in batch]
    r_anchor_mask = [x["anchor_seg_masks"].unsqueeze(1) for x in batch]
    r_target_mask = [x["target_seg_masks"].unsqueeze(1) for x in batch]
    
    orig_anchor_mask = [x["orig_anchor_seg_masks"] for x in batch]
    orig_target_mask = [x["orig_target_seg_masks"] for x in batch]

    orig_anchor_bbox = [x["orig_anchor_bbox"] for x in batch]
    orig_target_bbox = [x["orig_target_bbox"] for x in batch]
    names = [x["names"] for x in batch]
    relations = [x["relations"] for x in batch] # scene graph relations
    category_ids = [x["category_id"] for x in batch] # scene graph category ids
    data_category = [x["data_category"] for x in batch] # category of the data
    anchor_degs = [x["anchor_deg"] for x in batch]
    target_degs = [x["target_deg"] for x in batch]
    img_tags = [x["img_tag"] for x in batch]
    description = [x["description"] for x in batch]
    
    # tensorify with padding
    padded_severities = pad_sequence(r_severities, padding_value=-1, batch_first=True)
    padded_anchor_bboxes = pad_sequence(r_anchor_bboxes, padding_value=-1, batch_first=True)
    padded_target_bboxes = pad_sequence(r_target_bboxes, padding_value=-1, batch_first=True)
    padded_comparisons = pad_sequence(r_comparisons, padding_value=-1, batch_first=True)
    padded_distortions = pad_sequence(r_distortions, padding_value=-1, batch_first=True)
    padded_scores = pad_sequence(r_scores, padding_value=-1, batch_first=True)
    padded_anchor_masks = pad_masks(r_anchor_mask, h, w)
    padded_target_masks = pad_masks(r_target_mask, h, w)
    
    # finding valid indices: non-padded regions
    region_mask_flags = (padded_anchor_masks.abs().sum(dim=(2, 3)) > 0).reshape(-1) # (b*r,)

    return {
        "anchor_degs": anchor_degs, # List(B)
        "target_degs": target_degs, # List(B)
        "img_tags": img_tags, # List(B)
        "description": description, # List(R)
        "names": names, # List(R)
        "relations": relations, # List(R)
        "category_ids": category_ids, # List(R)
        "data_category": data_category,
        "anchor": anchor_imgs, # (B, C, H, W)
        "target": target_imgs, # (B, C, H, W)
        "orig_anchor": orig_anchor_imgs, # (B, C, H, W)
        "orig_target": orig_target_imgs, # (B, C, H, W)
        "anchor_bbox": padded_anchor_bboxes, # (B, R_n, 4)
        "target_bbox": padded_target_bboxes, # (B, R_n, 4)
        "orig_anchor_bbox": orig_anchor_bbox, # (B, R_n, 4)
        "orig_target_bbox": orig_target_bbox, # (B, R_n, 4)
        "severity": padded_severities, # (B, R_n, 2)
        "distortion": padded_distortions, # (B, R_n, 2) 
        "comparison": padded_comparisons, # (B, R_n)
        "scores": padded_scores, # (B, R_n, 2)
        "anchor_seg_masks": padded_anchor_masks, # (B, R_n, H, W)
        "target_seg_masks": padded_target_masks, # (B, R_n, H, W)
        "orig_anchor_seg_masks": orig_anchor_mask, # (B, R_n, H, W)
        "orig_target_seg_masks": orig_target_mask, # (B, R_n, H, W)
        "region_mask_flags": region_mask_flags # (B*R_n, )
    }