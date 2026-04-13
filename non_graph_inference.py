import torch
from pandadg import PandaDG
from loaddata import PandaBenchLoader, pandabench_test_collate_fn
from helper import loadconfig
from torch.utils.data import DataLoader
import argparse
from train import collate_accuracy
from functools import partial
import json, time
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def run_inference(model, test_dataloader, device):
    # these are used for SRCC/PLCC calculation
    a_score_pred_lst = []
    a_score_gt_lst = []
    t_score_pred_lst = []
    t_score_gt_lst = []

    a_dist_pred_lst = []
    a_dist_gt_lst = []
    t_dist_pred_lst = []
    t_dist_gt_lst = []

    a_sev_pred_lst = []
    a_sev_gt_lst = []
    t_sev_pred_lst = []
    t_sev_gt_lst = []

    comp_pred_lst = []
    comp_gt_lst = []

    for batch in tqdm(test_dataloader):
        # unroll the batch
        names = batch['names'][0]
        anchor_img, target_img = batch['orig_anchor'], batch['orig_target']
        orig_anchor_box, orig_target_box = batch['orig_anchor_bbox'], batch['orig_target_bbox']
        imgA, imgB = batch["anchor"], batch["target"]
        severities, distortions, comparisons, scores = batch["severity"], batch["distortion"], batch["comparison"], batch["scores"]
        region_mask_flags = batch["region_mask_flags"]
        
        (imgA, imgB, severities, 
         distortions, comparisons, 
         scores, region_mask_flags) = (imgA.to(device), imgB.to(device),
                                       severities.to(device), distortions.to(device),
                                       comparisons.to(device), scores.to(device), 
                                       region_mask_flags.to(device))
        anchor_masks, target_masks = batch["anchor_seg_masks"], batch["target_seg_masks"]
        anchor_masks, target_masks = anchor_masks.to(device), target_masks.to(device)
        orig_anchor_masks, orig_target_masks = batch["orig_anchor_seg_masks"], batch["orig_target_seg_masks"]
        data_category = batch["data_category"]

        with torch.no_grad():
            preds, _, valid_masks = model(imgA, imgB,
                                          anchor_masks, target_masks,
                                          severities, distortions,
                                          comparisons, scores, 
                                          region_mask_flags)
        # compute per-data accuracy
        gts = [comparisons, distortions, severities, scores]
        _, pred_gt_dct = collate_accuracy(preds, gts, valid_masks)

        # fetch relationships
        comp_pred = pred_gt_dct["comparison_masked_preds"]
        comp_gts = pred_gt_dct["comparison_masked_gts"]

        comp_pred_lst.append(comp_pred.detach().cpu().numpy())
        comp_gt_lst.append(comp_gts.detach().cpu().numpy())

        # fetch attributes
        # three attributes per node related to distortion
        a_dist_masked_preds = pred_gt_dct["a_dist_masked_preds"]
        a_dist_masked_gts = pred_gt_dct["a_dist_masked_gts"]
        t_dist_masked_preds = pred_gt_dct["t_dist_masked_preds"]
        t_dist_masked_gts = pred_gt_dct["t_dist_masked_gts"]

        a_dist_pred_lst.append(a_dist_masked_preds.detach().cpu().numpy())
        a_dist_gt_lst.append(a_dist_masked_gts.detach().cpu().numpy())
        t_dist_pred_lst.append(t_dist_masked_preds.detach().cpu().numpy())
        t_dist_gt_lst.append(t_dist_masked_gts.detach().cpu().numpy())

        a_sev_masked_preds = pred_gt_dct["a_sev_masked_preds"]
        a_sev_masked_gts = pred_gt_dct["a_sev_masked_gts"]
        t_sev_masked_preds = pred_gt_dct["t_sev_masked_preds"]
        t_sev_masked_gts = pred_gt_dct["t_sev_masked_gts"]

        a_sev_pred_lst.append(a_sev_masked_preds.detach().cpu().numpy())
        a_sev_gt_lst.append(a_sev_masked_gts.detach().cpu().numpy())
        t_sev_pred_lst.append(t_sev_masked_preds.detach().cpu().numpy())
        t_sev_gt_lst.append(t_sev_masked_gts.detach().cpu().numpy())

        a_score_masked_preds = pred_gt_dct["a_score_masked_preds"]
        a_score_masked_gts = pred_gt_dct["a_score_masked_gts"]
        t_score_masked_preds = pred_gt_dct["t_score_masked_preds"]
        t_score_masked_gts = pred_gt_dct["t_score_masked_gts"]

        # keep this separate for SRCC/PLCC
        a_score_pred_lst.append(a_score_masked_preds.detach().cpu().numpy())
        a_score_gt_lst.append(a_score_masked_gts.detach().cpu().numpy())
        t_score_pred_lst.append(t_score_masked_preds.detach().cpu().numpy())
        t_score_gt_lst.append(t_score_masked_gts.detach().cpu().numpy())

    # compute dist acc/precision/recall/f1
    (a_dist_acc, a_dist_precision, a_dist_recall, a_dist_f1,
    t_dist_acc, t_dist_precision, t_dist_recall, t_dist_f1) = compute_metrics(a_dist_pred_lst, a_dist_gt_lst, 
                                                                              t_dist_pred_lst, t_dist_gt_lst)
    (a_sev_acc, a_sev_precision, a_sev_recall, a_sev_f1,
    t_sev_acc, t_sev_precision, t_sev_recall, t_sev_f1) = compute_metrics(a_sev_pred_lst, a_sev_gt_lst, 
                                                                          t_sev_pred_lst, t_sev_gt_lst)
    comp_acc, comp_precision, comp_recall, comp_f1, *_ = compute_metrics(comp_pred_lst, comp_gt_lst, None, None) # no target

    # compute SRCC/PLCC
    anchor_score_pred_flat = flatten(a_score_pred_lst)
    anchor_score_gt_flat = flatten(a_score_gt_lst)
    target_score_pred_flat = flatten(t_score_pred_lst)
    target_score_gt_flat = flatten(t_score_gt_lst)
    a_srcc, _ = spearmanr(anchor_score_pred_flat, anchor_score_gt_flat)
    a_plcc, _ = pearsonr(anchor_score_pred_flat, anchor_score_gt_flat)
    t_srcc, _ = spearmanr(target_score_pred_flat, target_score_gt_flat)
    t_plcc, _ = pearsonr(target_score_pred_flat, target_score_gt_flat)
    
    # print the report
    print(f"""
        --  Distortion --  
        Accuracy - A: {round(a_dist_acc, 2)} | T: {round(t_dist_acc, 2)} -- Avg: {round((a_dist_acc+t_dist_acc)/2, 2)}
        Precision - A: {round(a_dist_precision, 2)} | T: {round(t_dist_precision, 2)} -- Avg: {round((a_dist_precision+t_dist_precision)/2, 2)}
        Recall - A: {round(a_dist_recall, 2)} | T: {round(t_dist_recall, 2)} -- Avg: {round((a_dist_recall+t_dist_recall)/2, 2)}
        F1 - A: {round(a_dist_f1, 2)} | T: {round(t_dist_f1, 2)} -- Avg: {round((a_dist_f1+t_dist_f1)/2, 2)}
    """)
    print(f"""
        --  Severity --  
        Accuracy - A: {round(a_sev_acc, 2)} | T: {round(t_sev_acc, 2)} -- Avg: {round((a_sev_acc+t_sev_acc)/2, 2)}
        Precision - A: {round(a_sev_precision, 2)} | T: {round(t_sev_precision, 2)} -- Avg: {round((a_sev_precision+t_sev_precision)/2, 2)}
        Recall - A: {round(a_sev_recall, 2)} | T: {round(t_sev_recall, 2)} -- Avg: {round((a_sev_recall+t_sev_recall)/2, 2)}
        F1 - A: {round(a_sev_f1, 2)} | T: {round(t_sev_f1, 2)} -- Avg: {round((a_sev_f1+t_sev_f1)/2, 2)}
    """)
    print(f"""
        --  Comparison --  
        Accuracy - {round(comp_acc, 2)}
        Precision - {round(comp_precision, 2)}
        Recall - {round(comp_recall, 2)}
        F1 - {round(comp_f1, 2)}
    """)
    print(f"""
        --  Score (SRCC/PLCC) --  
        SRCC - A: {round(a_srcc, 2)} | T: {round(t_srcc, 2)} -- Avg: {round((a_srcc+t_srcc)/2, 2)}
        PLCC - A: {round(a_plcc, 2)} | T: {round(t_plcc, 2)} -- Avg: {round((a_plcc+t_plcc)/2, 2)}
    """)

def flatten(lst_of_lsts):
    return [x for xs in lst_of_lsts for x in xs]

def compute_metrics(anchor_pred, anchor_gt, 
                    target_pred, target_gt):
    
    anchor_pred_flat = flatten(anchor_pred)
    anchor_gt_flat = flatten(anchor_gt)
    a_precision = precision_score(anchor_gt_flat, anchor_pred_flat, average="macro")
    a_recall = recall_score(anchor_gt_flat, anchor_pred_flat, average="macro")
    a_f1 = f1_score(anchor_gt_flat, anchor_pred_flat, average="macro")
    a_acc = accuracy_score(anchor_gt_flat, anchor_pred_flat)

    if target_pred is not None and target_gt is not None:
        target_pred_flat = flatten(target_pred)
        target_gt_flat = flatten(target_gt)
        t_precision = precision_score(target_gt_flat, target_pred_flat, average="macro")
        t_recall = recall_score(target_gt_flat, target_pred_flat, average="macro")
        t_f1 = f1_score(target_gt_flat, target_pred_flat, average="macro")
        t_acc = accuracy_score(target_gt_flat, target_pred_flat)
    else:
        t_acc, t_precision, t_recall, t_f1 = None, None, None, None

    return (a_acc, a_precision, a_recall, a_f1,
            t_acc, t_precision, t_recall, t_f1)

def main():
    parser = argparse.ArgumentParser(description="DistortionGraphs!")
    parser.add_argument('--configpath', type=str, help='Config Path.')
    args = parser.parse_args()
    
    # read config and loggers
    config = loadconfig(args.configpath)
    test_pandabench = PandaBenchLoader(config["general"]["datapath"],
                                    config["general"]["stats"],
                                    config["general"]["resize_shape"],
                                    mode="test", 
                                    inf_option=config["inference"]["inf_mode"])
    h = w = config['general']['resize_shape']
    test_dataloader = DataLoader(test_pandabench,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=partial(pandabench_test_collate_fn, h=h, w=w))
    print(f"Total Images to Process: {len(test_dataloader)}")
    
    # load the model
    device_no = config["general"]["device"]
    device = torch.device("cuda:{}".format(device_no) if torch.cuda.is_available() else "cpu")
    model = PandaDG(config, device)

    ckpt_path = config['inference'].get('ckpt', None)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Model Loaded!")
        model = model.to(device)
        model.eval() # put in eval mode
    else:
        raise ValueError(f"No ckpt path defined.")

    run_inference(model, test_dataloader, device)

if __name__ == "__main__":
    main()