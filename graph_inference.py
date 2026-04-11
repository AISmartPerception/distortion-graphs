import torch
from pandadg import PandaDG
from loaddata import PandaBenchLoader, dgbench_test_collate_fn
from helper import loadconfig, lbl2comparison, lbl2distortion, lbl2sev
from torch.utils.data import DataLoader
import argparse
from train import collate_accuracy
from functools import partial
import json, os
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
Here is the format of the distortion graph in json
"objects": [
        # an object (cat) with id 0 is in image 1 
        {"id":0, "name": "cat", "image": 1}
    ],
"attributes": [
        # the attribute is fluffly for object id 0 (which is cat) and is in the image id 1
        {"attribute": "fluffy", "object": 0, "image": 1}
    ],
"relationships": [
    # the cat is on some object 3 in image 1
    {"predicate": "on", "object": 3, "subject": 0, "image": 1},
],
"art": [
    # the object 0 is better than subject 0
    # object 0 belongs to image 1
    # subject 0 belongs to image 2
    {"predicate": "better than", "object": 0, "subject": 0},
]
"""
def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def run_inference(model, test_dataloader,
                  device, name_of_exp,
                  batchsize):

    psg_json = load_json("data/psg/psg_annots/psg.json")
    predicate_classes = psg_json['predicate_classes']

    img_id = 0
    for batch in tqdm(test_dataloader):
        img_id += 1
        # predicted graph
        distortion_graph = {
            "objects": [],
            "attributes": [],
            "relationships": [],
            "art": []
        }
        # ground truth graph
        distortion_graph_gt = {
            "objects": [],
            "attributes": [],
            "relationships": [],
            "art": []
        }

        # unroll the batch
        names = batch['names'][0]
        relations = batch["relations"][0]
        category_ids = batch["category_ids"][0]
        anchor_deg = batch["anchor_degs"][0]
        target_deg = batch["target_degs"][0]
        img_tag = batch["img_tags"][0]
        description = batch["description"][0]
        count = 0

        anchor_img, target_img = batch['orig_anchor'], batch['orig_target']
        orig_anchor_box, orig_target_box = batch['orig_anchor_bbox'], batch['orig_target_bbox']
        imgA, imgB = batch["anchor"], batch["target"]
        imgA_bbs, imgB_bbs = batch["anchor_bbox"], batch["target_bbox"]
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

        # fetch attributes
        # three attributes per node related to distortion
        a_dist_masked_preds = pred_gt_dct["a_dist_masked_preds"]
        a_dist_masked_gts = pred_gt_dct["a_dist_masked_gts"]
        t_dist_masked_preds = pred_gt_dct["t_dist_masked_preds"]
        t_dist_masked_gts = pred_gt_dct["t_dist_masked_gts"]

        a_sev_masked_preds = pred_gt_dct["a_sev_masked_preds"]
        a_sev_masked_gts = pred_gt_dct["a_sev_masked_gts"]
        t_sev_masked_preds = pred_gt_dct["t_sev_masked_preds"]
        t_sev_masked_gts = pred_gt_dct["t_sev_masked_gts"]

        a_score_masked_preds = pred_gt_dct["a_score_masked_preds"]
        a_score_masked_gts = pred_gt_dct["a_score_masked_gts"]
        t_score_masked_preds = pred_gt_dct["t_score_masked_preds"]
        t_score_masked_gts = pred_gt_dct["t_score_masked_gts"]

        comp_gts = comp_gts.reshape(batchsize, -1)
        b, regions = comp_gts.shape
        for region in range(regions):
            region_relationship_pred = lbl2comparison(int(comp_pred.squeeze(0)[region]))
            region_relationship_gt = lbl2comparison(int(comp_gts.squeeze(0)[region]))

            object_name = names[region]
            object_id = region
            object_description = description[region]

            # get the region from image
            region_bounding_box = orig_anchor_box[0][region] # coco format

            distortion_graph["objects"].append({
                "id": str(object_id),
                "name": str(object_name),
                "image": str(1) # 1 is for anchor
            })
            distortion_graph["objects"].append({
                "id": str(object_id+regions), # for target, it is regions+object_id
                "name": str(object_name),
                "image": str(2) # 2 is for target
            })
            # ART (<Anchor, Relation, Target>)
            # the subject and object are same but in different images
            distortion_graph["art"].append({
                "predicate": str(region_relationship_pred), 
                "object": str(object_id), # from the anchor
                "subject": str(object_id+regions) # from the target
            })

            anchor_distortion_pred = lbl2distortion(a_dist_masked_preds.squeeze(0)[region].item())
            anchor_distortion_gt = lbl2distortion(a_dist_masked_gts.squeeze(0)[region].item())
            target_distortion_pred = lbl2distortion(t_dist_masked_preds.squeeze(0)[region].item())
            target_distortion_gt = lbl2distortion(t_dist_masked_gts.squeeze(0)[region].item())
            
            anchor_sev_pred = lbl2sev(a_sev_masked_preds.squeeze(0)[region].item())
            anchor_sev_gt = lbl2sev(a_sev_masked_gts.squeeze(0)[region].item())
            target_sev_pred = lbl2sev(t_sev_masked_preds.squeeze(0)[region].item())
            target_sev_gt = lbl2sev(t_sev_masked_gts.squeeze(0)[region].item())

            a_score_pred = a_score_masked_preds.squeeze(0)[region].item()
            a_score_gt = a_score_masked_gts.squeeze(0)[region].item()
            t_score_pred = t_score_masked_preds.squeeze(0)[region].item()
            t_score_gt = t_score_masked_gts.squeeze(0)[region].item()

            # Distortion Attributes (across images)
            distortion_graph["attributes"].append({
                "attribute": str(anchor_distortion_pred),
                "object": str(object_id),
                "image": str(1),
            })
            distortion_graph["attributes"].append({
                "attribute": str(target_distortion_pred),
                "object": str(object_id+regions),
                "image": str(2),
            })
            # severity
            distortion_graph["attributes"].append({
                "attribute": str(anchor_sev_pred), 
                "object": str(object_id),
                "image": str(1),
            })
            distortion_graph["attributes"].append({
                "attribute": str(target_sev_pred), 
                "object": str(object_id+regions),
                "image": str(2),
            })
            # scores
            distortion_graph["attributes"].append({
                "attribute": str(round(a_score_pred,4)),
                "object": str(object_id),
                "image": str(1),
            })
            distortion_graph["attributes"].append({
                "attribute": str(round(t_score_pred,4)),
                "object": str(object_id+regions),
                "image": str(2),
            })

            # fetch scene information from this
            # this only works for COCO images (since they have information)
            # for seagull, implement scene graph parser on object_description

            category_id = category_ids[region]
            region_specific_relations = [x for x in relations[0][0] if count in x[:2]]          
            for s_idx, o_idx, rel_id in region_specific_relations:
                # scene relationships
                distortion_graph["relationships"].append({
                        "predicate": predicate_classes[rel_id], 
                        "object": str(o_idx),
                        "subject": str(s_idx),
                        "image": str(1),
                })
                
                distortion_graph["relationships"].append({
                        "predicate": predicate_classes[rel_id],
                        "object": str(o_idx),
                        "subject": str(s_idx),
                        "image": str(2),
                })
            
            count += 1
        
        os.makedirs("inf_graphs/", exist_ok=True)
        graph_name = f"inf_graphs/{img_id}_{img_tag}_{anchor_deg}_{target_deg}_{name_of_exp}.json"
        with open(graph_name, "w") as f:
            json.dump(distortion_graph, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="DistortionGraphs!")
    parser.add_argument('--configpath', type=str, help='Config Path.')
    args = parser.parse_args()
    
    # read config and loggers
    config = loadconfig(args.configpath)
    test_dgbench = PandaBenchLoader(config["general"]["datapath"],
                                    config["general"]["stats"],
                                    config["general"]["resize_shape"],
                                    mode="test", 
                                    inf_option=config["inference"]["inf_mode"])
    h = w = config['general']['resize_shape']
    test_dataloader = DataLoader(test_dgbench,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=partial(dgbench_test_collate_fn, h=h, w=w))
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
    name_of_exp = ckpt_path.split('/')[-2]
    run_inference(model, test_dataloader,
                  device, name_of_exp,
                  batchsize=1)

if __name__ == "__main__":
    main()
