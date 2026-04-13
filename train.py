import torch
import torch.nn as nn
import logging, argparse, os, random, yaml, time
import numpy as np
from helper import MetricMonitor, loadconfig
from loaddata import PandaBenchLoader, pandabench_train_collate_fn
from torch.utils.data import DataLoader
from pandadg import PandaDG
from tqdm import tqdm
from torchmetrics.regression import MeanAbsoluteError
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
import torch.distributed as dist
import datetime
from scipy.stats import pearsonr, spearmanr
import deepspeed

import warnings
warnings.filterwarnings('ignore')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CUDAPrefetcher():
    def __init__(self, loader, device):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def init_dist(backend='nccl', **kwargs):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    dist.init_process_group(backend=backend,
                            timeout=datetime.timedelta(seconds=8640000),
                            **kwargs)
    rank, world_size = get_dist_info()
    set_seed(42 + rank)
    return rank, world_size, 42 + rank

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_logger(output_dir, name):
    logfile = f"{name}.log"
    logpath = os.path.join(output_dir, logfile)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(logpath, mode='w')]
    )
    logger = logging.getLogger(__name__)
    return logger

def collate_losses(losses, config, per_head=False):
    region_loss, region_dist_loss, severity_loss, score_pred_loss = losses
    weighted_region_loss = config['train']['model']['region_loss_weight']*region_loss
    weighted_region_dist_loss = config['train']['model']['region_distortion_loss_weight']*region_dist_loss
    weighted_sev_loss = config['train']['model']['region_severity_loss_weight']*severity_loss
    weighted_score_loss = config['train']['model']['score_pred_loss_weight']*score_pred_loss
    total_loss = weighted_region_loss + weighted_region_dist_loss + weighted_sev_loss + weighted_score_loss
    if per_head:
        return (weighted_region_loss,
                weighted_region_dist_loss,
                weighted_sev_loss,
                weighted_score_loss,
                total_loss)
    else:
        return total_loss

def recall_at_k(pred, gt, valid_masks, k=2, softmax_dim=1):
    # gt: (b*r,) | pred: (b*r, c) | valid_masks: (b*r,)
    softmax_fn = nn.Softmax(dim=softmax_dim)
    pred_probs = softmax_fn(pred) # (b*r, c)
    topk_scores, topk_indices = pred_probs.topk(k, dim=softmax_dim) # (b*r, k)
    topk_indices_valid = topk_indices[valid_masks] # (b*correct_r, k)
    gt_valid = gt[valid_masks] # (b*correct_r,)
    
    hits = (topk_indices_valid == gt_valid.unsqueeze(1)).any(dim=1).float()
    recall_k = hits.mean().item() if hits.numel() > 0 else 0.0
    return recall_k

def compute_pred_accuracy(pred, gt, valid_masks, softmax_dim=1, 
                          no_mask=False, k=2, do_r_at_k=False):
    # gt: (b, r) | preds: (b*r, c) | valid_masks: (b*r,)
    softmax_fn = nn.Softmax(dim=softmax_dim)
    pred_proba = softmax_fn(pred) # (b*r, c)
    pred_classes = pred_proba.argmax(dim=softmax_dim) # (b*r,)
    
    if no_mask:
        # for scene comparison, there is no need for mask
        masked_preds = pred_classes
        masked_gts = gt
    else:
        gt = gt.reshape(-1) # (b*r,)
        masked_preds = pred_classes[valid_masks] # (b*correct_r,)
        masked_gts = gt[valid_masks] # (b*correct_r,)
    
    correct = (masked_preds == masked_gts).sum().item()
    accuracy = correct / masked_gts.numel()
    recall_k = None
    if do_r_at_k:
        recall_k = recall_at_k(pred, gt, valid_masks, k=k)
    
    return accuracy, masked_preds, masked_gts, recall_k

def compute_srcc_plcc(pred, gt, valid_masks):
    # pred: (b*r,) | gt: (b*r,) | valid_masks: (b*r, )    
    pred_filtered = pred[valid_masks].detach().cpu().numpy() # (b*correct_r,)
    gt_filtered = gt[valid_masks].detach().cpu().numpy() # (b*correct_r,)
    srcc, _ = spearmanr(pred_filtered, gt_filtered)
    plcc, _ = pearsonr(pred_filtered, gt_filtered)
    return float(srcc), float(plcc)

def compute_regression_mae(pred, gt, valid_masks):
    # pred: (b*r, 1) | gt: (b,r) | valid_masks: (b*r, )
    mean_absolute_error = MeanAbsoluteError().to(pred.device)
    pred = pred.squeeze(1) # (b*r,)
    gt = gt.reshape(-1) # (b*r,)
    masked_preds = pred[valid_masks] # (b*correct_r,)
    masked_gts = gt[valid_masks] # (b*correct_r,)
    error = mean_absolute_error(masked_preds, masked_gts)
    srcc, plcc = compute_srcc_plcc(pred, gt, valid_masks)
    return error, masked_preds, masked_gts, srcc, plcc

def collate_accuracy(preds, gts, valid_masks):
    (region_comparison_outputs,
    a_distoriton_pred_outputs, t_distortion_pred_outputs,
    a_sev_preds_outputs, t_sev_preds_outputs,
    a_score_preds_outputs, t_score_preds_outputs) = preds
    comparisons_gt, distortions_gt, severities_gt, scores_gt = gts
    
    preds_gt_masked_dict = {}
    # compute the relation accuracies here
    (comparison_accuracy, comparison_masked_preds, 
     comparison_masked_gts, recall_k) = compute_pred_accuracy(region_comparison_outputs,
                                                              comparisons_gt[:,1:], 
                                                              valid_masks, k=2,
                                                              do_r_at_k=True)
    preds_gt_masked_dict["comparison_masked_preds"] = comparison_masked_preds
    preds_gt_masked_dict["comparison_masked_gts"] = comparison_masked_gts

    # compute the attribubte accuracies/errors here
    (anchor_dist_accuracy, a_dist_masked_preds, 
     a_dist_masked_gts, _) = compute_pred_accuracy(a_distoriton_pred_outputs,
                                                   distortions_gt[:,:,0],
                                                   valid_masks)
    preds_gt_masked_dict["a_dist_masked_preds"] = a_dist_masked_preds
    preds_gt_masked_dict["a_dist_masked_gts"] = a_dist_masked_gts

    (target_dist_accuracy, t_dist_masked_preds,
     t_dist_masked_gts, _) = compute_pred_accuracy(t_distortion_pred_outputs,
                                                   distortions_gt[:,:,1],
                                                   valid_masks)
    preds_gt_masked_dict["t_dist_masked_preds"] = t_dist_masked_preds
    preds_gt_masked_dict["t_dist_masked_gts"] = t_dist_masked_gts
    
    (anchor_sev_accuracy, a_sev_masked_preds, 
     a_sev_masked_gts, a_sev_recall_2) = compute_pred_accuracy(a_sev_preds_outputs,
                                                  severities_gt[:,:,0], 
                                                  valid_masks, k=2,
                                                  do_r_at_k=True)
    preds_gt_masked_dict["a_sev_masked_preds"] = a_sev_masked_preds
    preds_gt_masked_dict["a_sev_masked_gts"] = a_sev_masked_gts
    
    (target_sev_accuracy, t_sev_masked_preds, 
     t_sev_masked_gts, t_sev_recall_2) = compute_pred_accuracy(t_sev_preds_outputs, 
                                                  severities_gt[:,:,1],
                                                  valid_masks, k=2,
                                                  do_r_at_k=True)
    preds_gt_masked_dict["t_sev_masked_preds"] = t_sev_masked_preds
    preds_gt_masked_dict["t_sev_masked_gts"] = t_sev_masked_gts
    
    (anchor_score_pred, a_score_masked_preds, 
    a_score_masked_gts, a_srcc, a_plcc) = compute_regression_mae(a_score_preds_outputs,
                                                                 scores_gt[:,:,0],
                                                                 valid_masks)
    
    preds_gt_masked_dict["a_score_masked_preds"] = a_score_masked_preds
    preds_gt_masked_dict["a_score_masked_gts"] = a_score_masked_gts

    (target_score_pred, t_score_masked_preds, 
    t_score_masked_gts, t_srcc, t_plcc) = compute_regression_mae(t_score_preds_outputs,
                                                                 scores_gt[:,:,1],
                                                                 valid_masks)

    preds_gt_masked_dict["t_score_masked_preds"] = t_score_masked_preds
    preds_gt_masked_dict["t_score_masked_gts"] = t_score_masked_gts
    
    return [comparison_accuracy, anchor_dist_accuracy,
            target_dist_accuracy, anchor_sev_accuracy, 
            target_sev_accuracy, anchor_score_pred,
            target_score_pred, a_srcc, a_plcc,
            t_srcc, t_plcc, recall_k,
            a_sev_recall_2, t_sev_recall_2], preds_gt_masked_dict

def load_model(config, device):
    return PandaDG(config, device)

def trainer(config,
            logger, metric_monitor,
            writer, device,
            train_sampler,
            train_prefetcher,
            val_prefetcher,
            test_prefetcher,
            tqdm_length):
    torch.cuda.set_device(device)
    model = load_model(config, device)
    if dist.get_rank() == 0:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        logger.info(f"Number of Model Parameters: {num_params}M")
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["train"]["learning_rate"]),
                                  weight_decay=float(config["train"]["weight_decay"]))
    # init deepspeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=config["deepspeed_config"],
        model_parameters=model.parameters()
    )
    do_grad_accumulation = config["train"].get("gradient_accumulation", False)
    grad_accumulation_steps = config["train"].get("gradient_accumulation_steps", 0)

    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_time_st = time.time()
        train_sampler.set_epoch(epoch)
        train_prefetcher.reset()
        batch = train_prefetcher.next()
        if dist.get_rank() == 0:
            pbar = tqdm(total=tqdm_length, colour='blue')
        batch_idx = 0
        while batch is not None:
            # unroll the batch
            imgA, imgB = batch["anchor"], batch["target"]
            severities, distortions, comparisons, scores = batch["severity"], batch["distortion"], batch["comparison"], batch["scores"]
            anchor_masks, target_masks = batch["anchor_seg_masks"], batch["target_seg_masks"]
            region_mask_flags = batch["region_mask_flags"]
            _, losses, _ = model_engine(imgA.half(), imgB.half(), 
                                        anchor_masks.half(),
                                        target_masks.half(),
                                        severities.half(),
                                        distortions.half(),
                                        comparisons.half(),
                                        scores.half(),
                                        region_mask_flags)
            total_loss = collate_losses(losses, config,
                                        per_head=False)
            total_loss += 0 * sum(p.sum() for p in model.parameters())
            model_engine.backward(total_loss)
            if not do_grad_accumulation or (batch_idx + 1) % grad_accumulation_steps == 0:
                model_engine.step()

            if dist.get_rank() == 0:
                pbar.set_description(f"Ep. {epoch+1} | Loss: {total_loss.item():3f}")
                pbar.update(1)
            metric_monitor.set_metric("total_loss", total_loss.item())
            batch = train_prefetcher.next()
            batch_idx += 1

        epoch_time_end = time.time()
        epoch_train_loss = metric_monitor.get_specific_metric("total_loss")
        epoch_loss_tensor = torch.tensor(epoch_train_loss).to(device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_tensor /= dist.get_world_size()
        time_taken = epoch_time_end-epoch_time_st
        if dist.get_rank() == 0:
            pbar.close()
            time_remaining_in_training = str(datetime.timedelta(seconds=time_taken*(int(config["train"]["epochs"])-(epoch+1))))
            logger.info(f"[Epoch {epoch+1}/{config['train']['epochs']}] Finished in {time_taken}s "
                        f"| Loss: {epoch_loss_tensor.item()} "
                        f"| Time Remaining: {time_remaining_in_training}")
            if writer is not None:
                writer.add_scalar("train_loss", epoch_loss_tensor.item(), epoch+1)
        if ((epoch+1)%config["train"]["validation_epoch"])==0:
            model.eval()
            evaluate(model_engine, logger,
                    device, epoch,
                    metric_monitor, writer,
                    val_prefetcher,
                    config,
                    mode="val")
            # check for intermediate model saving here
            if dist.get_rank() == 0:
                if ((epoch+1)%config["train"]["save_model_ep"])==0:
                    logger.info(f"Saving Model at Epoch {epoch+1}")
                    save_model(config, epoch, model_engine, 
                               optimizer, epoch_train_loss)

    model.eval()
    evaluate(model_engine, logger,
            device, epoch,
            metric_monitor, 
            writer,
            test_prefetcher,
            config,
            mode="test")
    if dist.get_rank() == 0:
        to_save_loss = metric_monitor.get_specific_metric("total_loss")
        save_model(config, epoch, model_engine, optimizer, to_save_loss, "final")

def save_model(config, epoch, model, optimizer, loss, placeholder=None):
    if placeholder is not None:
        path = config["train"]["save_dir"]+config['exp_name']+"/"+f"{placeholder}_ep_{epoch+1}.pth"
    else:
        path = config["train"]["save_dir"]+config['exp_name']+"/"+f"ep_{epoch+1}.pth"
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, path)

def reduce_metrics_across_gpus(metric_monitor, mode, device):
    world_size = dist.get_world_size()
    metrics = [
        f"{mode}_comparison_accuracy",
        f"{mode}_comparison_recall_2",
        f"{mode}_anchor_dist_accuracy",
        f"{mode}_target_dist_accuracy",
        f"{mode}_anchor_sev_accuracy",
        f"{mode}_target_sev_accuracy",
        f"{mode}_a_sev_recall_2",
        f"{mode}_t_sev_recall_2",
        f"{mode}_anchor_score_mae",
        f"{mode}_target_score_mae",
    ]
    if mode == "val":
        metrics += [f"{mode}_region_loss",  f"{mode}_reg_distortion_loss",
                    f"{mode}_reg_severity_loss", 
                    f"{mode}_mae_loss", f"{mode}_loss"]
    
    for metric_name in metrics:
        value = metric_monitor.get_specific_metric(metric_name)
        if value is None:
            continue
        tensor_val = torch.tensor(value).to(device)
        dist.all_reduce(tensor_val, op=dist.ReduceOp.SUM)
        tensor_val /= world_size
        # clean up batch-level numbers
        metric_monitor.reset_specific_metric(metric_name)
        metric_monitor.set_metric(metric_name,
                                  tensor_val.item(),
                                  reduced=True)

def evaluate(model, logger,
             device, epoch,
             metric_monitor, 
             writer,
             prefetcher,
             config,
             mode="val"):
    # clean up metric monitor here
    # this helps disentangle validation at every step
    metric_monitor.reset(mode)
    with torch.no_grad():
        prefetcher.reset()
        batch = prefetcher.next()
        while batch is not None:
            imgA, imgB = batch["anchor"], batch["target"]
            severities, distortions, comparisons, scores = batch["severity"], batch["distortion"], batch["comparison"], batch["scores"]
            anchor_masks, target_masks = batch["anchor_seg_masks"], batch["target_seg_masks"]
            region_mask_flags = batch["region_mask_flags"]

            preds, losses, valid_masks = model(imgA.half(), imgB.half(), 
                                               anchor_masks.half(),
                                               target_masks.half(),
                                               severities.half(),
                                               distortions.half(),
                                               comparisons.half(),
                                               scores.half(),
                                               region_mask_flags)
            (val_region_loss, val_reg_distortion_loss,
            val_reg_severity_loss, val_mae_loss, val_loss) = collate_losses(losses, config,
                                                                            per_head=True)
            # compute per-data accuracy
            gts = [comparisons, distortions, severities, scores]
            outputs, _ = collate_accuracy(preds, gts, valid_masks)
            (comparison_accuracy, anchor_dist_accuracy, 
             target_dist_accuracy, anchor_sev_accuracy, 
             target_sev_accuracy, anchor_score_mae, 
             target_score_mae, a_srcc, a_plcc, t_srcc, 
             t_plcc, recall_2, a_sev_recall_2, 
             t_sev_recall_2) = outputs
            
            # log the metrics
            if mode == "val":
                metric_monitor.set_metric(f"{mode}_loss", val_loss.item())
                metric_monitor.set_metric(f"{mode}_region_loss", val_region_loss.item())
                metric_monitor.set_metric(f"{mode}_reg_distortion_loss", val_reg_distortion_loss.item())
                metric_monitor.set_metric(f"{mode}_reg_severity_loss", val_reg_severity_loss.item())
                metric_monitor.set_metric(f"{mode}_mae_loss", val_mae_loss.item())
            metric_monitor.set_metric(f"{mode}_comparison_accuracy", comparison_accuracy)
            metric_monitor.set_metric(f"{mode}_anchor_dist_accuracy", anchor_dist_accuracy)
            metric_monitor.set_metric(f"{mode}_target_dist_accuracy", target_dist_accuracy)
            metric_monitor.set_metric(f"{mode}_anchor_sev_accuracy", anchor_sev_accuracy)
            metric_monitor.set_metric(f"{mode}_target_sev_accuracy", target_sev_accuracy)
            metric_monitor.set_metric(f"{mode}_anchor_score_mae", anchor_score_mae)
            metric_monitor.set_metric(f"{mode}_target_score_mae", target_score_mae)
            metric_monitor.set_metric(f"{mode}_comparison_recall_2", recall_2)
            metric_monitor.set_metric(f"{mode}_a_sev_recall_2", a_sev_recall_2)
            metric_monitor.set_metric(f"{mode}_t_sev_recall_2", t_sev_recall_2)
            batch = prefetcher.next()
    
    reduce_metrics_across_gpus(metric_monitor, mode, device)
    # clean up
    del preds, losses, valid_masks
    if dist.get_rank() == 0:
        logger.info(f"Doing {mode.title()} Now!")
        metric_monitor.print_log(logger, epoch, mode)
        if writer is not None:
            metric_monitor.write_to_tensorboard(writer, epoch, mode)

def main(rank, world_size, 
         config, logger, writer):
    train_pandabench = PandaBenchLoader(config["general"]["datapath"],
                                     config["general"]["stats"],
                                     config['general']['resize_shape'],
                                     mode="train")
    val_pandabench = PandaBenchLoader(config["general"]["datapath"],
                                   config["general"]["stats"],
                                   config['general']['resize_shape'],
                                   mode="val") # val on entire PandaSet val set
    test_pandabench = PandaBenchLoader(config["general"]["datapath"],
                                    config["general"]["stats"],
                                    config['general']['resize_shape'],
                                    mode="test", inf_option="hard")
    h = w = config['general']['resize_shape']

    train_sampler = DistributedSampler(train_pandabench,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       drop_last=True)
    train_dataloader = DataLoader(train_pandabench, batch_size=config["train"]["batch_size"],
                                  num_workers=config["train"]["num_workers"],
                                  collate_fn=partial(pandabench_train_collate_fn, h=h, w=w),
                                  sampler=train_sampler,
                                  pin_memory=True)
    train_prefetcher = CUDAPrefetcher(train_dataloader, rank)

    val_sampler = DistributedSampler(val_pandabench,
                                     num_replicas=world_size,
                                     rank=rank)
    val_dataloader = DataLoader(val_pandabench, batch_size=config["train"]["batch_size"],
                                num_workers=config["train"]["num_workers"],
                                collate_fn=partial(pandabench_train_collate_fn, h=h, w=w),
                                sampler=val_sampler,
                                pin_memory=True)
    val_prefetcher = CUDAPrefetcher(val_dataloader, rank)
    
    test_sampler = DistributedSampler(test_pandabench,
                                      num_replicas=world_size,
                                      rank=rank)
    test_dataloader = DataLoader(test_pandabench, batch_size=config["train"]["batch_size"],
                                 shuffle=False, 
                                 num_workers=config["train"]["num_workers"],
                                 collate_fn=partial(pandabench_train_collate_fn, h=h, w=w),
                                 sampler=test_sampler,
                                 pin_memory=True)
    test_prefetcher = CUDAPrefetcher(test_dataloader, rank)
    tqdm_length = len(train_dataloader)

    if dist.get_rank() == 0:
        logger.info("""
            Panoptic Pairwise Distortion Graph (Panda)
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⣿⣿⣿⣿⣿⣆⠀⢀⣀⣀⣤⣤⣤⣦⣦⣤⣤⣄⣀⣀⠀⢠⣾⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⢿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢿⣿⠟⠀⠀⠀⠀⠀⣀⣤⣤⣤⡀⠀⠀⠀⠀⠀⢀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠘⣿⡿⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡟⠀⠀⠀⠀⣠⣾⣿⣿⣟⣿⡇⠀⠀⠀⠀⠀⢸⣿⣿⣻⣿⣿⣦⠀⠀⠀⠀⠸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠁⠀⠀⠀⠀⣿⣿⣿⣿⣿⡟⢠⣶⣾⣿⣿⣷⣤⢽⣿⣿⣿⣿⣿⡇⠀⠀⣀⣤⣿⣷⣴⣶⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣤⣤⣠⣇⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⠀⠘⠻⣿⣿⣿⡿⠋⠀⢹⣿⣿⣿⣿⡇⠀⣿⣿⣿⡏⢹⣿⠉⣿⣿⣿⣷⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣶⣄⠀⠀⠹⣿⣿⠿⠋⠀⢤⣀⢀⣼⡄⠀⣠⠀⠈⠻⣿⣿⠟⠀⢸⣿⣇⣽⣿⠿⠿⠿⣿⣅⣽⣿⡇⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣟⠁⠀⠀⠀⠈⣿⣿⣿⡇⠀⠀⠀⠀⢀
        ⠛⠛⠛⠛⠛⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛
        ⠀⠀⠀⠀⠀⠀⠘⠛⠻⢿⣿⣿⣿⣿⣿⠟⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠈⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
        """)
        logger.info(yaml.dump(config)) # print the config
        logger.info(f"Total Training Data Loaded (divided by batch size): {tqdm_length*world_size*config['train']['batch_size']}")
        logger.info(f"Total Validation Data Loaded (divided by batch size): {len(val_dataloader)*world_size*config['train']['batch_size']}")
        logger.info(f"Total Test Data Loaded (divided by batch size): {len(test_dataloader)*world_size*config['train']['batch_size']}")
        logger.info(f"Steps Per Epoch: {tqdm_length}")

    # setup metric monitors
    metric_monitor = MetricMonitor(["val_comparison_accuracy",
                                    "val_anchor_dist_accuracy", "val_target_dist_accuracy",
                                    "val_anchor_sev_accuracy", "val_target_sev_accuracy",
                                    "val_anchor_score_mae", "val_target_score_mae",
                                    "val_comparison_recall_2",
                                    "val_a_sev_recall_2", "val_t_sev_recall_2",
                                    "total_loss",
                                    "val_region_loss", "val_reg_distortion_loss",
                                    "val_reg_severity_loss", "val_mae_loss", "val_loss",
                                    "test_comparison_accuracy", "test_anchor_dist_accuracy", 
                                    "test_target_dist_accuracy", "test_anchor_sev_accuracy", 
                                    "test_target_sev_accuracy", "test_anchor_score_mae", 
                                    "test_target_score_mae", 
                                    "test_comparison_recall_2", 
                                    "test_a_sev_recall_2", "test_t_sev_recall_2"])
    trainer(config, logger,
            metric_monitor, writer,
            rank, train_sampler,
            train_prefetcher,
            val_prefetcher, 
            test_prefetcher,
            tqdm_length)
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanopticPairwiseDistortionGraph!")
    parser.add_argument('--name', type=str, help='Name of the experiment.')
    parser.add_argument('--configpath', type=str, help='Config Path.')
    parser.add_argument('--master_port', default=29500, type=int, help='Master port for DDP.')
    args = parser.parse_args()
    rank, world_size, seed = init_dist()
    
    # set log level
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["NCCL_DEBUG"]="INFO"
    os.environ["NCCL_DEBUG_SUBSYS"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

    # read config and loggers
    config = loadconfig(args.configpath)
    config['exp_name'] = args.name
    config["deepspeed_config"] = {
        "train_micro_batch_size_per_gpu": config["train"]["batch_size"],
        "gradient_accumulation_steps": config["train"].get("gradient_accumulation_steps", 1),
        "train_batch_size": config["train"]["batch_size"] * world_size,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
        },
        "logging": {"steps_per_print": 100},
    }

    os.makedirs(config["train"]["save_dir"]+config['exp_name'], exist_ok=True)

    if dist.get_rank() == 0:
        logger = create_logger(config["train"]["save_dir"]+config['exp_name'], config['exp_name'])
        writer = SummaryWriter(log_dir=os.path.join("runs", config['exp_name']))
    else:
        logger = None
        writer = None

    world_size = torch.cuda.device_count()
    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank} is using device {device}")
    main(rank, world_size, config, logger, writer)