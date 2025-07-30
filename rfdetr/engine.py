# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable

import torch

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor

#结果可视化函数新增导入
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from rfdetr.util import box_ops
import numpy as np

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
    writer=None,
    representative_image_ids=None,
    coco_dataset=None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    saved_representative_images_this_epoch = set()

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                
                # 新增的可视化步骤
                # Save images periodically to check detection results
                if epoch % 5 == 0 and data_iter_step % 100 == 0:
                    try:
                        save_detection_images(outputs, new_targets, new_samples, epoch, coco_dataset=coco_dataset, criterion=criterion)
                    except Exception as e:
                        print(f"Error saving detection image in epoch {epoch}: {e}")

                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )


            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if writer:
            writer.add_scalar('train/loss', loss_value, it)
            writer.add_scalar('train/class_error', loss_dict_reduced["class_error"], it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], it)
            # Log individual losses
            for k, v in loss_dict_reduced_scaled.items():
                writer.add_scalar(f'train/{k}', v, it)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None, writer=None, epoch=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    # For FPS calculation
    forward_pass_times = []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            start_time = time.time()
            outputs = model(samples)
            end_time = time.time()
            # We only measure forward pass time for batch size 1 for a realistic FPS measurement
            if samples.tensors.shape[0] == 1:
                forward_pass_times.append(end_time - start_time)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # Calculate FPS
    if forward_pass_times:
        avg_forward_time = np.mean(forward_pass_times)
        fps = 1 / avg_forward_time
    else:
        fps = 0
    
    print(f"FPS (batch_size=1): {fps:.2f}")
    if writer and epoch is not None:
        writer.add_scalar('eval/FPS', fps, epoch)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['fps'] = fps

    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            stats['mAP@.50-.95'] = stats["coco_eval_bbox"][0]
            stats['mAP@.50'] = stats["coco_eval_bbox"][1]

    if writer and epoch is not None:
        writer.add_scalar('eval/loss', stats['loss'], epoch)
        if 'class_error' in stats:
            writer.add_scalar('eval/class_error', stats['class_error'], epoch)
        
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP@.50-.95', stats['mAP@.50-.95'], epoch)
        else:
            print("Warning: 'mAP@.50-.95' not found in stats. Skipping TensorBoard log.")
            
        if 'mAP@.50' in stats:
            writer.add_scalar('eval/mAP@.50', stats['mAP@.50'], epoch)
        else:
            print("Warning: 'mAP@.50' not found in stats. Skipping TensorBoard log.")

        if 'fps' in stats:
            writer.add_scalar('eval/FPS', stats['fps'], epoch)
        else:
            print("Warning: 'fps' not found in stats. Skipping TensorBoard log.")

    if coco_evaluator and "segm" in postprocessors.keys():
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator

#新增的可视化函数
def denormalize_image(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def save_detection_images(outputs, targets, samples, epoch, output_dir="./result", coco_dataset=None, criterion=None):
    """
    Saves images by drawing matched predicted boxes on top of the original images
    which already contain ground truth boxes.
    """
    # Get image_id and filename
    target = targets[0]
    image_id = target['image_id'].item()
    img_info = coco_dataset.loadImgs(image_id)[0]
    img_filename = img_info['file_name']

    # Load the original image from the visual dataset (which has GT boxes)
    original_image_path = os.path.join("dataset/visual/train", img_filename)
    if not os.path.exists(original_image_path):
        # If not found, try to reconstruct from the tensor
        print(f"Warning: Original image not found at {original_image_path}. Reconstructing from tensor.")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        # Denormalize and convert to PIL image
        denormalized_tensor = denormalize_image(samples.tensors[0].cpu(), mean, std)
        img = F.to_pil_image(denormalized_tensor)
    else:
        img = Image.open(original_image_path).convert('RGB')

    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found. Falling back to default font.")
        font = ImageFont.load_default()

    # Get original size for scaling boxes
    orig_size = target['orig_size']
    img_h, img_w = orig_size.cpu().numpy()

    # Prepare output directory
    img_name_without_ext = os.path.splitext(img_filename)[0]
    img_output_dir = os.path.join(output_dir, img_name_without_ext)
    os.makedirs(img_output_dir, exist_ok=True)
    print(f"Saving image to directory: {img_output_dir}")

    if criterion:
        # Use matcher to find corresponding predicted boxes
        indices = criterion.matcher(outputs, targets)
        pred_idx, _ = indices[0]  # We only need the indices of the predicted boxes

        # --- Draw Matched Predicted Boxes (Red) ---
        pred_boxes = outputs['pred_boxes'][0][pred_idx]
        pred_logits = outputs['pred_logits'][0][pred_idx]
        scores, labels = pred_logits.softmax(-1).max(-1)

        if pred_boxes.shape[0] > 0:
            scaled_pred_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            scaled_pred_boxes = scaled_pred_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=scaled_pred_boxes.device)
            
            for box, label, score in zip(scaled_pred_boxes, labels, scores):
                box = box.cpu().tolist()
                category_id = label.item()
                category_name = coco_dataset.cats[category_id]['name'] if category_id in coco_dataset.cats else f"ID:{category_id}"
                
                draw.rectangle(box, outline="red", width=2)
                
                text = f"{category_name} {score.item():.2f}"
                text_position = (box[0], box[1] - 15 if box[1] - 15 > 0 else box[1])
                
                try:
                    # Use textbbox for modern Pillow versions
                    text_bbox = draw.textbbox(text_position, text, font=font)
                    # Adjust bbox to add some padding
                    padded_bbox = (text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2)
                    draw.rectangle(padded_bbox, fill="red")
                    draw.text(text_position, text, fill="white", font=font)
                except AttributeError:
                    # Fallback for older Pillow versions
                    text_size = draw.textsize(text, font=font)
                    draw.rectangle(
                        (text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1]),
                        fill="red"
                    )
                    draw.text(text_position, text, fill='white', font=font)
    else:
        print("Warning: `criterion` not provided to `save_detection_images`. Cannot determine which boxes are used for loss. Skipping image save.")
        return

    # Save the modified image
    save_path = os.path.join(img_output_dir, f"epoch_{epoch}.png")
    img.save(save_path)
