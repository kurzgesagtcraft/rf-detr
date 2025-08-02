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
from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.util.custom_classes import CUSTOM_CLASSES
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
    print(f"Starting train_one_epoch for epoch {epoch}")
    print(f"LENGTH OF DATA LOADER:", len(data_loader))
    print(f"representative_image_ids: {representative_image_ids}")
    if representative_image_ids is not None and len(representative_image_ids) > 0:
        print(f"First few representative_image_ids: {list(representative_image_ids)[:10]}")
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        print(f"Processing epoch {epoch}, data_iter_step {data_iter_step}")
        # 在每个epoch开始时检查是否应该保存图像
        if data_iter_step == 0:
            print(f"First batch of epoch {epoch}, will attempt to save visualization")
        
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
                # 选择一些代表性的图片在每个epoch都保存其预测结果
                # 检查当前batch中的图像是否在representative_image_ids中
                for i, target in enumerate(new_targets):
                    image_id = target['image_id'].item()
                    print(f"Checking image saving condition for epoch {epoch}, image_id: {image_id}")
                    print(f"representative_image_ids is None: {representative_image_ids is None}")
                    if representative_image_ids is not None:
                        print(f"image_id {image_id} in representative_image_ids: {image_id in representative_image_ids}")
                        if len(representative_image_ids) > 0:
                            print(f"First few representative_image_ids: {list(representative_image_ids)[:10]}")
                    
                    # 如果图像在representative_image_ids中，则保存它
                    if representative_image_ids is None or image_id in representative_image_ids:
                        print(f"Attempting to save detection image for epoch {epoch}, image_id: {image_id}")
                        try:
                            # 为当前目标创建新的targets和samples列表
                            single_target = [target]
                            single_sample_tensors = new_samples.tensors[i:i+1]
                            single_sample_mask = new_samples.mask[i:i+1]
                            single_sample = NestedTensor(single_sample_tensors, single_sample_mask)
                            
                            save_detection_images(outputs, single_target, single_sample, epoch, coco_dataset=coco_dataset, criterion=criterion)
                            print(f"Successfully saved detection image for epoch {epoch}, image_id: {image_id}")
                            # 一旦找到并保存了一个代表性图像，就跳出循环
                            break
                        except Exception as e:
                            print(f"Error saving detection image in epoch {epoch}, image_id {image_id}: {e}")
                            import traceback
                            traceback.print_exc()  # 打印详细的错误信息

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

    # Calculate FPS - 移除重复计算
    if forward_pass_times:
        avg_forward_time = np.mean(forward_pass_times)
        fps = 1 / avg_forward_time
    else:
        fps = 0
    
    print(f"FPS (batch_size=1): {fps:.2f}")
    
    # 确保FPS被正确记录到stats中
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['fps'] = fps
    
    # 确保mAP值被正确提取和记录
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            coco_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()
            stats["coco_eval_bbox"] = coco_stats
            
            # 确保mAP值存在
            if len(coco_stats) >= 12:  # COCO评估有12个指标
                stats['mAP@.50-.95'] = float(coco_stats[0])
                stats['mAP@.50'] = float(coco_stats[1])
                stats['mAP@.75'] = float(coco_stats[2])
                stats['mAP_small'] = float(coco_stats[3])
                stats['mAP_medium'] = float(coco_stats[4])
                stats['mAP_large'] = float(coco_stats[5])
                print(f"mAP@.50-.95: {stats['mAP@.50-.95']:.4f}")
                print(f"mAP@.50: {stats['mAP@.50']:.4f}")
                print(f"mAP@.75: {stats['mAP@.75']:.4f}")
            else:
                print("Warning: coco_eval_bbox stats list is incomplete")
                # 至少设置基本的mAP值
                stats['mAP@.50-.95'] = float(coco_stats[0]) if len(coco_stats) > 0 else 0.0
                stats['mAP@.50'] = float(coco_stats[1]) if len(coco_stats) > 1 else 0.0

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['fps'] = fps

    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            # Ensure mAP values are properly extracted
            if len(stats["coco_eval_bbox"]) > 1:
                stats['mAP@.50-.95'] = stats["coco_eval_bbox"][0]
                stats['mAP@.50'] = stats["coco_eval_bbox"][1]
                print(f"mAP@.50-.95: {stats['mAP@.50-.95']:.4f}")
                print(f"mAP@.50: {stats['mAP@.50']:.4f}")
            else:
                print("Warning: coco_eval_bbox stats list is too short.")

    if writer and epoch is not None:
        # 确保所有重要指标都被记录到TensorBoard
        print(f"Logging metrics to TensorBoard for epoch {epoch}")
        
        # 基本指标
        writer.add_scalar('eval/loss', stats['loss'], epoch)
        if 'class_error' in stats:
            writer.add_scalar('eval/class_error', stats['class_error'], epoch)
        
        # FPS指标 - 确保记录
        fps_value = stats.get('fps', 0.0)
        writer.add_scalar('eval/FPS', fps_value, epoch)
        print(f"Logged FPS to TensorBoard: {fps_value:.2f}")
        
        # mAP指标 - 确保记录
        map_5095 = stats.get('mAP@.50-.95', 0.0)
        map_50 = stats.get('mAP@.50', 0.0)
        map_75 = stats.get('mAP@.75', 0.0)
        
        writer.add_scalar('eval/mAP@.50-.95', map_5095, epoch)
        writer.add_scalar('eval/mAP@.50', map_50, epoch)
        writer.add_scalar('eval/mAP@.75', map_75, epoch)
        
        print(f"Logged mAP metrics to TensorBoard:")
        print(f"  mAP@.50-.95: {map_5095:.4f}")
        print(f"  mAP@.50: {map_50:.4f}")
        print(f"  mAP@.75: {map_75:.4f}")
        
        # 额外的mAP细分指标
        if 'mAP_small' in stats:
            writer.add_scalar('eval/mAP_small', stats['mAP_small'], epoch)
        if 'mAP_medium' in stats:
            writer.add_scalar('eval/mAP_medium', stats['mAP_medium'], epoch)
        if 'mAP_large' in stats:
            writer.add_scalar('eval/mAP_large', stats['mAP_large'], epoch)
        
        # 强制刷新确保写入
        writer.flush()
        print("TensorBoard logs flushed successfully")

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
    print(f"save_detection_images called with epoch: {epoch}")
    
    # Get image_id and filename
    target = targets[0]
    image_id = target['image_id'].item()
    print(f"Processing image_id: {image_id}")
    
    # Check if coco_dataset is valid
    if coco_dataset is None:
        print("Warning: coco_dataset is None, using fallback method")
        # 使用一个简单的回退方法
        img_filename = f"{image_id:012d}.jpg"  # COCO数据集的标准文件名格式
    else:
        try:
            img_info = coco_dataset.loadImgs(image_id)[0]
            img_filename = img_info['file_name']
            print(f"Image filename: {img_filename}")
        except Exception as e:
            print(f"Error getting image info: {e}")
            # 回退到标准的COCO文件名格式
            img_filename = f"{image_id:012d}.jpg"
    
    # Load the original image from the visual dataset (which has GT boxes)
    original_image_path = os.path.join("dataset/visual/train", img_filename)
    print(f"Looking for original image at: {original_image_path}")
    
    # 检查目录是否存在
    visual_train_dir = "dataset/visual/train"
    if not os.path.exists(visual_train_dir):
        print(f"Warning: Visual train directory does not exist: {visual_train_dir}")
        # 列出dataset目录的内容
        if os.path.exists("dataset"):
            print("Contents of dataset directory:")
            for item in os.listdir("dataset"):
                print(f"  {item}")
        else:
            print("Dataset directory does not exist")
    else:
        print(f"Visual train directory exists: {visual_train_dir}")
        # 列出目录中的文件数量
        files = os.listdir(visual_train_dir)
        print(f"Number of files in visual train directory: {len(files)}")
        if len(files) > 0:
            print(f"First few files: {files[:5]}")
    
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

    # Load font with multiple fallback options and better error处理
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        "C:/Windows/Fonts/tahoma.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
    ]
    font_size = 20  # 适中的字体大小
    font = None
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                print(f"Successfully loaded font from {font_path}")
                break
        except (IOError, OSError) as e:
            print(f"Warning: Font not found at {font_path}: {e}")
    
    if font is None:
        print("Warning: No truetype font found. Using default font with larger size.")
        try:
            # 尝试使用Pillow的默认字体，但增大尺寸
            font = ImageFont.load_default()
            font_size = 16
        except:
            # 最后的备选方案
            font = ImageFont.load_default()
            font_size = 12

    # Get original size for scaling boxes
    orig_size = target['orig_size']
    img_h, img_w = orig_size.cpu().numpy()
    print(f"Image dimensions: {img_w}x{img_h}")

    # Prepare output directory
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    img_name_without_ext = os.path.splitext(img_filename)[0]
    img_output_dir = os.path.join(output_dir, img_name_without_ext)
    os.makedirs(img_output_dir, exist_ok=True)
    print(f"Saving image to directory: {img_output_dir}")

    # Count of drawn boxes for debugging
    drawn_boxes = 0

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
            
            # 改进的类别名称获取逻辑 - 优先使用用户自定义类别映射
            category_mapping = {}
            try:
                # 从coco_dataset获取类别映射
                if coco_dataset and hasattr(coco_dataset, 'cats'):
                    for cat_id, cat_info in coco_dataset.cats.items():
                        category_mapping[cat_id] = cat_info['name']
                    print(f"从coco_dataset获取类别: {category_mapping}")
                else:
                    print("coco_dataset为空或没有cats属性")
                
                # 如果coco_dataset没有类别信息，使用用户自定义类别映射
                if not category_mapping:
                    category_mapping = CUSTOM_CLASSES
                    print(f"使用CUSTOM_CLASSES映射: {len(category_mapping)}个类别")
                    print(f"CUSTOM_CLASSES内容: {CUSTOM_CLASSES}")
                else:
                    print(f"使用coco_dataset类别映射: {len(category_mapping)}个类别")
                
            except Exception as e:
                print(f"获取类别映射时出错: {e}, 使用CUSTOM_CLASSES回退方案")
                category_mapping = CUSTOM_CLASSES
                print(f"CUSTOM_CLASSES内容: {CUSTOM_CLASSES}")
            
            # 过滤低置信度预测
            # 使用更合理的阈值来显示预测框
            confidence_threshold = 0.3  # 提高阈值以减少低质量预测框
            
            # 添加调试信息，帮助用户理解为什么有些图片没有预测框
            print(f"Total predicted boxes before filtering: {len(scaled_pred_boxes)}")
            
            for box, label, score in zip(scaled_pred_boxes, labels, scores):
                score_value = score.item()
                
                # 只显示高于阈值的预测框
                if score_value < confidence_threshold:
                    continue
                
                box = box.cpu().tolist()
                category_id = label.item()
                
                # 获取类别名称 - 修复类别索引问题
                # 模型输出的类别索引是从0开始的，其中0通常表示背景类别
                # 用户自定义类别映射使用的是1-based索引（1-10）
                # 但模型输出的类别ID可能直接对应用户自定义类别映射的ID
                if category_id == 0:
                    continue  # 跳过背景类别
                
                # 直接使用模型输出的类别ID作为用户自定义类别映射的ID
                custom_category_id = category_id
                print(f"模型输出的类别ID: {category_id}, 直接使用的ID: {custom_category_id}")
                
                # 检查ID是否在用户自定义类别映射范围内
                max_custom_id = max(CUSTOM_CLASSES.keys()) if CUSTOM_CLASSES else 0
                print(f"用户自定义类别映射的最大ID: {max_custom_id}")
                if custom_category_id > max_custom_id:
                    print(f"警告: 类别ID {custom_category_id} 超出了用户自定义类别映射范围 [1, {max_custom_id}]")
                
                # 使用直接的ID获取类别名称
                category_name = category_mapping.get(custom_category_id, f"Class_{custom_category_id}")
                print(f"使用类别ID {custom_category_id} 映射到类别名称: {category_name}")
                print(f"类别映射内容: {category_mapping}")
                
                # 调试输出
                print(f"类别ID: {category_id}, 映射名称: {category_name}, 置信度: {score_value:.3f}")
                
                # 确保box坐标在图像范围内
                box = [
                    max(0, min(box[0], img_w)),
                    max(0, min(box[1], img_h)),
                    max(0, min(box[2], img_w)),
                    max(0, min(box[3], img_h))
                ]
                
                # 使用统一的颜色和宽度绘制边界框
                box_color = "red"
                box_width = 3
                
                # 绘制边界框
                draw.rectangle(box, outline=box_color, width=box_width)
                
                # 准备文本
                text = f"{category_name}: {score_value:.2f}"
                
                # 计算文本位置 - 确保在图像内
                text_x = max(2, min(box[0] + 2, img_w - 100))
                text_y = max(2, box[1] - font_size - 2)
                
                # 如果文本会超出上边界，放在框内
                if text_y < 2:
                    text_y = max(2, box[1] + 2)
                
                text_position = (text_x, text_y)
                
                # 使用统一的文本颜色
                text_color = "white"
                bg_color = "red"  # 保持背景色一致
                
                # 绘制带背景的文本
                try:
                    # 获取文本边界框
                    if hasattr(draw, 'textbbox'):
                        text_bbox = draw.textbbox(text_position, text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    else:
                        # 旧版本Pillow的备选方案
                        text_width = len(text) * font_size * 0.6
                        text_height = font_size
                        text_bbox = (text_position[0], text_position[1],
                                   text_position[0] + text_width, text_position[1] + text_height)
                    
                    # 调整背景框大小
                    padding = 2
                    bg_box = [
                        text_bbox[0] - padding,
                        text_bbox[1] - padding,
                        text_bbox[2] + padding,
                        text_bbox[3] + padding
                    ]
                    
                    # 确保背景框在图像内
                    bg_box[0] = max(0, bg_box[0])
                    bg_box[1] = max(0, bg_box[1])
                    bg_box[2] = min(img_w, bg_box[2])
                    bg_box[3] = min(img_h, bg_box[3])
                    
                    # 绘制背景
                    draw.rectangle(bg_box, fill=bg_color)
                    
                    # 绘制文本
                    draw.text(text_position, text, fill=text_color, font=font)
                    drawn_boxes += 1
                    
                except Exception as e:
                    print(f"Error drawing text for {category_name}: {e}")
                    # 最后的备选：直接绘制文本
                    try:
                        draw.text(text_position, text, fill=box_color, font=font)
                        drawn_boxes += 1
                    except:
                        pass
            
            # 添加最终的调试信息
            print(f"Total predicted boxes after filtering: {drawn_boxes}")
            if drawn_boxes == 0:
                print("警告: 没有预测框被绘制。可能的原因:")
                print("1. 所有预测的置信度都低于阈值 ({:.2f})".format(confidence_threshold))
                print("2. 模型可能没有正确训练或加载")
                print("3. 输入图像可能不包含模型训练时见过的类别")
        else:
            print("No predicted boxes to draw.")
    else:
        print("Warning: `criterion` not provided to `save_detection_images`. Saving image without predicted boxes.")
        # 即使没有criterion，也保存原始图像
        pass  # 继续执行保存图像的代码

    # Save the modified image with unique filename to prevent overwriting
    save_path = os.path.join(img_output_dir, f"epoch_{epoch}.png")
    print(f"Attempting to save image to: {save_path}")
    try:
        img.save(save_path)
        print(f"Image saved to {save_path} with {drawn_boxes} labeled boxes")
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")
        import traceback
        traceback.print_exc()
    sys.stdout.flush()  # Ensure immediate output
