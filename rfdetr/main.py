# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

'''
v7修改说明：解决tensorboard中map曲线显示问题
1.注释rfdetr\cli\main.py的trainer函数与Roboflow工作区相关的无用代码
2.rfdetr\config.py的TrainConfig类dataset_file改为"coco"
3.rfdetr\config.py的TrainConfig类dataset_dir改为"./dataset",pretrain_weights改为"./pretrain_weights/rf-detr-base-coco.pth"
4.创建dataset\train\_annotations.coco.json与dataset\train\example.json文件
5.rfdetr\main.py文件的get_args_parser函数的'--dataset_dir'设置default='./dataset',
'--pretrain_weights'default='./pretrain_weights/rf-detr-base-coco.pth',
'--encoder', default='vit_tiny'改为default='dinov2_windowed_registers_small'
dinov2预训练权重对源码模型结构适配
class Model:
    def __init__
169行添加for key in list(checkpoint['model'].keys()):
                if 'backbone.0.projector' in key:
                    del checkpoint['model'][key]

6.修复了 `if __name__ == '__main__'` 中直接调用不存在的 `main` 函数的错误，改为实例化 `Model` 并调用 `train` 方法。
7.在 `rfdetr/engine.py` 中添加了 `save_detection_images` 函数，用于在训练过程中将检测结果可视化并保存到 `./result` 目录。
#结果可视化函数新增导入
import os
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from rfdetr.util import box_ops
import numpy as np
#新增的可视化函数
def denormalize_image(tensor, mean, std):
    """Denormalizes a tensor image with mean and standard deviation."""
    tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor.mul_(std).add_(mean)
    return tensor

def save_detection_images(outputs, targets, samples, epoch, step, sub_step, output_dir="./result", threshold=0.5):
    """
    Saves images with predicted bounding boxes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get predictions
    pred_logits = outputs['pred_logits'].softmax(-1)
    pred_boxes = outputs['pred_boxes']
    
    # Denormalize images for saving
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(len(targets)):
        # Get image
        img_tensor = samples.tensors[i]
        img_tensor_denorm = denormalize_image(img_tensor, mean, std)
        img = F.to_pil_image(img_tensor_denorm.cpu())
        draw = ImageDraw.Draw(img)

        # Get original size and scale boxes
        orig_size = targets[i]['orig_size']
        img_w, img_h = orig_size.cpu().numpy()
        
        # Filter predictions by threshold
        scores, labels = pred_logits[i].max(-1)
        keep = scores > threshold
        
        boxes_to_draw = pred_boxes[i][keep]
        labels_to_draw = labels[keep]
        scores_to_draw = scores[keep]

        if boxes_to_draw.shape[0] == 0:
            continue

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] and scale
        scaled_boxes = box_ops.box_cxcywh_to_xyxy(boxes_to_draw)
        scaled_boxes = scaled_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=scaled_boxes.device)

        # Draw boxes
        for box, label, score in zip(scaled_boxes, labels_to_draw, scores_to_draw):
            box = box.cpu().tolist()
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0] + 2, box[1] + 2), f"L:{label.item()} S:{score.item():.2f}", fill="red")

        # Save image
        image_id = targets[i]['image_id'].item()
        save_path = os.path.join(output_dir, f"epoch{epoch}_step{step}_{sub_step}_img{image_id}.png")
        img.save(save_path)

8.将当前文件路径添加到python系统路径靠前防止导入其它项目的文件
9.在rfdetr\models\backbone\backbone.py文件Backbone类__init__方法中添加非窗口注意力强制下载预训练权重模型
添加if 'large' in name:
            num_layers = 24
        else:
            num_layers = 12
        
        processed_out_feature_indexes = out_feature_indexes
        if out_feature_indexes == [-1]:
            processed_out_feature_indexes = [num_layers - 1]
修改out_feature_indexes=out_feature_indexes,为out_feature_indexes=processed_out_feature_indexes,
10.rfdetr\models\backbone\dinov2_configs\dinov2_with_registers_base.json文件移除了"out_features": [
        "stage12"
    ],
    "out_indices": [
        12
    ],
    防止与定义的'--out_feature_indexes', default=[-1]改为default=[2, 5, 8, 11]冲突
11.创建rfdetr\models\backbone\dinov2_configs\dinov2_with_registers_small.json
12.rfdetr\models\backbone\dinov2.py文件74-77行
dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]
改为processed_out_feature_indexes = out_feature_indexes
            if out_feature_indexes == [-1]:
                processed_out_feature_indexes = [11]
            dino_config["out_features"] = [f"stage{i+1}" for i in processed_out_feature_indexes]
13.rfdetr\datasets\coco.py文件def build(image_set, args, resolution):
    root = Path(args.coco_path)
    改为def build(image_set, args, resolution):
    root = Path(args.dataset_dir)
14.修改 rfdetr/datasets/transforms.py 中的 resize 函数
129行if target is None:
        return rescaled_image, None
改为# Calculate padding to make dimensions divisible by 14
    w, h = rescaled_image.size
    pad_w = (14 - (w % 14)) % 14
    pad_h = (14 - (h % 14)) % 14
    padded_image = F.pad(rescaled_image, (0, 0, pad_w, pad_h))

    if target is None:
        return padded_image, None

148行 h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
    

    return rescaled_image, target
改为
# Update target size to padded image size
    target["size"] = torch.tensor([h + pad_h, w + pad_w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), (h + pad_h, w + pad_w), mode="nearest")[:, 0] > 0.5
    
    return padded_image, target

SquareResize 类__call__函数，
234行rescaled_img=F.resize(img, (size, size))
        w, h = rescaled_img.size
        if target is None:
            return rescaled_img, None
改为rescaled_img = F.resize(img, (size, size))

        # Calculate padding to make dimensions divisible by 14
        w, h = rescaled_img.size
        pad_w = (14 - (w % 14)) % 14
        pad_h = (14 - (h % 14)) % 14
        padded_image = F.pad(rescaled_img, (0, 0, pad_w, pad_h))

        if target is None:
            return padded_image, None

254行target["size"] = torch.tensor([h, w])

        return rescaled_img, target
改为# Update target size to padded image size
        target["size"] = torch.tensor([h + pad_h, w + pad_w])

        return padded_image, target
以确保图像尺寸能够被 14 整除。
15.rfdetr\models\backbone\dinov2_with_windowed_attn.py文件WindowedDinov2WithRegistersEmbeddings类forward函数
291行插入# Ensure height and width are divisible by patch_size * num_windows
        adjusted_height = (height // (self.config.patch_size * self.config.num_windows)) * (self.config.patch_size * self.config.num_windows)
        adjusted_width = (width // (self.config.patch_size * self.config.num_windows)) * (self.config.patch_size * self.config.num_windows)

        # Resize pixel_values to adjusted_height, adjusted_width before patch embedding
        if adjusted_height != height or adjusted_width != width:
            pixel_values = F.resize(pixel_values, (adjusted_height, adjusted_width))
            batch_size, _, height, width = pixel_values.shape # Update height and width

315行 windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(batch_size, num_windows, num_h_patches_per_window, num_windows, num_h_patches_per_window, -1)
改为windowed_pixel_tokens = pixel_tokens_with_pos_embed.view(batch_size, num_windows, num_h_patches_per_window, num_windows, num_w_patches_per_window, -1)

1101行batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
改为
# Get the actual height and width from the hidden_state (after removing CLS and register tokens)
                    # hidden_state shape: (batch_size, num_patches, hidden_size)
                    batch_size, num_patches, hidden_size = hidden_state.shape
                    patch_size = self.config.patch_size
                    
                    # Calculate height and width based on num_patches and patch_size
                    # Assuming square patches and original image was square or close to square
                    # This might need adjustment if aspect ratio is highly variable
                    height = int(math.sqrt(num_patches)) * patch_size
                    width = int(math.sqrt(num_patches)) * patch_size

16.rfdetr\models\backbone\dinov2_with_windowed_attn.py文件WindowedDinov2WithRegistersBackbone类forward函数
1119行num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows
                        hidden_state = hidden_state.reshape(B // num_windows_squared, num_windows_squared * HW, C)
                        hidden_state = hidden_state.view(B // num_windows_squared, self.config.num_windows, self.config.num_windows, num_h_patches_per_window, num_w_patches_per_window, C)
                        hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5)

                    hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)
改为# Calculate num_h_patches_per_window and num_w_patches_per_window from HW
                        # HW is the number of patches per window (including CLS token if present)
                        # Assuming square patches within a window
                        # num_patches_per_window = HW # This line is problematic, HW is already the number of patches per window
                        
                        # Recalculate num_h_patches_per_window and num_w_patches_per_window based on total patches and number of windows
                        num_h_patches_per_window = num_h_patches // self.config.num_windows
                        num_w_patches_per_window = num_w_patches // self.config.num_windows

                        hidden_state = hidden_state.reshape(B // num_windows_squared, num_windows_squared, num_h_patches_per_window, num_w_patches_per_window, C)
                        hidden_state = hidden_state.permute(0, 1, 3, 2, 4)
                        hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, C)
                    else:
                        hidden_state = hidden_state.reshape(batch_size, num_h_patches, num_w_patches, -1)

然后
1105行# this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    # Get the actual height and width from the hidden_state (after removing CLS and register tokens)
                    # hidden_state shape: (batch_size, num_patches, hidden_size)
                    batch_size, num_patches, hidden_size = hidden_state.shape
                    patch_size = self.config.patch_size
                    
                    # Calculate height and width based on num_patches and patch_size
                    # Assuming square patches and original image was square or close to square
                    # This might need adjustment if aspect ratio is highly variable
                    height = int(math.sqrt(num_patches)) * patch_size
                    width = int(math.sqrt(num_patches)) * patch_size

                    num_h_patches = height // patch_size
                    num_w_patches = width // patch_size
改为
batch_size = pixel_values.shape[0]
                    height, width = pixel_values.shape[2:]
                    patch_size = self.config.patch_size
                    
                    adjusted_height = (height // (self.config.patch_size * self.config.num_windows)) * (self.config.patch_size * self.config.num_windows)
                    adjusted_width = (width // (self.config.patch_size * self.config.num_windows)) * (self.config.patch_size * self.config.num_windows)

                    num_h_patches = adjusted_height // patch_size
                    num_w_patches = adjusted_width // patch_size

17.rfdetr\models\backbone\dinov2_with_windowed_attn.py文件WindowedDinov2WithRegistersLayer类forward函数
628行hidden_states = hidden_states.view(B // num_windows_squared, num_windows_squared * HW, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            # reshape x to add windows back
            B, HW, C = hidden_states.shape
            num_windows_squared = self.num_windows ** 2
            # hidden_states = hidden_states.view(B * num_windows_squared, HW // num_windows_squared, C)
            attention_output = attention_output.view(B * num_windows_squared, HW // num_windows_squared, C)
改为
# Calculate the number of patches per window (excluding CLS and register tokens)
            # HW is 1 (CLS) + num_register_tokens + num_patches_per_window
            num_patches_per_window_actual = HW - 1 - self.config.num_register_tokens
            
            # Calculate the total number of tokens (including CLS and register tokens)
            total_tokens = (1 + self.config.num_register_tokens + num_patches_per_window_actual) * num_windows_squared
            
            # Reshape to original batch size and total tokens (including CLS and register tokens)
            hidden_states = hidden_states.view(B // num_windows_squared, total_tokens, C)

        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2WithRegisters, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        if run_full_attention:
            attention_output = attention_output.view(shortcut.shape)

18.在rfdetr\models\backbone\dinov2_with_windowed_attn.py文件 WindowedDinov2WithRegistersLayer类 的 __init__ 方法中添加 self.config = config。

19.rfdetr\main.py文件
添加from torch.utils.tensorboard import SummaryWriter
class Model:
def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
改为
def train(self, callbacks: DefaultDict[str, List[Callable]], writer=None, **kwargs):
624行effective_batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules, 
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks)
改为
effective_batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks, writer=writer)
677行self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args
改为self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args, writer=writer, epoch=epoch

1260行model = Model(**config)
        model.train(callbacks=DefaultDict(list), **config)
改为writer = SummaryWriter('runs')
        model = Model(**config)
        model.train(callbacks=DefaultDict(list), writer=writer, **config)
        writer.close()

20.rfdetr\engine.py文件
添加import time

train_one_epoch方法添加writer=None,参数
211行添加
# For FPS calculation
forward_pass_times = []
223行
            outputs = model(samples)
改为
            start_time = time.time()
            outputs = model(samples)
            end_time = time.time()
            # We only measure forward pass time for batch size 1 for a realistic FPS measurement
            if samples.tensors.shape[0] == 1:
                forward_pass_times.append(end_time - start_time)

284行
stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if writer:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'eval/{k}', v, epoch)
    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            if writer:
                for i, stat in enumerate(coco_evaluator.coco_eval["bbox"].stats):
                    writer.add_scalar(f'eval/bbox_stat_{i}', stat, epoch)
改为
# Calculate FPS
    if forward_pass_times:
        avg_forward_time = np.mean(forward_pass_times)
        fps = 1 / avg_forward_time
    else:
        fps = 0
    
    print(f"FPS (batch_size=1): {fps:.2f}")

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['fps'] = fps

    if coco_evaluator is not None:
        if "bbox" in postprocessors.keys():
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
            map_50_95 = coco_evaluator.coco_eval['bbox'].stats[0]
            map_50 = coco_evaluator.coco_eval['bbox'].stats[1]
            stats['mAP@.50-.95'] = map_50_95
            stats['mAP@.50'] = map_50
            print(f"mAP@.50-.95: {map_50_95:.4f}")
            print(f"mAP@.50: {map_50:.4f}")

    if writer:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'eval/{k}', v, epoch)
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP_50-95', stats['mAP@.50-.95'], epoch)
            writer.add_scalar('eval/mAP_50', stats['mAP@.50'], epoch)
            writer.add_scalar('eval/FPS', stats['fps'], epoch)



180行添加if writer:
            writer.add_scalar('train/loss', loss_value, it)
            writer.add_scalar('train/class_error', loss_dict_reduced["class_error"], it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], it)
191行def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):
改为def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args=None):

271行添加if writer:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'eval/{k}', v, epoch)
278行添加if writer:
                for i, stat in enumerate(coco_evaluator.coco_eval["bbox"].stats):
                    writer.add_scalar(f'eval/bbox_stat_{i}', stat, epoch)

21.rfdetr\main.py的train方法中
535行添加# Select 100 representative images for visualization
        image_ids = list(dataset_train.coco.imgs.keys())
        if len(image_ids) > 100:
            # Use a fixed seed for reproducibility of selected images
            random.seed(args.seed)
            representative_image_ids = random.sample(image_ids, 100)
        else:
            representative_image_ids = image_ids
        # convert to a set for faster lookup
        representative_image_ids = set(representative_image_ids)
691行
 vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks, writer=writer)
 改为
 vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks, writer=writer,
                representative_image_ids=representative_image_ids,
                coco_dataset=dataset_train.coco)
22.rfdetr\engine.py文件train_one_epoch方法添加representative_image_ids=None,coco_dataset=None,参数
133行save_detection_images(outputs, new_targets, new_samples, epoch, data_iter_step, i)
改为image_id = new_targets[0]['image_id'].item()
    if representative_image_ids and image_id in representative_image_ids:
        save_detection_images(outputs, new_targets, new_samples, epoch, data_iter_step, i, coco_dataset=coco_dataset)

最后函数改为
def save_detection_images(outputs, targets, samples, epoch, step, sub_step, output_dir="./result", threshold=0.5, coco_dataset=None):
    """
    Saves images with predicted and ground truth bounding boxes.
    """
    # Get predictions
    pred_logits = outputs['pred_logits'].softmax(-1)
    pred_boxes = outputs['pred_boxes']
    
    # Denormalize images for saving
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for i in range(len(targets)):
        image_id = targets[i]['image_id'].item()
        
        # Get original image filename
        img_info = coco_dataset.loadImgs(image_id)[0]
        img_filename = img_info['file_name']
        img_name_without_ext = os.path.splitext(img_filename)[0]

        # Create a directory for the image if it doesn't exist
        img_output_dir = os.path.join(output_dir, img_name_without_ext)
        os.makedirs(img_output_dir, exist_ok=True)

        # Get image
        img_tensor = samples.tensors[i]
        img_tensor_denorm = denormalize_image(img_tensor, mean, std)
        img = F.to_pil_image(img_tensor_denorm.cpu())
        draw = ImageDraw.Draw(img)

        # Get original size and scale boxes
        orig_size = targets[i]['orig_size']
        img_h, img_w = orig_size.cpu().numpy()
        
        # --- Draw Ground Truth Boxes (Green) ---
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']
        
        # Convert GT boxes from [cx, cy, w, h] to [x1, y1, x2, y2] and scale
        scaled_gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
        scaled_gt_boxes = scaled_gt_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=scaled_gt_boxes.device)

        for box, label in zip(scaled_gt_boxes, gt_labels):
            box = box.cpu().tolist()
            draw.rectangle(box, outline="lime", width=2)
            draw.text((box[0] + 2, box[1] - 12), f"GT_L:{label.item()}", fill="lime")

        # --- Draw Predicted Boxes (Red) ---
        scores, labels = pred_logits[i].max(-1)
        keep = scores > threshold
        
        boxes_to_draw = pred_boxes[i][keep]
        labels_to_draw = labels[keep]
        scores_to_draw = scores[keep]

        if boxes_to_draw.shape[0] > 0:
            # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] and scale
            scaled_boxes = box_ops.box_cxcywh_to_xyxy(boxes_to_draw)
            scaled_boxes = scaled_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=scaled_boxes.device)

            # Draw boxes
            for box, label, score in zip(scaled_boxes, labels_to_draw, scores_to_draw):
                box = box.cpu().tolist()
                draw.rectangle(box, outline="red", width=2)
                draw.text((box[0] + 2, box[1] + 2), f"L:{label.item()} S:{score.item():.2f}", fill="red")

        # Save image
        save_path = os.path.join(img_output_dir, f"epoch{epoch}_step{step}.png")
        img.save(save_path)

23.rfdetr\engine.py文件train_one_epoch方法
132行
if epoch % 5 == 0 and data_iter_step % 10 == 0:
                    try:
                        image_id = new_targets[0]['image_id'].item()
                        if representative_image_ids and image_id in representative_image_ids:
                            save_detection_images(outputs, new_targets, new_samples, epoch, data_iter_step, i, coco_dataset=coco_dataset)
改为
if epoch % 5 == 0:
                    try:
                        # Iterate over each target in the sub-batch
                        for j in range(len(new_targets)):
                            image_id = new_targets[j]['image_id'].item()
                            if representative_image_ids and image_id in representative_image_ids:
                                # Pass only the relevant data for the specific image
                                single_output = {k: v[j:j+1] for k, v in outputs.items()}
                                single_target = [new_targets[j]]
                                single_sample = NestedTensor(new_samples.tensors[j:j+1], new_samples.mask[j:j+1])
                                save_detection_images(single_output, single_target, single_sample, epoch, data_iter_step, i, coco_dataset=coco_dataset)
def evaluate
if writer:
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'eval/{k}', v, epoch)
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP_50-95', stats['mAP@.50-.95'], epoch)
            writer.add_scalar('eval/mAP_50', stats['mAP@.50'], epoch)
            writer.add_scalar('eval/FPS', stats['fps'], epoch)
        if "segm" in postprocessors.keys():
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
改为
if writer:
        # Log main stats
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['fps', 'mAP@.50-.95', 'mAP@.50']:
                writer.add_scalar(f'eval/{k}', v, epoch)
        
        # Log specific metrics for clarity in TensorBoard
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP@.50-.95', stats['mAP@.50-.95'], epoch)
        if 'mAP@.50' in stats:
            writer.add_scalar('eval/mAP@.50', stats['mAP@.50'], epoch)
        if 'fps' in stats:
            writer.add_scalar('eval/FPS', stats['fps'], epoch)

    if coco_evaluator and "segm" in postprocessors.keys():
        stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
309行
if len(stats["coco_eval_bbox"]) > 1: # Ensure stats has enough elements
                map_50_95 = 0.0
                map_50 = 0.0
                if len(stats["coco_eval_bbox"]) > 1:
                    map_50_95 = stats["coco_eval_bbox"][0]
                    map_50 = stats["coco_eval_bbox"][1]
                stats['mAP@.50-.95'] = map_50_95
                stats['mAP@.50'] = map_50
                # Add mAP values to metric_logger
                metric_logger.add_meter("mAP@.50-.95", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
                metric_logger.add_meter("mAP@.50", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
                metric_logger.meters["mAP@.50-.95"].update(map_50_95)
                metric_logger.meters["mAP@.50"].update(map_50)
                print(f"mAP@.50-.95: {map_50_95:.4f}")
                print(f"mAP@.50: {map_50:.4f}")

    if writer and epoch is not None:
        writer.add_scalar('eval/FPS', fps, epoch)
        # Log all stats from metric_logger
        for k, meter in metric_logger.meters.items():
            if k != 'coco_eval_bbox': # Exclude coco_eval_bbox as it's a list
                writer.add_scalar(f'eval/{k}', meter.global_avg, epoch)
改为
stats['mAP@.50-.95'] = stats["coco_eval_bbox"][0]
            stats['mAP@.50'] = stats["coco_eval_bbox"][1]

    if writer and epoch is not None:
        # Log main stats
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['fps', 'mAP@.50-.95', 'mAP@.50', 'coco_eval_bbox']:
                writer.add_scalar(f'eval/{k}', v, epoch)
        
        # Log specific metrics for clarity in TensorBoard
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP@.50-.95', stats['mAP@.50-.95'], epoch)
        if 'mAP@.50' in stats:
            writer.add_scalar('eval/mAP@.50', stats['mAP@.50'], epoch)
        if 'fps' in stats:
            writer.add_scalar('eval/FPS', stats['fps'], epoch)

def save_detection_images
# Save image
        save_path = os.path.join(img_output_dir, f"epoch{epoch}_step{step}.png")
改为
# Save image with a unique name
        save_path = os.path.join(img_output_dir, f"epoch{epoch}_step{step}_sub{sub_step}.png")

        
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


v4改动
24.rfdetr\engine.py文件
def train_one_epoch
83行插入
saved_representative_images_this_epoch = set()

131行
# Save images periodically to check detection results
                if epoch % 5 == 0:
                    try:
                        # Iterate over each target in the sub-batch
                        for j in range(len(new_targets)):
                            image_id = new_targets[j]['image_id'].item()
                            if representative_image_ids and image_id in representative_image_ids:
                                # Pass only the relevant data for the specific image
                                single_output = {k: v[j:j+1] for k, v in outputs.items()}
                                single_target = [new_targets[j]]
                                single_sample = NestedTensor(new_samples.tensors[j:j+1], new_samples.mask[j:j+1])
                                save_detection_images(single_output, single_target, single_sample, epoch, data_iter_step, i, coco_dataset=coco_dataset)
                    except Exception as e:
                        print(f"Error saving detection image: {e}")
改为
# Save images periodically to check detection results
                if representative_image_ids:
                    for j in range(len(new_targets)):
                        image_id = new_targets[j]['image_id'].item()
                        # print(f"Checking image_id: {image_id}, is in representative: {image_id in representative_image_ids}, is already saved: {image_id in saved_representative_images_this_epoch}")
                        if image_id in representative_image_ids and image_id not in saved_representative_images_this_epoch:
                            try:
                                print(f"Saving detection image for image_id: {image_id} in epoch {epoch}")
                                single_output = {k: v[j:j+1] for k, v in outputs.items()}
                                single_target = [new_targets[j]]
                                single_sample = NestedTensor(new_samples.tensors[j:j+1], new_samples.mask[j:j+1])
                                save_detection_images(single_output, single_target, single_sample, epoch, coco_dataset=coco_dataset, criterion=criterion)
                                saved_representative_images_this_epoch.add(image_id)
                            except Exception as e:
                                print(f"Error saving detection image {image_id} in epoch {epoch}: {e}")


def evaluate
map_50_95 = coco_evaluator.coco_eval['bbox'].stats[0]
            map_50 = coco_evaluator.coco_eval['bbox'].stats[1]
            stats['mAP@.50-.95'] = map_50_95
            stats['mAP@.50'] = map_50
            print(f"mAP@.50-.95: {map_50_95:.4f}")
            print(f"mAP@.50: {map_50:.4f}")

    if writer:
        # Log main stats
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['fps', 'mAP@.50-.95', 'mAP@.50']:
改为
# Ensure mAP stats are calculated and available before writing
            if stats["coco_eval_bbox"]:
                map_50_95 = stats["coco_eval_bbox"][0]
                map_50 = stats["coco_eval_bbox"][1]
                stats['mAP@.50-.95'] = map_50_95
                stats['mAP@.50'] = map_50
                print(f"mAP@.50-.95: {map_50_95:.4f}")
                print(f"mAP@.50: {map_50:.4f}")

    if writer and epoch is not None:
        # Log main stats
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['fps', 'mAP@.50-.95', 'mAP@.50', 'coco_eval_bbox']:
之后   
# Ensure mAP stats are calculated and available before writing
            if stats["coco_eval_bbox"]:
                map_50_95 = stats["coco_eval_bbox"][0]
                map_50 = stats["coco_eval_bbox"][1]
                stats['mAP@.50-.95'] = map_50_95
                stats['mAP@.50'] = map_50
                print(f"mAP@.50-.95: {map_50_95:.4f}")
                print(f"mAP@.50: {map_50:.4f}")

    if writer and epoch is not None:
        # Log main stats
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['fps', 'mAP@.50-.95', 'mAP@.50', 'coco_eval_bbox']:
                writer.add_scalar(f'eval/{k}', v, epoch)
        
        # Log specific metrics for clarity in TensorBoard
        if 'mAP@.50-.95' in stats:
            writer.add_scalar('eval/mAP@.50-.95', stats['mAP@.50-.95'], epoch)
        if 'mAP@.50' in stats:
            writer.add_scalar('eval/mAP@.50', stats['mAP@.50'], epoch)
        if 'fps' in stats:
            writer.add_scalar('eval/FPS', stats['fps'], epoch)
改为
if len(stats["coco_eval_bbox"]) > 1: # Ensure stats has enough elements
                map_50_95 = 0.0
                map_50 = 0.0
                if len(stats["coco_eval_bbox"]) > 1:
                    map_50_95 = stats["coco_eval_bbox"][0]
                    map_50 = stats["coco_eval_bbox"][1]
                stats['mAP@.50-.95'] = map_50_95
                stats['mAP@.50'] = map_50
                # Add mAP values to metric_logger
                metric_logger.add_meter("mAP@.50-.95", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
                metric_logger.add_meter("mAP@.50", utils.SmoothedValue(window_size=1, fmt="{value:.4f}"))
                metric_logger.meters["mAP@.50-.95"].update(map_50_95)
                metric_logger.meters["mAP@.50"].update(map_50)
                print(f"mAP@.50-.95: {map_50_95:.4f}")
                print(f"mAP@.50: {map_50:.4f}")

    if writer and epoch is not None:
        writer.add_scalar('eval/FPS', fps, epoch)
        # Log all stats from metric_logger
        for k, meter in metric_logger.meters.items():
            if k != 'coco_eval_bbox': # Exclude coco_eval_bbox as it's a list
                writer.add_scalar(f'eval/{k}', meter.global_avg, epoch)


                 

def save_detection_images(outputs, targets, samples, epoch, step, sub_step, output_dir="./result", threshold=0.7, coco_dataset=None):
改为
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
        print(f"Warning: Original image not found at {original_image_path}, skipping.")
        return
    
    img = Image.open(original_image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

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
                text_position = (box[0] + 2, box[1] + 2)
                
                try:
                    text_bbox = draw.textbbox(text_position, text)
                    draw.rectangle(text_bbox, fill="red")
                    draw.text(text_position, text, fill='white')
                except AttributeError:  # Fallback for older Pillow versions
                    text_size = draw.textsize(text)
                    draw.rectangle([text_position[0], text_position[1], text_position[0] + text_size[0], text_position[1] + text_size[1]], fill="red")
                    draw.text(text_position, text, fill='white')
    else:
        print("Warning: `criterion` not provided to `save_detection_images`. Cannot determine which boxes are used for loss. Skipping image save.")
        return

    # Save the modified image
    save_path = os.path.join(img_output_dir, f"epoch_{epoch}.png")
    img.save(save_path)




23.rfdetr\main.py文件class Model:def train
添加print(f"Selected {len(representative_image_ids)} representative image IDs for visualization.")
1165行
if args.use_ema:
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args, writer=writer, epoch=epoch
                )
改为
if args.use_ema:
                ema_writer = SummaryWriter(os.path.join(args.output_dir, 'ema_eval'))
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args, writer=ema_writer, epoch=epoch
                )
                ema_writer.close()

if __name__ == '__main__':
1767行
writer = SummaryWriter('runs')
改为
writer = SummaryWriter(args.output_dir)

24.注释微调部分
# from peft import LoraConfig, get_peft_model

25.rfdetr\models\backbone\backbone.py
# from peft import LoraConfig, get_peft_model, PeftModel

26.rfdetr\models\backbone\dinov2_with_windowed_attn.py
28行去除torch_int,
261行class WindowedDinov2WithRegistersEmbeddings(nn.Module):
def interpolate_pos_encoding

sqrt_num_positions = torch_int(num_positions**0.5)
改为
sqrt_num_positions = int(num_positions**0.5)

271行
size=(torch_int(height), torch_int(width)),  # Explicit size instead of scale_factor
改为
size=(int(height), int(width)),  # Explicit size instead of scale_factor

598行class WindowedDinov2WithRegistersLayer(nn.Module):添加
config._attn_implementation = "eager"

27.1633行parser.add_argument('--batch_size', default=2, type=int)
default=2改为1

训练效果：
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.006
FPS (batch_size=1): 49.56
mAP@.50-.95: 0.0000
mAP@.50: 0.0001
Training time 9:20:12
Results saved to output\results.json

'''
# 将当前文件路径添加到python系统路径靠前防止导入其它项目的文件
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import ast
import copy
import datetime
import json
import math
import shutil
import time
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import DefaultDict, List, Callable
import random
import numpy as np
import torch
# from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_model, build_criterion_and_postprocessors
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.files import download_file
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.utils import ModelEma, BestMetricHolder, clean_state_dict

from torch.utils.tensorboard import SummaryWriter

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth"
}

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(
                f"Downloading pretrained weights for {pretrain_weights}"
            )
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )

class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)

            # Extract class_names from checkpoint if available
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
                self.class_names = checkpoint['args'].class_names
                
            checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
            if checkpoint_num_classes != args.num_classes + 1:
                logger.warning(
                    f"num_classes mismatch: pretrain weights has {checkpoint_num_classes - 1} classes, but your model has {args.num_classes} classes\n"
                    f"reinitializing detection head with {checkpoint_num_classes - 1} classes"
                )
                self.reinitialize_detection_head(checkpoint_num_classes)
            # add support to exclude_keys
            # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                for exclude_key in args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)
            if args.pretrain_keys_modify_to_load is not None:
                from util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    try:
                        checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                            model_without_ddp.state_dict()[modify_key_to_load],
                            checkpoint['model'][modify_key_to_load]
                        )
                    except:
                        print(f"Failed to load {modify_key_to_load}, deleting from checkpoint")
                        checkpoint['model'].pop(modify_key_to_load)

            # we may want to resume training with a smaller number of groups for group detr
            num_desired_queries = args.num_queries * args.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint['model'].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint['model'][name] = state[:num_desired_queries]

            # remove projector weights from checkpoint to avoid size mismatch
            # 在加载权重之前，手动将这个不匹配的 projector 权重从 checkpoint 中移除。这样，编码器的权重可以被正确加载，而投影层则会使用随机初始化的权重，这对于微调来说是完全可以接受的。
            for key in list(checkpoint['model'].keys()):
                if 'backbone.0.projector' in key:
                    del checkpoint['model'][key]
 
            self.model.load_state_dict(checkpoint['model'], strict=False)

        # if args.backbone_lora:
        #     print("Applying LORA to backbone")
        #     lora_config = LoraConfig(
        #         r=16,
        #         lora_alpha=16,
        #         use_dora=True,
        #         target_modules=[
        #             "q_proj", "v_proj", "k_proj",  # covers OWL-ViT
        #             "qkv", # covers open_clip ie Siglip2
        #             "query", "key", "value", "cls_token", "register_tokens", # covers Dinov2 with windowed attn
        #         ]
        #     )
        #     self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)
        self.model = self.model.to(self.device)
        self.criterion, self.postprocessors = build_criterion_and_postprocessors(args)
        self.stop_early = False
    
    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], writer=None, **kwargs):
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )
        args = populate_args(**kwargs)
        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        print(args)
        device = torch.device(args.device)
        
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocessors = build_criterion_and_postprocessors(args)
        model = self.model
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        param_dicts = get_param_dict(args, model_without_ddp)

        param_dicts = [p for p in param_dicts if p['params'].requires_grad]

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, 
                                    weight_decay=args.weight_decay)
        # Choose the learning rate scheduler based on the new argument

        dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
        dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)

        # Select 100 representative images for visualization
        image_ids = list(dataset_train.coco.imgs.keys())
        if len(image_ids) > 100:
            # Use a fixed seed for reproducibility of selected images
            random.seed(args.seed)
            representative_image_ids = random.sample(image_ids, 100)
        else:
            representative_image_ids = image_ids
        # convert to a set for faster lookup
        representative_image_ids = set(representative_image_ids)
        print(f"Selected {len(representative_image_ids)} representative image IDs for visualization.")

        # for cosine annealing, calculate total training steps and warmup steps
        total_batch_size_for_lr = args.batch_size * utils.get_world_size() * args.grad_accum_steps
        num_training_steps_per_epoch_lr = (len(dataset_train) + total_batch_size_for_lr - 1) // total_batch_size_for_lr
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr = num_training_steps_per_epoch_lr * args.warmup_epochs
        def lr_lambda(current_step: int):
            if current_step < warmup_steps_lr:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps_lr))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if args.lr_scheduler == 'cosine':
                    progress = float(current_step - warmup_steps_lr) / float(max(1, total_training_steps_lr - warmup_steps_lr))
                    return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
                elif args.lr_scheduler == 'step':
                    if current_step < args.lr_drop * num_training_steps_per_epoch_lr:
                        return 1.0
                    else:
                        return 0.1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        effective_batch_size = args.batch_size * args.grad_accum_steps
        min_batches = kwargs.get('min_batches', 5)
        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: {len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=effective_batch_size * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=effective_batch_size,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                sampler=sampler,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, effective_batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train, 
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn, 
                num_workers=args.num_workers
            )
        
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, 
                                    num_workers=args.num_workers)

        base_ds = get_coco_api_from_dataset(dataset_val)

        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None


        output_dir = Path(args.output_dir)
        
        if  utils.is_main_process():
            print("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(json.dumps(bm, indent=2))
                del benchmark_model
        
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            if args.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau) 
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:                
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            return
        
        # for drop
        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = (len(dataset_train) + total_batch_size - 1) // total_batch_size
        schedules = {}
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))

        print("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095 = 0
        best_map_50 = 0
        best_map_ema_5095 = 0
        best_map_ema_50 = 0
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()
            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
                effective_batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks, writer=writer,
                representative_image_ids=representative_image_ids,
                coco_dataset=dataset_train.coco)
            train_epoch_time = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every `checkpoint_interval` epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': self.ema_m.module.state_dict(),
                        })
                    if not args.dont_save_weights:
                        # create checkpoint dir
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        utils.save_on_master(weights, checkpoint_path)

            with torch.inference_mode():
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, args=args, writer=writer, epoch=epoch
                )
            
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                best_map_5095 = max(best_map_5095, map_regular)
                best_map_50 = max(best_map_50, test_stats["coco_eval_bbox"][1])
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                if not args.dont_save_weights:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            if args.use_ema:
                ema_writer = SummaryWriter(os.path.join(args.output_dir, 'ema_eval'))
                ema_test_stats, _ = evaluate(
                    self.ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args=args, writer=ema_writer, epoch=epoch
                )
                ema_writer.close()
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                map_ema = ema_test_stats['coco_eval_bbox'][0]
                best_map_ema_5095 = max(best_map_ema_5095, map_ema)
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    best_map_ema_50 = max(best_map_ema_50, ema_test_stats["coco_eval_bbox"][1])
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not args.dont_save_weights:
                        utils.save_on_master({
                            'model': self.ema_m.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
            log_stats.update(best_map_holder.summary())
            
            # epoch parameters
            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    output_dir / "eval" / name)
            
            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = best_map_ema_5095 > best_map_5095
        
        if utils.is_main_process():
            if best_is_ema:
                shutil.copy2(output_dir / 'checkpoint_best_ema.pth', output_dir / 'checkpoint_best_total.pth')
            else:
                shutil.copy2(output_dir / 'checkpoint_best_regular.pth', output_dir / 'checkpoint_best_total.pth')
            
            utils.strip_checkpoint(output_dir / 'checkpoint_best_total.pth')
        
            best_map_5095 = max(best_map_5095, best_map_ema_5095)
            best_map_50 = max(best_map_50, best_map_ema_50)

            results_json = {
                "map95": best_map_5095,
                "map50": best_map_50,
                "class": "all"
            }
            results = {
                "class_map": {
                    "valid": [
                        results_json
                    ]
                }
            }
            with open(output_dir / "results.json", "w") as f:
                json.dump(results, f)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Results saved to {}'.format(output_dir / "results.json"))
        
        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        for callback in callbacks["on_train_end"]:
            callback()
    
    def export(self, output_dir="output", infer_dir=None, simplify=False,  backbone_only=False, opset_version=17, verbose=True, force=False, shape=None, batch_size=1, **kwargs):
        """Export the trained model to ONNX format"""
        print(f"Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, onnx_simplify, make_infer_image
        except ImportError:
            print("It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again.")
            raise


        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ['input']
        output_names = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes = None
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version
        )
        
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")
        
        print("ONNX export completed successfully")
        self.model = self.model.to(device)
            
def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=1,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=2,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,
    
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None, 
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    
    # Backbone parameters
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale='P4',
    lite_refpoint_refine=False,
    num_select=100,
    dec_n_points=4,
    decoder_norm='LN',
    bbox_reparam=False,
    freeze_batch_norm=False,
    
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    
    # Dataset parameters
    dataset_file='coco',
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    
    # Output parameters
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=2,
    
    # Distributed training parameters
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,
    
    # FP16
    fp16_eval=False,
    
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=640,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs
    )
    return args
# 确保文件底部有get_args_parser函数定义（已在您的代码中看到）
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--grad_accum_steps', default=1, type=int)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    parser.add_argument('--do_benchmark', action='store_true', help='benchmark the model')

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default='./pretrain_weights/rf-detr-base-coco.pth',
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='dinov2_windowed_registers_small', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str,
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    # parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain", action="store_true", dest="force_no_pretrain")

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default=['P4'], type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--freeze_batch_norm', action='store_true')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--square_resize_div_64', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--dont_save_weights', action='store_true')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--ema_tau', default=0, type=float)

    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')

    # custom args
    parser.add_argument('--encoder_only', action='store_true', help='Export and benchmark encoder only')
    parser.add_argument('--backbone_only', action='store_true', help='Export and benchmark backbone only')
    parser.add_argument('--resolution', type=int, default=640, help="input resolution")
    parser.add_argument('--use_cls_token', action='store_true', help='use cls token')
    parser.add_argument('--multi_scale', action='store_true', help='use multi scale')
    parser.add_argument('--expanded_scales', action='store_true', help='use expanded scales')
    parser.add_argument('--warmup_epochs', default=1, type=float, 
        help='Number of warmup epochs for linear warmup before cosine annealing')
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        '--lr_scheduler',
        default='step',
        choices=['step', 'cosine'],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'"
    )
    parser.add_argument('--lr_min_factor', default=0.0, type=float, 
        help='Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing')
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on mAP improvement')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_min_delta', default=0.001, type=float,
                        help='Minimum change in mAP to qualify as an improvement')
    parser.add_argument('--early_stopping_use_ema', action='store_true',
                        help='Use EMA model metrics for early stopping')
    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    parser_export.add_argument('--profile', action='store_true', help='Run nsys profiling during TensorRT export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(640, 640), help="input shape (width, height)")
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = vars(args)  # Convert Namespace to dictionary
    
    if args.subcommand == 'distill':
        distill(**config)   
    elif args.subcommand is None:
        writer = SummaryWriter(args.output_dir)
        model = Model(**config)
        model.train(callbacks=DefaultDict(list), writer=writer, **config)
        writer.close()
    elif args.subcommand == 'export_model':
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError
            
        from deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        export_main(**config)



