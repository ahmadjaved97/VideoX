import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from sklearn.metrics import average_precision_score

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config): 
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES, 
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()

    mixup_fn = None
    if config.AUG.MIXUP > 0:
        criterion = SoftTargetCrossEntropy()
        mixup_fn = CutmixMixupBlending(num_classes=config.DATA.NUM_CLASSES, 
                                       smoothing=config.AUG.LABEL_SMOOTH, 
                                       mixup_alpha=config.AUG.MIXUP, 
                                       cutmix_alpha=config.AUG.CUTMIX, 
                                       switch_prob=config.AUG.MIXUP_SWITCH_PROB)
    elif config.AUG.LABEL_SMOOTH > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.AUG.LABEL_SMOOTH)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch, max_accuracy = 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)


    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST:
        print("Accuracy calculation-------")

        filename = 'event'
        logits_dir = './event_logits_dir_ov'
        acc1 = validate(val_loader, text_labels, model, config, filename, logits_dir)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1 * 100:.5f}%")

        mAP1 = validate_from_saved_logits2(val_loader, config, logits_dir, filename)
        print('map from saved individual logits: ', mAP1)

        full_lp = os.path.join('./', f'{filename}_logits.pth')
        mAP2 = validate_from_saved_logits(val_loader, full_lp, config)
        print('mAP from logits: ', mAP2)

        text_features = model.module.extracted_text_features  # Already extracted

        # Save text features
        torch.save(text_features, f"{filename}_text_features.pt")

        return

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn)

        acc1 = validate(val_loader, text_labels, model, config)
        logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1:.1f}%")
        is_best = acc1 > max_accuracy
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_accuracy, optimizer, lr_scheduler, logger, config.OUTPUT, is_best)

    config.defrost()
    config.TEST.NUM_CLIP = 4
    config.TEST.NUM_CROP = 3
    config.freeze()
    train_data, val_data, train_loader, val_loader = build_dataloader(logger, config)
    acc1 = validate(val_loader, text_labels, model, config)
    logger.info(f"Accuracy of the network on the {len(val_data)} test videos: {acc1 * 100:.1f}%")
    


def train_one_epoch(epoch, model, criterion, optimizer, lr_scheduler, train_loader, text_labels, config, mixup_fn):
    model.train()
    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    
    start = time.time()
    end = time.time()
    
    texts = text_labels.cuda(non_blocking=True)
    for idx, batch_data in enumerate(train_loader):

        images = batch_data["imgs"].cuda(non_blocking=True)
        label_id = batch_data["label"].cuda(non_blocking=True).float()
        
        label_id = label_id.reshape(-1)
        images = images.view((-1,config.DATA.NUM_FRAMES,3)+images.size()[-2:])
        
        if mixup_fn is not None:
            images, label_id = mixup_fn(images, label_id)

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)
        
        output = model(images, texts)

        total_loss = criterion(output, label_id)
        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot_loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

@torch.no_grad()
def validate(val_loader, text_labels, model, config):
    model.eval()
    
    all_targets = []
    all_outputs = []

    # Initialize mAP tracker
    mAP_meter = AverageMeter()

    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"].cuda(non_blocking=True).float()  # Already multi-hot formatted

            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
           
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            for i in range(n):  
                image = _image[:, i, :, :, :, :] 
                image_input = image.cuda(non_blocking=True)

                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                output, vid_feats = model(image_input, text_inputs)
                mean = output.mean(dim=1, keepdim=True)
                std = output.std(dim=1, keepdim=True) + 1e-6  # Prevent divide by zero
                standardized_logits = (output - mean) / std

                # Apply sigmoid activation 
                # similarity = torch.sigmoid(output.view(b, -1))
                similarity = output.view(b, -1)

                # similarity = torch.sigmoid(standardized_logits.view(b, -1))


                # Aggregate across different temporal clips
                tot_similarity += similarity

            tot_similarity = tot_similarity / n
            # Store outputs and ground truth 
            all_outputs.append(tot_similarity.cpu().numpy())
            all_targets.append(label_id.cpu().numpy())  # Already multi-hot

            if idx % config.PRINT_FREQ == 0:
                logger.info(f'Processed {idx}/{len(val_loader)} batches')

    #mean Average Precision
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    mAP_per_class = average_precision_score(all_targets, all_outputs, average=None)  
    mean_mAP = np.mean(mAP_per_class)  

    # Sync mAP across all GPUs
    mAP_meter.update(mean_mAP, n=1)
    mAP_meter.sync()

    logger.info(f" * Mean Average Precision (mAP): {mAP_meter.avg:.5f}")
    return mAP_meter.avg

@torch.no_grad()
def validate(val_loader, text_labels, model, config, filename):
    model.eval()
    
    all_targets = []
    all_outputs = []


    all_features = []  # Store features
    all_logits = []    # Store logits
    

    mAP_meter = AverageMeter()
    
    import pdb
    pdb.set_trace()
        
    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        
        for idx, batch_data in enumerate(val_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"].cuda(non_blocking=True).float()  # Already multi-hot formatted
            
            b, tn, c, h, w = _image.size()
            t = config.DATA.NUM_FRAMES
            n = tn // t
            _image = _image.view(b, n, t, c, h, w)
            
            tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
            
            batch_features = []
            batch_logits = []
            # import pdb
            # pdb.set_trace()

            for i in range(n):
                image = _image[:, i, :, :, :, :]
                image_input = image.cuda(non_blocking=True)
                
                if config.TRAIN.OPT_LEVEL == 'O2':
                    image_input = image_input.half()
                
                # logit, video_features
                output, features = model(image_input, text_inputs)
                
                batch_features.append(features.cpu())
                
                batch_logits.append(output.cpu())
                
                mean = output.mean(dim=1, keepdim=True)
                std = output.std(dim=1, keepdim=True) + 1e-6  # Prevent divide by zero
                standardized_logits = (output - mean) / std
                
                similarity = output.view(b, -1)
                

                tot_similarity += similarity
            
            # Stack features and logits for this batch (b, n, embedding_size)
            batch_features = torch.stack(batch_features, dim=1)  # Shape: (b, n, embedding_size)
            batch_logits = torch.stack(batch_logits, dim=1)      # Shape: (b, n, num_classes)
            
            # Store features and logits
            all_features.append(batch_features)
            all_logits.append(batch_logits)
            
            # Normalize similarity by number of views
            tot_similarity = tot_similarity / n
            
            # Store outputs and ground truth
            all_outputs.append(tot_similarity.cpu().numpy())
            all_targets.append(label_id.cpu().numpy())  # Already multi-hot
            
            if idx % config.PRINT_FREQ == 0:
                logger.info(f'Processed {idx}/{len(val_loader)} batches')
    
    # Concatenate all features and logits across batches
    all_features = torch.cat(all_features, dim=0)  # Shape: (num_instance, num_views, embedding_size)
    all_logits = torch.cat(all_logits, dim=0)      # Shape: (num_instance, num_views, num_classes)
    
    # Save features and logits to file
    features_path = os.path.join('./', f'{filename}_features.pth')
    logits_path = os.path.join('./', f'{filename}_logits.pth')
    
    torch.save(all_features, features_path)
    torch.save(all_logits, logits_path)
    
    logger.info(f"Saved features with shape {all_features.shape} to {features_path}")
    logger.info(f"Saved logits with shape {all_logits.shape} to {logits_path}")
    
    # Mean Average Precision calculation
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    mAP_per_class = average_precision_score(all_targets, all_outputs, average=None)
    mean_mAP = np.mean(mAP_per_class)
    
    # Sync mAP across all GPUs
    mAP_meter.update(mean_mAP, n=1)
    mAP_meter.sync()
    
    logger.info(f" * Mean Average Precision (mAP): {mAP_meter.avg:.5f}")
    return mAP_meter.avg

def validate_from_saved_logits(val_loader, logits_path, config):
    
    print("========== Inside validation from whole logits ================")
    # Initialize mAP tracker
    mAP_meter = AverageMeter()
    
    # Load saved logits - shape: (num_instance, num_views, num_classes)
    all_saved_logits = torch.load(logits_path)
    logger.info(f"Loaded logits with shape {all_saved_logits.shape}")
    
    all_targets = []
    all_outputs = []
    
    # Track current position in saved logits
    logit_idx = 0
    
    for idx, batch_data in enumerate(val_loader):
        label_id = batch_data["label"].cuda(non_blocking=True).float()  # Already multi-hot formatted
        b = label_id.shape[0]
        
        # Extract logits for this batch
        batch_logits = all_saved_logits[logit_idx:logit_idx + b]
        logit_idx += b
        
        # Get number of views
        n = batch_logits.shape[1]
        
        # Process logits for this batch
        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()

        for i in range(n):
            # Get logits for this view
            view_logits = batch_logits[:, i, :].cuda()
            
            # Apply the same processing as in the original validate function
            similarity = view_logits.view(b, -1)
            
            tot_similarity += similarity
        
        tot_similarity = tot_similarity / n
        
        # Store outputs and ground truth
        all_outputs.append(tot_similarity.cpu().numpy())
        all_targets.append(label_id.cpu().numpy())  # Already multi-hot
        
        if idx % config.PRINT_FREQ == 0:
            logger.info(f'Processed {idx}/{len(val_loader)} batches')
    
    # Mean Average Precision calculation
    all_outputs = np.vstack(all_outputs)
    all_targets = np.vstack(all_targets)
    
    logger.info(f"Final shapes: outputs {all_outputs.shape}, targets {all_targets.shape}")
    
    mAP_per_class = average_precision_score(all_targets, all_outputs, average=None)
    mean_mAP = np.mean(mAP_per_class)
    
    mAP_meter.update(mean_mAP, n=1)
    mAP_meter.sync()
    
    logger.info(f" * Mean Average Precision (mAP) from saved logits: {mAP_meter.avg:.5f}")
    
    return mAP_meter.avg


@torch.no_grad()
def validate(val_loader, text_labels, model, config, filename_prefix, logits_dir):
    """
    This validation function is used.
    """
    print("============ Inside validation function ================")
    os.makedirs(logits_dir, exist_ok=True)
    model.eval()
    
    # Prepare accumulators
    all_targets = []
    all_outputs = []

    all_features = []
    all_logits = []
    
    # AverageMeter for synchronization of mAP across GPUs
    mAP_meter = AverageMeter()
    
    
    # Move text embeddings to GPU
    text_inputs = text_labels.cuda()
    logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")

    for idx, batch_data in enumerate(val_loader):
    
        img_metas = batch_data["img_metas"].data  # e.g., [[{'filename': 'path/to/video'}]]
        video_name = img_metas[0][0]["filename"]
        

        _image = batch_data["imgs"]
        label_id = batch_data["label"].cuda(non_blocking=True).float()  # Multi-hot format

        b, tn, c, h, w = _image.size()
        t = config.DATA.NUM_FRAMES
        n = tn // t
        _image = _image.view(b, n, t, c, h, w)

        tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
        
        # Store per-batch (video) features/logits
        batch_features = []
        batch_logits = []

        for i in range(n):
            image_input = _image[:, i, :, :, :, :].cuda(non_blocking=True)
            if config.TRAIN.OPT_LEVEL == 'O2':
                image_input = image_input.half()

            # logits, features
            output, features = model(image_input, text_inputs)

            # Collect features/logits on CPU
            batch_features.append(features.cpu())
            batch_logits.append(output.cpu())

            
            mean = output.mean(dim=1, keepdim=True)
            std = output.std(dim=1, keepdim=True) + 1e-6
            standardized_logits = (output - mean) / std  # Not used further, but kept for reference

            # Accumulate similarity
            similarity = output.view(b, -1)  # (b, num_classes)
            tot_similarity += similarity

        # Stack per-view features and logits: (b, n, embed_size or num_classes)
        batch_features = torch.stack(batch_features, dim=1)
        batch_logits = torch.stack(batch_logits, dim=1)

        # Save per-video logits to an individual file (additional feature of 1st version)
        # Save per-video logits to an individual file
        video_basename = os.path.basename(video_name)
        per_video_logits_path = os.path.join(logits_dir, f"{filename_prefix}_{video_basename}_logits.pth")
        per_video_logits_npy = os.path.join(logits_dir, f"{filename_prefix}_{video_basename}_logits.npy")

        torch.save(batch_logits, per_video_logits_path)
        np.save(per_video_logits_npy, batch_logits.numpy())
        # logger.info(f"Saved logits for video={video_name} to {per_video_logits_path}")

        tot_similarity = tot_similarity / n

        # Accumulate outputs and targets for final mAP
        all_outputs.append(tot_similarity.cpu().numpy())
        all_targets.append(label_id.cpu().numpy())

        all_features.append(batch_features)
        all_logits.append(batch_logits)

        if idx % config.PRINT_FREQ == 0:
            logger.info(f"Processed {idx}/{len(val_loader)} videos")

   
    # Concatenate features/logits across all videos
    all_features = torch.cat(all_features, dim=0)  # (num_samples, num_views, embedding_size)
    all_logits = torch.cat(all_logits, dim=0)      # (num_samples, num_views, num_classes)

    # Final paths for concatenated features and logits
    features_path = os.path.join('./', f'{filename_prefix}_features.pth')
    total_logits_path = os.path.join('./', f'{filename_prefix}_logits.pth')

    torch.save(all_features, features_path)
    torch.save(all_logits, total_logits_path)

    logger.info(f"Saved all features with shape {all_features.shape} to {features_path}")
    logger.info(f"Saved all logits with shape {all_logits.shape} to {total_logits_path}")


    all_outputs = np.vstack(all_outputs)  # (num_videos, num_classes)
    all_targets = np.vstack(all_targets)  # (num_videos, num_classes)

    mAP_per_class = average_precision_score(all_targets, all_outputs, average=None)
    mean_mAP = np.mean(mAP_per_class)

    # Sync mAP across GPUs
    mAP_meter.update(mean_mAP, n=1)
    mAP_meter.sync()

    logger.info(f" * Mean Average Precision (mAP): {mAP_meter.avg:.5f}")

    return mAP_meter.avg

@torch.no_grad()
def validate_from_saved_logits2(val_loader, config, logits_dir, filename_prefix):
    """
    Loads per-video logits (saved as individual files) and computes mAP.
    """

    print('===== Inside saved individual logits function =======')
    all_outputs = []
    all_targets = []

    for idx, batch_data in enumerate(val_loader):
        # 1) Extract label(s) and move them to CPU for final scoring
        label_id = batch_data["label"].float().cpu().numpy().astype(np.float32)  # shape (b, num_classes)
        b = label_id.shape[0]

        # 2) Extract the video filename from img_metas
        
        img_metas = batch_data["img_metas"].data
        video_name = img_metas[0][0]["filename"]
        video_basename = os.path.basename(video_name)

        # logits_path = os.path.join(
        #     logits_dir, f"{filename_prefix}_{video_basename}_logits.pth"
        # )
        logits_path = os.path.join(logits_dir, f"{filename_prefix}_{video_basename}_logits.npy")

        # saved_logits = torch.load(logits_path, map_location="cpu")
        saved_logits = np.load(logits_path).astype(np.float32)

        # 5) Aggregate across the n views
        n = saved_logits.shape[1]
        # tot_similarity = saved_logits.sum(dim=1) / n  # shape: (b, num_classes)
        tot_similarity = np.sum(saved_logits, axis=1) / n

        # tot_similarity_np = tot_similarity.numpy()
        tot_similarity_np = tot_similarity


        # Collect for final mAP
        all_outputs.append(tot_similarity_np)
        all_targets.append(label_id)

        if idx % config.PRINT_FREQ == 0:
            print(f"Processed {idx}/{len(val_loader)} videos")

    # 8) Compute mAP across the entire set
    all_outputs = np.vstack(all_outputs)  # shape: (num_videos, num_classes)
    all_targets = np.vstack(all_targets)  # shape: (num_videos, num_classes)

    mAP_per_class = average_precision_score(all_targets, all_outputs, average=None)
    mean_mAP = np.mean(mAP_per_class)
    print(f" * Mean Average Precision (mAP) from saved logits: {mean_mAP:.5f}")

    return mean_mAP



# @torch.no_grad()
# def validate(val_loader, text_labels, model, config):
#     import pdb
#     pdb.set_trace()
#     model.eval()
    
#     acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
#     with torch.no_grad():
#         text_inputs = text_labels.cuda()
#         logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
#         for idx, batch_data in enumerate(val_loader):
#             _image = batch_data["imgs"]
#             label_id = batch_data["label"]
#             # print(label_id)
#             label_id = label_id.reshape(-1)

#             b, tn, c, h, w = _image.size()
#             t = config.DATA.NUM_FRAMES
#             n = tn // t
#             _image = _image.view(b, n, t, c, h, w)
           
#             tot_similarity = torch.zeros((b, config.DATA.NUM_CLASSES)).cuda()
#             for i in range(n):
#                 image = _image[:, i, :, :, :, :] # [b,t,c,h,w]
#                 label_id = label_id.cuda(non_blocking=True)
#                 image_input = image.cuda(non_blocking=True)

#                 if config.TRAIN.OPT_LEVEL == 'O2':
#                     image_input = image_input.half()
                
#                 output = model(image_input, text_inputs)
                
#                 similarity = output.view(b, -1).softmax(dim=-1)
#                 tot_similarity += similarity

#             values_1, indices_1 = tot_similarity.topk(1, dim=-1)
#             values_5, indices_5 = tot_similarity.topk(5, dim=-1)
#             acc1, acc5 = 0, 0
#             for i in range(b):
#                 if indices_1[i] == label_id[i]:
#                     acc1 += 1
#                 if label_id[i] in indices_5[i]:
#                     acc5 += 1
           
#             acc1_meter.update(float(acc1) / b * 100, b)
#             acc5_meter.update(float(acc5) / b * 100, b)
#             if idx % config.PRINT_FREQ == 0:
#                 logger.info(
#                     f'Test: [{idx}/{len(val_loader)}]\t'
#                     f'Acc@1: {acc1_meter.avg:.3f}\t'
#                 )
#     acc1_meter.sync()
#     acc5_meter.sync()
#     logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
#     return acc1_meter.avg


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)