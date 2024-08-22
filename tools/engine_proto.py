# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
from cProfile import label
import math
import os
import sys
import logging
import pickle
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from torch.nn.modules.loss import _Loss
import tools.utils as utils
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def train_one_epoch(model: torch.nn.Module, criterion: _Loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    tb_writer: SummaryWriter, iteration: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args=None,
                    set_training_mode=True,):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.log_print_freq

    logger = logging.getLogger("train")
    logger.info("Start train one epoch")
    it = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # print(f"Samples shape: {samples.shape}")
        # print(f"Targets shape: {targets.shape}")
        

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, auxi_item = model(samples)
            # print(f"Outputs shape: {outputs.shape}")

            loss = criterion(outputs, targets)

            if args.use_ppc_loss:
                ppc_cov_coe, ppc_mean_coe = args.ppc_cov_coe, args.ppc_mean_coe
                total_proto_act, cls_attn_rollout, original_fea_len = auxi_item[2], auxi_item[3], auxi_item[4]
                if hasattr(model, 'module'):
                    ppc_cov_loss, ppc_mean_loss = model.module.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
                else:
                    ppc_cov_loss, ppc_mean_loss = model.get_PPC_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)

                ppc_cov_loss = ppc_cov_coe * ppc_cov_loss
                ppc_mean_loss = ppc_mean_coe * ppc_mean_loss
                
                # Compute the orthogonality loss
                orth_coe = args.orth_coe
                orth_loss = model.get_orthogonal_loss(total_proto_act, cls_attn_rollout, original_fea_len, targets)
                orth_loss = orth_coe * orth_loss
                
                # Compute the separation & cluster loss
                separation_loss, cluster_loss = model.get_separation_and_cluster_loss(total_proto_act, original_fea_len, targets)
                
                                
                if epoch >= 20:
                    loss += (ppc_cov_loss + ppc_mean_loss)
                    loss += orth_loss
                    loss += (separation_loss + cluster_loss)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        tb_writer.add_scalars(
            main_tag="train/loss",
            tag_scalar_dict={
                "cls": loss.item(),
            },
            global_step=iteration+it
        )
        tb_writer.add_scalars(
            main_tag="train/orth_loss",
            tag_scalar_dict={
                "orth_loss": orth_loss.item(),
            },
            global_step=iteration+it
        )
        tb_writer.add_scalars(
            main_tag="train/cluster_loss",
            tag_scalar_dict={
                "cluster_loss": cluster_loss.item(),
            },
            global_step=iteration+it
        )
        tb_writer.add_scalars(
            main_tag="train/separation_loss",
            tag_scalar_dict={
                "separation_loss": separation_loss.item(),
            },
            global_step=iteration+it
        )
        if args.use_global and args.use_ppc_loss:
            tb_writer.add_scalars(
                main_tag="train/ppc_cov_loss",
                tag_scalar_dict={
                    "ppc_cov_loss": ppc_cov_loss.item(),
                },
                global_step=iteration+it
            )
            tb_writer.add_scalars(
                main_tag="train/ppc_mean_loss",
                tag_scalar_dict={
                    "ppc_mean_loss": ppc_mean_loss.item(),
                },
                global_step=iteration+it
            )
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_img_mask(data_loader, model, device, args):
    logger = logging.getLogger("get mask")
    logger.info("Get mask")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Get Mask:'

    # switch to evaluation mode
    model.eval()

    all_attn_mask = []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            cat_mask = model.get_attn_mask(images)
            all_attn_mask.append(cat_mask.cpu())
    all_attn_mask = torch.cat(all_attn_mask, dim=0) # (num, 2, 14, 14)
    if hasattr(model, 'module'):
        model.module.all_attn_mask = all_attn_mask
    else:
        model.all_attn_mask = all_attn_mask

@torch.no_grad()
def evaluate(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    all_token_attn, pred_labels = [], []
    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, auxi_items = model(images,)
            loss = criterion(output, target)
            
        # print(f"Images shape: {images.shape}")
        # print(f"Output shape: {output.shape}")
        # print(f"Target shape: {target.shape}")
        # print(f"Target: {target}")
        # print(f"First predicted label: {torch.argmax(output[0])}")
        # print(f"Target label: {target[0]}")


        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        _, pred = output.topk(k=1, dim=1)
        pred_labels.append(pred)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.use_global:
            global_acc1 = accuracy(auxi_items[2], target)[0]
            local_acc1 = accuracy(auxi_items[3], target)[0]
            metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
            metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
        all_token_attn.append(auxi_items[0])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def few_shot_evaluation(data_loader, model, device, args):
    logger = logging.getLogger("validate")
    logger.info("Start few shot validation")
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Few Shot Test:'

    # switch to evaluation mode
    model.eval()

    all_token_attn, pred_labels = [], []
    
    for support_images, support_labels, query_images, query_labels, _ in metric_logger.log_every(data_loader, 10, header):
        support_images = support_images.to(device, non_blocking=True)
        support_labels = support_labels.to(device, non_blocking=True)
        query_images = query_images.to(device, non_blocking=True)
        query_labels = query_labels.to(device, non_blocking=True)
        
        # compute the output of support images and query images   
        support_output, support_auxi_items = model(support_images)    
        query_output, query_auxi_items = model(query_images)
        
        # support_cls_token, support_img_tokens = model.img_token(support_images)
        (support_cls_token, support_img_tokens), (support_token_attn, support_cls_attn, _) = model.conv_features(support_images, model.reserve_layer_nums)
        (query_cls_token, query_img_tokens), (query_token_attn, query_cls_attn, _) = model.conv_features(query_images, model.reserve_layer_nums)
        
        # print(f"Support images shape and support labels shape: {support_images.shape}, {support_labels.shape}")
        # print(f"Query images shape and query labels shape: {query_images.shape}, {query_labels.shape}")
        
        # print(f"Support labels & query labels: {support_labels}, {query_labels}")
        # return
        
        # print(f"Support cls token shape: {support_cls_token.shape}")
        # print(f"Query cls token shape: {query_cls_token.shape}")
        # print(f"Support img tokens shape: {support_img_tokens.shape}")
        # print(f"Support token attn shape: {support_token_attn.shape}")
        # print(f"Support cls attn shape: {support_cls_attn.shape}")
        
        # print(f"Support cls token: {support_cls_token[0]}")
        
        # Using class token to compute the similarity between support and query images
        '''
        Support cls token: 25, 192, 1, 1
        Query cls token: 50, 192, 1, 1
        '''

        support_cls_token = torch.squeeze(support_cls_token)
        query_cls_token = torch.squeeze(query_cls_token)
        
        # Compute the cosine similarity matrix between support and query cls tokens
        sim_matrix = torch.zeros((support_cls_token.size(0), query_cls_token.size(0)))

        use_mean_of_prototypes = args.use_prototype_mean
        if use_mean_of_prototypes == False:
            for i, support in enumerate(support_cls_token):
                for j, query in enumerate(query_cls_token):
                    # print(f"Support cls token: {support.shape}")
                    sim_matrix[i, j] = torch.nn.functional.cosine_similarity(support.unsqueeze(0), query.unsqueeze(0))
        else:
            # Compute the mean cls token for each class in support set
            support_cls_token_dict = {}
            for i, label in enumerate(support_labels):
                if label.item() not in support_cls_token_dict:
                    support_cls_token_dict[label.item()] = []
                support_cls_token_dict[label.item()].append(support_cls_token[i])

            # Compute the mean of cls tokens for each class
            mean_support_cls_tokens = {}
            for label, tokens in support_cls_token_dict.items():
                mean_support_cls_tokens[label] = torch.stack(tokens).mean(dim=0)

            # Compute the cosine similarity matrix between mean support and query cls tokens
            sim_matrix = torch.zeros((len(mean_support_cls_tokens), query_cls_token.size(0)))

            for i, (label, mean_support) in enumerate(mean_support_cls_tokens.items()):
                for j, query in enumerate(query_cls_token):
                    sim_matrix[i, j] = torch.nn.functional.cosine_similarity(mean_support.unsqueeze(0), query.unsqueeze(0))
                            
        # Find the most similar support images for each query image
        predicted_labels = []
        for i in range(sim_matrix.size(1)):
            most_similar_support_index = torch.argmax(sim_matrix[:, i])
            most_similar_label = support_labels[most_similar_support_index]
            # print(f"Most similar support image for query image {i}: {most_similar_support_index}")
            # print(f"Most similar label: {most_similar_label}")
            predicted_labels.append(most_similar_label)
        
        # Compute the loss
        ## Compute softmax over negative distances
        # probs = torch.nn.functional.softmax(-sim_matrix, dim=0)
        # ## Compute cross-entropy loss
        # loss = torch.nn.functional.cross_entropy(probs, query_labels, reduction='mean')
        # metric_logger.meters["fsl_loss"].update(loss.item())
        
        # Compute the accuracy
        correct = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == query_labels[i]:
                correct += 1
                # print(f"Predicted label: {pred_labels[i]}, True label: {query_labels[i]}")
        
        fsl_accuracy = correct / len(predicted_labels)  
        batch_size = query_images.shape[0]              
        
        metric_logger.meters['fsl_acc'].update(fsl_accuracy , n=batch_size)

        # compute output
        with torch.cuda.amp.autocast():
            # output, auxi_items = model(support_images, support_labels, query_images, query_labels)
            loss = criterion(query_output, query_labels)

        acc1, acc5 = accuracy(query_output, query_labels, topk=(1, 5))
        _, pred = query_output.topk(k=1, dim=1)
        pred_labels.append(pred)

        # batch_size = query_images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        if args.use_global:
            global_acc1 = accuracy(query_auxi_items[2], query_labels)[0]
            local_acc1 = accuracy(query_auxi_items[3], query_labels)[0]
            metric_logger.meters['global_acc1'].update(global_acc1.item(), n=batch_size)
            metric_logger.meters['local_acc1'].update(local_acc1.item(), n=batch_size)
        all_token_attn.append(query_auxi_items[0])
    
    
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    
    # logger.info(f'Accuracy: {metric_logger.fsl_acc.global_avg * 100}%')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}