"""Module for training the fine matching module
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections

import torch_geometric.transforms as T
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp
import tqdm
import cv2
from models.fine.cross_matcher import CrossMatch

from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
from datapreparation.kitti360pose.utils import CLASS_TO_COLOR
from datapreparation.kitti360pose.imports import (
    Object3d,
    Cell,
    Pose,
    DescriptionPoseCell,
    DescriptionBestCell,
)
# from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error2
from training.checkpointing import build_training_state, load_training_state, save_training_state


"Training Process for fine localization"
def train_epoch(model, dataloader, args):
    model.train()

    offset_lambda = args.offset_lambda
    
    stats = EasyDict(
        loss=[],
        loss_offsets=[],
        # recall=[],
        # precision=[],
        # pose_mid=[],
        # pose_mean=[],
        pose_offsets=[],
    )
        
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        texts = batch["texts"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            output= model(batch["objects"], texts, batch["object_points"])
            if not torch.isfinite(output).all():
                raise RuntimeError("Non-finite fine-stage offsets detected before loss computation.")
            offsets_target = np.asarray(batch["offsets"], dtype=np.float32)
            loss_offsets = criterion_offsets(
                output, torch.tensor(offsets_target, dtype=torch.float, device=device)
            )
            loss =  offset_lambda * loss_offsets
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite fine-stage loss detected before backward.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()


        stats.loss.append(loss.item())
        stats.loss_offsets.append(loss_offsets.item())
        error = calc_pose_error2(
                batch["objects"],
                # output.matches0.detach().cpu().numpy(),
                batch["poses"],
                offsets=output.detach().cpu().numpy(),
            )
        stats.pose_offsets.append(
            error
        )
        pbar.set_postfix(loss = loss.item(), error = error)


    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


@torch.no_grad()
def eval_epoch(model, dataloader, args,):
    model.eval() 
    
    stats = EasyDict(
        # recall=[],
        # precision=[],
        # pose_mid=[],
        # pose_mean=[],
        pose_offsets=[],
    )
    
    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        
        texts = batch["texts"]
        with torch.cuda.amp.autocast(enabled=use_amp):
            output= model(batch["objects"], texts, batch["object_points"])
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite fine-stage offsets detected during evaluation.")
        stats.pose_offsets.append(
            calc_pose_error2(
                batch["objects"],
                # output.matches0.detach().cpu().numpy(),
                batch["poses"],
                offsets=output.detach().cpu().numpy(),
            )
        )

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

def seed_everything(seed: int): 
   random.seed(seed) 
   os.environ['PYTHONHASHSEED'] = str(seed) 
   np.random.seed(seed) 
   torch.manual_seed(seed) 
   torch.cuda.manual_seed(seed) 
   torch.backends.cudnn.deterministic = True 
   torch.backends.cudnn.benchmark = True 

if __name__ == "__main__":
    seed_everything(42)
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    # --- Robust dataset name ---
    dataset_path = args.base_path.rstrip("/\\")      # remove trailing / or \
    dataset_name = osp.basename(dataset_path)        # e.g. "k360_30-10_scG_pd10_pc4_spY_all"
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")

    # --- Create checkpoint directory safely ---
    checkpoint_dir = osp.join(".", "checkpoints", dataset_name, folder_name)
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    """
    Create data loaders
    """
    if args.dataset == "K360":
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False,
            pmc_prob = args.pmc_prob,
            pmc_threshold = args.pmc_threshold,
        ) 
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            shuffle=args.shuffle,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args,)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

        dataset_test = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform, args,)
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print("device:", device, device_name)
    torch.autograd.set_detect_anomaly(False)
    use_amp = bool(args.use_amp and device.type == "cuda")
    if not (args.prealign_pointnet_path and args.prealign_color_path and args.prealign_mlp_path):
        raise ValueError(
            "Fine training requires --prealign_pointnet_path, --prealign_color_path, and --prealign_mlp_path."
        )

    best_val_offset = 1000  # Measured by mean of recall and precision
    last_model_save_path = None

    lr = args.learning_rate 

    train_stats_loss = {lr: []}
    train_stats_loss_offsets = {lr: []}
    train_stats_pose_offsets = {lr: []}

    val_stats_pose_offsets = {lr: []}
    test_stats_pose_offsets = {lr: []}

    model = CrossMatch(
        dataset_train.get_known_classes(),
        COLOR_NAMES_K360,
        args,
    )
    model.to(device)
    start_epoch = 1
    resume_state_path = osp.join(checkpoint_dir, "resume_training_state.pth")
    if args.continue_path:
        print("Loading fine checkpoint from", args.continue_path)
        loaded_payload = torch.load(args.continue_path, map_location=device)
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            model.load_state_dict(loaded_payload["model_state"], strict=False)
        else:
            model.load_state_dict(loaded_payload, strict=False)

    criterion_offsets = nn.MSELoss()

    # Warm-up
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    num_epoch_warmup = 3
    if args.continue_path:
        loaded_payload = torch.load(args.continue_path, map_location=device)
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            start_epoch = int(loaded_payload.get("epoch", 0)) + 1
            best_val_offset = loaded_payload.get("best_metric", best_val_offset)
            last_model_save_path = loaded_payload.get("best_model_path") or None
            extra_state = loaded_payload.get("extra_state", {})
            optimizer_state = loaded_payload.get("optimizer_state")
            scheduler_state = loaded_payload.get("scheduler_state")
            scaler_state = loaded_payload.get("scaler_state")
            current_phase = extra_state.get("phase", "warmup")
            if current_phase == "main":
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                if args.lr_scheduler == "exponential":
                    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
                elif args.lr_scheduler == "step":
                    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
                else:
                    raise TypeError
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None and scheduler is not None:
                scheduler.load_state_dict(scheduler_state)
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
            print(f"Resuming fine training from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == num_epoch_warmup:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            if args.lr_scheduler == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
            elif args.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
            else:
                raise TypeError

        train_out = train_epoch(model, dataloader_train, args,)

        train_stats_loss[lr].append(train_out.loss)
        train_stats_loss_offsets[lr].append(train_out.loss_offsets)
        train_stats_pose_offsets[lr].append(train_out.pose_offsets)

        val_out = eval_epoch(model, dataloader_val, args,)  # CARE: which loader for val!
        val_stats_pose_offsets[lr].append(val_out.pose_offsets)

        print()

        test_out = eval_epoch(model, dataloader_test, args,)  # CARE: which loader for test!
        test_stats_pose_offsets[lr].append(test_out.pose_offsets)

        print()

        if scheduler:
            scheduler.step()

        print(
            (
                f"\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} "
                f"t-offset {train_out.pose_offsets:0.3f} "
                f"v-offset {val_out.pose_offsets:0.3f} "
                f"e-offset {test_out.pose_offsets:0.3f} "
            ),
            flush=True,
        )

        offset = np.mean(val_out.pose_offsets)
        if offset < best_val_offset:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/fine_cont{cont}_epoch{epoch}_offset{offset:0.3f}_lr{args.learning_rate}_obj-{args.num_mentioned}-{args.pad_size}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_f-{feats}.pth"
            os.makedirs(osp.dirname(model_path), exist_ok=True)

            print("Saving model to", model_path)
            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path is not None
                    and last_model_save_path != model_path
                    and osp.isfile(last_model_save_path)
                ):
                    print("Removing", last_model_save_path)
                    os.remove(last_model_save_path)
                last_model_save_path = model_path
            except Exception as e:
                print("Error saving model!", str(e))
            best_val_offset = offset

        current_phase = "main" if epoch >= num_epoch_warmup else "warmup"
        resume_payload = build_training_state(
            model_state=model.state_dict(),
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_val_offset,
            best_model_path=last_model_save_path or "",
            extra_state={"phase": current_phase},
        )
        save_training_state(resume_state_path, resume_payload)
