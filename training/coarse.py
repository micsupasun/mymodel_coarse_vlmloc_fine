"""Module for training the coarse cell-retrieval module
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
import collections

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp
import tqdm
import random
from models.coarse.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ContrastiveLoss
from training.losses import MNCLContrastiveLoss, SceneAwareHardNegativeLoss, SemanticDistillationLoss, RetrievalDistillationLoss  # CMMLoc++ additions
from training.utils import plot_retrievals
from training.checkpointing import build_training_state, save_training_state

"Training Process for global place recognition"
def _cell_symbolic_stats(cell):
    label_counts = {}
    color_label_counts = {}
    for obj in cell.objects:
        label_counts[obj.label] = label_counts.get(obj.label, 0) + 1
        key = (obj.get_color_text(), obj.label)
        color_label_counts[key] = color_label_counts.get(key, 0) + 1
    return label_counts, color_label_counts


def _pose_symbolic_requirements(pose):
    label_counts = {}
    color_label_counts = {}
    for descr in pose.descriptions:
        label_counts[descr.object_label] = label_counts.get(descr.object_label, 0) + 1
        key = (descr.object_color_text, descr.object_label)
        color_label_counts[key] = color_label_counts.get(key, 0) + 1
    return label_counts, color_label_counts


def _coverage_score(required_counts, available_counts):
    total = sum(required_counts.values())
    if total == 0:
        return 0.0
    matched = 0
    for key, required in required_counts.items():
        matched += min(required, available_counts.get(key, 0))
    return matched / total


def _model_rerank_candidate_ids(model, pose, candidate_cell_ids, candidate_scores, cells_dict):
    pose_labels, pose_color_labels = _pose_symbolic_requirements(pose)
    label_covs = []
    color_covs = []
    for cell_id in candidate_cell_ids:
        cell_labels, cell_color_labels = _cell_symbolic_stats(cells_dict[cell_id])
        label_covs.append(_coverage_score(pose_labels, cell_labels))
        color_covs.append(_coverage_score(pose_color_labels, cell_color_labels))

    device = model.device
    base_scores_t = torch.as_tensor(candidate_scores, device=device, dtype=torch.float32)
    label_covs_t = torch.as_tensor(label_covs, device=device, dtype=torch.float32)
    color_covs_t = torch.as_tensor(color_covs, device=device, dtype=torch.float32)
    rerank_scores = model.combine_rerank_scores(base_scores_t, label_covs_t, color_covs_t)
    rerank_order = torch.argsort(rerank_scores, descending=True).detach().cpu().numpy()
    return candidate_cell_ids[rerank_order]


def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))


    for i_batch, batch in pbar:

        optimizer.zero_grad(set_to_none=True)

        # Coarse training is small enough to run fully in float32, which is
        # more stable than AMP for this repo's contrastive/object encoder path.
        anchor = model.encode_text(batch["texts"])
        positive = model.encode_objects(batch["objects"], batch["object_points"])

        if lambda_teacher_distill > 0.0:
            with torch.no_grad():
                teacher_anchor = teacher_model.encode_text(batch["texts"])
                teacher_positive = teacher_model.encode_objects(batch["objects"], batch["object_points"])
        else:
            teacher_anchor = None
            teacher_positive = None

        # CMMLoc++: project into MNCL-style contrastive space
        z_anchor = model.project_text(anchor)
        z_positive = model.project_submaps(positive)
        if lambda_semantic_mncl > 0.0 or lambda_semantic_distill > 0.0:
            with torch.no_grad():
                semantic_positive = model.encode_cell_semantics(batch["objects"])
                z_semantic = model.project_semantics(semantic_positive)
        else:
            semantic_positive = None
            z_semantic = None

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            base_loss = criterion(anchor, positive, negative)
        else:
            base_loss = criterion(anchor, positive)

        # CMMLoc++: MNCL-style contrastive loss
        if lambda_mncl > 0.0:
            mncl_loss = mncl_criterion(z_anchor, z_positive)
            loss = base_loss + lambda_mncl * mncl_loss
        else:
            mncl_loss = torch.zeros((), device=anchor.device)
            loss = base_loss

        if lambda_semantic_mncl > 0.0:
            semantic_loss = semantic_mncl_criterion(z_anchor, z_semantic)
            loss = loss + lambda_semantic_mncl * semantic_loss
        else:
            semantic_loss = torch.zeros((), device=anchor.device)

        if lambda_semantic_distill > 0.0:
            semantic_distill_loss = semantic_distill_criterion(z_positive, z_semantic)
            loss = loss + lambda_semantic_distill * semantic_distill_loss
        else:
            semantic_distill_loss = torch.zeros((), device=anchor.device)

        if lambda_scene_hardneg > 0.0:
            scene_hardneg_loss = scene_hardneg_criterion(z_anchor, z_positive, batch["scene_names"])
            loss = loss + lambda_scene_hardneg * scene_hardneg_loss
        else:
            scene_hardneg_loss = torch.zeros((), device=anchor.device)

        if lambda_teacher_distill > 0.0:
            teacher_distill_loss = teacher_distill_criterion(
                anchor, positive, teacher_anchor, teacher_positive
            )
            loss = loss + lambda_teacher_distill * teacher_distill_loss
        else:
            teacher_distill_loss = torch.zeros((), device=anchor.device)

        if lambda_reranker > 0.0 and getattr(model, "reranker_head", None) is not None:
            base_logits = torch.matmul(anchor, positive.t()) / max(float(args.temperature), 1e-6)
            rerank_logits = model.compute_reranker_logits(base_logits, batch["poses"], batch["objects"])
            labels = torch.arange(rerank_logits.size(0), device=rerank_logits.device)
            rerank_loss_i2s = F.cross_entropy(rerank_logits, labels)
            rerank_loss_s2i = F.cross_entropy(rerank_logits.t(), labels)
            rerank_loss = 0.5 * (rerank_loss_i2s + rerank_loss_s2i)
            loss = loss + lambda_reranker * rerank_loss
        else:
            rerank_loss = torch.zeros((), device=anchor.device)

        if not torch.isfinite(anchor).all():
            raise RuntimeError("Non-finite anchor/text embeddings detected in coarse training.")
        if not torch.isfinite(positive).all():
            raise RuntimeError("Non-finite positive/submap embeddings detected in coarse training.")
        if semantic_positive is not None and not torch.isfinite(semantic_positive).all():
            raise RuntimeError("Non-finite semantic cell embeddings detected in coarse training.")
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite coarse loss detected before backward.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())
        pbar.set_postfix(
            loss=f"{loss.item():0.3f}",
            base=f"{base_loss.item():0.3f}",
            mncl=f"{mncl_loss.item():0.3f}",
            semantic=f"{semantic_loss.item():0.3f}",
            distill=f"{semantic_distill_loss.item():0.3f}",
            hardneg=f"{scene_hardneg_loss.item():0.3f}",
            teacher=f"{teacher_distill_loss.item():0.3f}",
            rerank=f"{rerank_loss.item():0.3f}",
        )
        torch.cuda.empty_cache()

    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False):
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):
        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []
    use_model_reranker = getattr(args, "use_trainable_reranker", False) and getattr(model, "reranker_head", None) is not None
    candidate_count = max(np.max(args.top_k), getattr(args, "trainable_rerank_topn", 0))
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0:candidate_count]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        if use_model_reranker and candidate_count > np.max(args.top_k):
            retrieved_cell_ids = _model_rerank_candidate_ids(
                model,
                dataloader.dataset.all_poses[query_idx],
                retrieved_cell_ids,
                scores[sorted_indices],
                cells_dict,
            )
        retrieved_cell_ids = retrieved_cell_ids[0 : np.max(args.top_k)]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    elif return_distance:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings, np.stack(dists_list), np.stack(scores_list)
    else:
        return accuracies, accuracies_close, top_retrievals

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

    
    dataset_path = args.base_path.rstrip("/\\")          # remove trailing / or \
    dataset_name = osp.basename(dataset_path)            # e.g. "k360_30-10_scG_pd10_pc4_spY_all"
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")

    # --- Robust checkpoint directory creation ---
    checkpoint_dir = osp.join(".", "checkpoints", dataset_name, folder_name)
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)


    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
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

        dataset_train = Kitti360CoarseDatasetMulti(
            args.base_path,
            SCENE_NAMES_TRAIN,
            train_transform,
            shuffle_hints=True,
            flip_poses=True,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle= args.shuffle,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform,)

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

        dataset_test = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform,)

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
            num_workers=args.cpus,
            pin_memory=torch.cuda.is_available(),
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(False)
    use_amp = False

    lr = args.learning_rate

    dict_loss = {lr: []}
    dict_acc = {k: [] for k in args.top_k}
    dict_acc_val = {k: [] for k in args.top_k}
    dict_acc_val_close = {k: [] for k in args.top_k}
    dict_acc_test = {k: [] for k in args.top_k}
    dict_acc_test_close = {k: [] for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path_val = None

    model = CellRetrievalNetwork(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args,
        )
    start_epoch = 1
    if args.continue_path:
        loaded_payload = torch.load(args.continue_path, map_location=torch.device("cpu"))
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            model.load_state_dict(loaded_payload["model_state"], strict=False)
        else:
            model.load_state_dict(loaded_payload, strict = False)

    model.to(device)

    teacher_model = None
    if args.lambda_teacher_distill > 0.0 and not args.teacher_coarse_path:
        raise ValueError("Teacher distillation requires --teacher_coarse_path.")
    if args.teacher_coarse_path and args.lambda_teacher_distill > 0.0:
        teacher_model = CellRetrievalNetwork(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args,
        )
        teacher_state = torch.load(args.teacher_coarse_path, map_location=torch.device("cpu"))
        teacher_model.load_state_dict(teacher_state, strict=False)
        teacher_model.to(device)
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.ranking_loss == "pairwise":
        criterion = PairwiseRankingLoss(margin=args.margin)
    if args.ranking_loss == "hardest":
        criterion = HardestRankingLoss(margin=args.margin)
    if args.ranking_loss == "triplet":
        criterion = nn.TripletMarginLoss(margin=args.margin)
    if args.ranking_loss == "contrastive":
        criterion = ContrastiveLoss(temperature=args.temperature)
    # ================= CMMLoc++: MNCL-style loss and weight =================
    mncl_criterion = MNCLContrastiveLoss(temperature=args.temperature)
    lambda_mncl = getattr(args, "lambda_mncl", 0.0)
    semantic_mncl_criterion = MNCLContrastiveLoss(temperature=args.temperature)
    lambda_semantic_mncl = getattr(args, "lambda_semantic_mncl", 0.0)
    semantic_distill_criterion = SemanticDistillationLoss()
    lambda_semantic_distill = getattr(args, "lambda_semantic_distill", 0.0)
    teacher_distill_criterion = RetrievalDistillationLoss(temperature=args.teacher_temperature)
    lambda_teacher_distill = getattr(args, "lambda_teacher_distill", 0.0)
    lambda_reranker = getattr(args, "lambda_reranker", 0.0)
    scene_hardneg_criterion = SceneAwareHardNegativeLoss(
        temperature=args.temperature,
        same_scene_negative_weight=args.same_scene_negative_weight,
        hard_negative_topk=args.hard_negative_topk,
        hard_negative_weight=args.hard_negative_weight,
    )
    lambda_scene_hardneg = getattr(args, "lambda_scene_hardneg", 0.0)
    # =======================================================================

    if args.lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    else:
        raise TypeError

    if args.continue_path:
        loaded_payload = torch.load(args.continue_path, map_location=torch.device("cpu"))
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            start_epoch = int(loaded_payload.get("epoch", 0)) + 1
            best_val_accuracy = loaded_payload.get("best_metric", best_val_accuracy)
            best_model_path = loaded_payload.get("best_model_path") or None
            last_model_save_path_val = best_model_path
            optimizer_state = loaded_payload.get("optimizer_state")
            scheduler_state = loaded_payload.get("scheduler_state")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            if scheduler_state is not None and scheduler is not None:
                scheduler.load_state_dict(scheduler_state)
            print(f"Resuming coarse training from epoch {start_epoch}.")

    resume_state_path = osp.join(checkpoint_dir, "resume_training_state.pth")

    for epoch in range(start_epoch, args.epochs + 1):
        # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
        
        loss, train_batches = train_epoch(model, dataloader_train, args)
        train_eval_frequency = max(int(getattr(args, "train_eval_frequency", 1)), 0)
        run_train_eval = (
            not args.skip_train_eval
            and train_eval_frequency > 0
            and (epoch == start_epoch or epoch % train_eval_frequency == 0)
        )
        if run_train_eval:
            train_acc, train_acc_close, train_retrievals = eval_epoch(
                model, dataloader_train, args
            )
        else:
            train_acc = {k: float("nan") for k in args.top_k}
            train_acc_close = {k: float("nan") for k in args.top_k}
            train_retrievals = None
            if args.skip_train_eval or train_eval_frequency == 0:
                print("Skipping full train-set retrieval evaluation.")
            else:
                print(
                    f"Skipping full train-set retrieval evaluation this epoch "
                    f"(--train_eval_frequency {train_eval_frequency})."
                )
        val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
        test_acc, test_acc_close, test_retrievals = eval_epoch(model, dataloader_test, args)

        key = lr
        dict_loss[key].append(loss)
        for k in args.top_k:
            if run_train_eval:
                dict_acc[k].append(train_acc[k])
            dict_acc_val[k].append(val_acc[k])
            dict_acc_val_close[k].append(val_acc_close[k])
            dict_acc_test[k].append(test_acc[k])
            dict_acc_test_close[k].append(test_acc_close[k])

        scheduler.step()
        print(f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} train-acc: ", end="")
        if not run_train_eval:
            print("skipped ", end="")
        else:
            for k, v in train_acc.items():
                print(f"{k}-{v:0.3f} ", end="")
        print("val-acc: ", end="")
        for k, v in val_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc-close: ", end="")
        for k, v in val_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")

        print("test-acc: ", end="")
        for k, v in test_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("test-acc-close: ", end="")
        for k, v in test_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("\n", flush=True)

        # Saving best model
        acc_val = val_acc[max(args.top_k)]
        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/coarse_cont{cont}_epoch{epoch}_acc{acc_val:0.3f}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_loss-{args.ranking_loss}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print(f"Saving model at {acc_val:0.2f} to {model_path}")
            
            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path_val is not None
                    and last_model_save_path_val != model_path
                    and osp.isfile(last_model_save_path_val)
                ):  
                    print("Removing", last_model_save_path_val)
                    os.remove(last_model_save_path_val)
                
                last_model_save_path_val = model_path
                
            except Exception as e:
                print(f"Error saving model!", str(e))
            best_val_accuracy = acc_val

        resume_payload = build_training_state(
            model_state=model.state_dict(),
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_val_accuracy,
            best_model_path=last_model_save_path_val or "",
        )
        save_training_state(resume_state_path, resume_payload)

                           

