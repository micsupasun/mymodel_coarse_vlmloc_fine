import sys
from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict
from copy import deepcopy
import pickle

import torch
from torch.utils.data import DataLoader
import time
import random
from scipy.spatial.distance import cdist

from models.coarse.cell_retrieval import CellRetrievalNetwork
from models.fine.cross_matcher import CrossMatch

from evaluation.args import parse_arguments
from evaluation.checkpoint_loading import audit_and_load_checkpoint
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360

from training.coarse import eval_epoch as eval_epoch_retrieval
from training.utils import plot_retrievals

import torch_geometric.transforms as T
import tqdm
import json

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


def rerank_candidate_cells(pose, candidate_cell_ids, candidate_scores, cells_dict, args, model=None):
    if args.rerank_topn <= 0:
        return candidate_cell_ids

    topn = min(args.rerank_topn, len(candidate_cell_ids))
    pose_labels, pose_color_labels = _pose_symbolic_requirements(pose)

    label_covs = []
    color_covs = []
    rerank_scores = []
    for idx in range(topn):
        cell = cells_dict[candidate_cell_ids[idx]]
        cell_labels, cell_color_labels = _cell_symbolic_stats(cell)
        label_cov = _coverage_score(pose_labels, cell_labels)
        color_cov = _coverage_score(pose_color_labels, cell_color_labels)
        label_covs.append(label_cov)
        color_covs.append(color_cov)
        rerank_score = (
            args.rerank_base_weight * candidate_scores[idx]
            + args.rerank_label_weight * label_cov
            + args.rerank_color_weight * color_cov
        )
        rerank_scores.append(rerank_score)

    if (
        model is not None
        and args.use_model_reranker
        and getattr(model, "reranker_head", None) is not None
    ):
        device = model.device
        base_scores_t = torch.as_tensor(candidate_scores[:topn], device=device, dtype=torch.float32)
        label_covs_t = torch.as_tensor(label_covs, device=device, dtype=torch.float32)
        color_covs_t = torch.as_tensor(color_covs, device=device, dtype=torch.float32)
        rerank_scores = model.combine_rerank_scores(base_scores_t, label_covs_t, color_covs_t)
        rerank_order = torch.argsort(rerank_scores, descending=True).detach().cpu().numpy()
    else:
        rerank_scores = np.asarray(rerank_scores, dtype=np.float32)
        rerank_order = np.argsort(-rerank_scores)

    reranked_ids = list(candidate_cell_ids[:topn][rerank_order])
    reranked_ids.extend(list(candidate_cell_ids[topn:]))
    return np.asarray(reranked_ids)


@torch.no_grad()
def encode_retrieval_embeddings(model, dataloader, args):
    """Encode query texts and database cells for coarse retrieval."""
    model.eval()

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim), dtype=np.float32)
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim), dtype=np.float32)
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    index_offset = 0
    for batch in tqdm.tqdm(dataloader):
        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    return {
        "cell_encodings": cell_encodings,
        "db_cell_ids": db_cell_ids,
        "text_encodings": text_encodings,
        "query_cell_ids": query_cell_ids,
        "query_poses_w": query_poses_w,
        "cells_dict": cells_dict,
        "cell_size": cell_size,
    }


def _zscore_scores(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-6:
        return scores - mean
    return (scores - mean) / std


def _rrf_scores(scores, weight, rrf_k):
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(len(scores), dtype=np.int32)
    return weight / (rrf_k + ranks + 1.0)


@torch.no_grad()
def run_coarse_fused(model_primary, model_secondary, dataloader, args):
    """Fuse two coarse checkpoints and evaluate the combined retrieval ranking."""
    primary = encode_retrieval_embeddings(model_primary, dataloader, args)
    secondary = encode_retrieval_embeddings(model_secondary, dataloader, args)

    if not np.array_equal(primary["db_cell_ids"], secondary["db_cell_ids"]):
        raise RuntimeError("Primary and secondary coarse models produced different database cell ordering.")
    if not np.array_equal(primary["query_cell_ids"], secondary["query_cell_ids"]):
        raise RuntimeError("Primary and secondary coarse models produced different query ordering.")

    cells_dict = primary["cells_dict"]
    cell_size = primary["cell_size"]
    db_cell_ids = primary["db_cell_ids"]
    query_cell_ids = primary["query_cell_ids"]
    query_poses_w = primary["query_poses_w"]

    top_retrievals = {}
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    for query_idx in range(len(primary["text_encodings"])):
        scores_primary = primary["cell_encodings"] @ primary["text_encodings"][query_idx]
        scores_secondary = secondary["cell_encodings"] @ secondary["text_encodings"][query_idx]

        if args.coarse_fusion == "rrf":
            combined_scores = _rrf_scores(
                scores_primary, args.coarse_weight_primary, args.rrf_k
            ) + _rrf_scores(scores_secondary, args.coarse_weight_secondary, args.rrf_k)
        else:
            combined_scores = (
                args.coarse_weight_primary * _zscore_scores(scores_primary)
                + args.coarse_weight_secondary * _zscore_scores(scores_secondary)
            )

        candidate_count = max(np.max(args.top_k), args.rerank_topn)
        sorted_indices = np.argsort(-combined_scores)[0 : candidate_count]
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        retrieved_cell_ids = rerank_candidate_cells(
            dataloader.dataset.all_poses[query_idx],
            retrieved_cell_ids,
            combined_scores[sorted_indices],
            cells_dict,
            args,
            model_primary,
        )[0 : np.max(args.top_k)]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    print("Exact-cell Retrieval Recall@K:")
    print(accuracies)
    print("Legacy cell-center <= cell_size/2 diagnostic Recall@K:")
    print(accuracies_close)

    sample_accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(top_retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [cells_dict[cell_id] for cell_id in top_retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)
        for k in args.top_k:
            for t in args.threshs:
                sample_accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            sample_accuracies[k][t] = np.mean(sample_accuracies[k][t])

    retrievals = [top_retrievals[idx].tolist() for idx in range(len(top_retrievals))]
    return retrievals, sample_accuracies


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    retrieval_accuracies, retrieval_accuracies_close, retrieval_dict, cell_encodings, text_encodings = eval_epoch_retrieval(
        model, dataloader, args, return_encodings=True
    )

    if args.rerank_topn > 0:
        db_cell_ids = np.array([cell.id for cell in dataloader.dataset.all_cells], dtype="<U32")
        reranked = {}
        for query_idx in range(len(text_encodings)):
            scores = cell_encodings[:] @ text_encodings[query_idx]
            candidate_count = max(np.max(args.top_k), args.rerank_topn)
            sorted_indices = np.argsort(-scores)[0 : candidate_count]
            candidate_ids = db_cell_ids[sorted_indices]
            reranked_ids = rerank_candidate_cells(
                dataloader.dataset.all_poses[query_idx],
                candidate_ids,
                scores[sorted_indices],
                all_cells_dict,
                args,
                model,
            )
            reranked[query_idx] = reranked_ids[0 : np.max(args.top_k)]

        retrievals = [reranked[idx].tolist() for idx in range(len(reranked))]

        retrieval_accuracies = {k: [] for k in args.top_k}
        retrieval_accuracies_close = {k: [] for k in args.top_k}
        cell_size = dataloader.dataset.all_cells[0].cell_size
        query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])
        query_cell_ids = np.array([pose.cell_id for pose in dataloader.dataset.all_poses], dtype="<U32")

        for query_idx in range(len(retrievals)):
            retrieved_cell_ids = retrievals[query_idx]
            target_cell_id = query_cell_ids[query_idx]
            for k in args.top_k:
                retrieval_accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
            target_pose_w = query_poses_w[query_idx]
            retrieved_cell_poses = [
                all_cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
            ]
            dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
            for k in args.top_k:
                retrieval_accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

        for k in args.top_k:
            retrieval_accuracies[k] = np.mean(retrieval_accuracies[k])
            retrieval_accuracies_close[k] = np.mean(retrieval_accuracies_close[k])
    else:
        retrievals = [retrieval_dict[idx].tolist() for idx in range(len(retrieval_dict))]

    print("Exact-cell Retrieval Recall@K:")
    print(retrieval_accuracies)
    print("Legacy cell-center <= cell_size/2 diagnostic Recall@K:")
    print(retrieval_accuracies_close)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    # np.save("retrievals_test_s.npy", retrievals)
    # retrievals = np.load("retrievals_test_s.npy", allow_pickle=True)
    return retrievals, accuracies


@torch.no_grad()
def run_fine(model, retrievals, dataloader, args, transform_fine):
    # A batch in this dataset contains max(top_k) times the pose vs. each of the max(top_k) top-cells.

    # import pdb; pdb.set_trace()
    model.eval()
    dataset_topk = Kitti360TopKDataset(
        dataloader.dataset.all_poses, dataloader.dataset.all_cells, retrievals, transform_fine, args,)


    num_samples = max(args.top_k)

    t0 = time.time()
    # Obtain the matches, offsets and confidences for each pose vs. its top-cells
    # Using a dataloader does not make it much faster ;)
    matches = []
    offsets = []
    confidences = []
    cell_ids = []
    poses_w = []
    
    t0 = time.time()
    pbar = tqdm.tqdm(enumerate(dataset_topk), total = len(dataset_topk))
    for i_sample, sample in pbar:
        output = model(sample["objects"], sample["texts"], sample["object_points"])
        offsets.append(output.detach().cpu().numpy())
        cell_ids.append([cell.id for cell in sample["cells"]])
        poses_w.append(sample["poses"][0].pose_w)
    print(f"Ran matching for {len(dataset_topk)} queries in {time.time() - t0:0.2f}.")

    assert len(offsets) == len(retrievals)
    cell_ids = np.array(cell_ids)

    t1 = time.time()
    print("ela:", t1 - t0)

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}

    # Gather the accuracies for each sample
    # accuracies_mean = {k: {t: [] for t in args.threshs} for k in args.top_k}
    accuracies_offset = {k: {t: [] for t in args.threshs} for k in args.top_k}
    # accuracies_mean_conf = {1: {t: [] for t in args.threshs}}
    save_offsets = []
    for i_sample in tqdm.tqdm(range(len(retrievals))):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        # sample_matches = matches[i_sample]
        sample_offsets = offsets[i_sample]
        # sample_confidences = confidences[i_sample]

        if not np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample]):
            print()
            print([cell.id for cell in top_cells])
            print(cell_ids[i_sample])

        assert np.all(np.array([cell.id for cell in top_cells]) == cell_ids[i_sample])
        assert np.allclose(pose.pose_w, poses_w[i_sample])

        # Get objects, matches and offsets for each of the top-cells

        pos_in_cells_offsets = []
        for i_cell in range(len(top_cells)):
            # Copy the cell and pad it again, as the fine model might have matched a padding-object
            cell = deepcopy(top_cells[i_cell])
            while len(cell.objects) < args.pad_size:
                cell.objects.append(Object3d.create_padding())

            # cell_matches = sample_matches[i_cell]
            cell_offsets = sample_offsets[i_cell]
            # pos_in_cells_mean.append(
            #     get_pos_in_cell(cell.objects, cell_matches, np.zeros_like(cell_offsets))
            # )
            pos_in_cells_offsets.append(cell_offsets)
        # pos_in_cells_mean = np.array(pos_in_cells_mean)
        pos_in_cells_offsets = np.array(pos_in_cells_offsets)

        # accs_mean = calc_sample_accuracies(
        #     pose, top_cells, pos_in_cells_mean, args.top_k, args.threshs
        # )
        accs_offsets = calc_sample_accuracies(
            pose, top_cells, pos_in_cells_offsets, args.top_k, args.threshs
        )

        for k in args.top_k:
            for t in args.threshs:
                # accuracies_mean[k][t].append(accs_mean[k][t])
                accuracies_offset[k][t].append(accs_offsets[k][t])
                # accuracies_mean_conf[1][t].append(accs_mean_conf[1][t])

    for k in args.top_k:
        for t in args.threshs:
            # accuracies_mean[k][t] = np.mean(accuracies_mean[k][t])
            accuracies_offset[k][t] = np.mean(accuracies_offset[k][t])
            # accuracies_mean_conf[1][t] = np.mean(accuracies_mean_conf[1][t])

    return accuracies_offset

def seed_everything(seed: int): 
   random.seed(seed) 
   os.environ['PYTHONHASHSEED'] = str(seed) 
   np.random.seed(seed) 
   torch.manual_seed(seed) 
   torch.cuda.manual_seed(seed) 
   torch.backends.cudnn.deterministic = True 
   torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":
    seed_everything(42)
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("device:", device, device_name)
    
    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.no_pc_augment_fine:
        transform_fine = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform_fine = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])


    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
        )
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # dataset_cell_only = dataset_retrieval.get_cell_dataset()

    # Load models
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
            )
    audit_and_load_checkpoint(
        model_coarse,
        args.path_coarse,
        f"{args.path_coarse}.audit.json",
        backend="legacy_pipeline_coarse",
        allowed_missing_prefixes=("language_encoder.llm_model.",),
    )
    model_coarse.to(device)

    model_coarse_secondary = None
    if args.path_coarse_secondary:
        model_coarse_secondary = CellRetrievalNetwork(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            args,
        )
        audit_and_load_checkpoint(
            model_coarse_secondary,
            args.path_coarse_secondary,
            f"{args.path_coarse_secondary}.audit.json",
            backend="legacy_pipeline_coarse_secondary",
            allowed_missing_prefixes=("language_encoder.llm_model.",),
        )
        model_coarse_secondary.to(device)

    # if not hasattr(model_retrieval.language_encoder, "use_attn"):
    #     model_retrieval.language_encoder.use_attn = True #　False
    # if not hasattr(model_retrieval, "obj_attn"):
    #     model_retrieval.obj_attn = None


    if args.path_fine:
        model_fine = CrossMatch(
            KNOWN_CLASS,
            COLOR_NAMES_K360,
            args,
        )
        audit_and_load_checkpoint(
            model_fine,
            args.path_fine,
            f"{args.path_fine}.audit.json",
            backend="legacy_pipeline_fine",
            allowed_missing_prefixes=("language_encoder.llm_model.",),
        )
        model_fine.to(device)
    
    # eval_conf(model_matching, dataset_retrieval)
    # quit()

    # # Run coarse
    if model_coarse_secondary is not None:
        retrievals, coarse_accuracies = run_coarse_fused(
            model_coarse, model_coarse_secondary, dataloader_retrieval, args
        )
    else:
        retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

    if args.coarse_only or not args.path_fine:
        sys.exit(0)

    accuracies_offsets = run_fine(
        model_fine, retrievals, dataloader_retrieval, args, transform_fine
    )
    print_accuracies(accuracies_offsets, "Fine")
