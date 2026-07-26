from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import math

from datapreparation.kitti360pose.imports import Object3d, Pose, Cell

from models.fine.cross_matcher import get_pos_in_cell, get_pos_in_cell_intersect


class MatchingLoss(nn.Module):
    def __init__(self):
        """Matching loss for SuperGlue-based matching training"""
        super(MatchingLoss, self).__init__()
        self.eps = 1e-3

    # Matches as list[tensor ∈ [M1, 2], tensor ∈ [M2, 2], ...] for Mi: i ∈ [1, batch_size]
    def forward(self, P, all_matches):
        assert len(P.shape) == 3
        assert len(all_matches[0].shape) == 2 and all_matches[0].shape[-1] == 2
        assert len(P) == len(all_matches)
        batch_losses = []
        for i in range(len(all_matches)):
            matches = all_matches[i]
            cell_losses = -torch.log(P[i, matches[:, 0], matches[:, 1]])
            batch_losses.append(torch.mean(cell_losses))

        return torch.mean(torch.stack(batch_losses))


def calc_recall_precision(batch_gt_matches, batch_matches0, batch_matches1):
    assert len(batch_gt_matches) == len(batch_matches0) == len(batch_matches1)
    all_recalls = []
    all_precisions = []

    for idx in range(len(batch_gt_matches)):
        gt_matches, matches0, matches1 = (
            batch_gt_matches[idx],
            batch_matches0[idx],
            batch_matches1[idx],
        )
        gt_matches = gt_matches.tolist()

        recall = []
        for i, j in gt_matches:
            recall.append(matches0[i] == j or matches1[j] == i)

        precision = []
        for i, j in enumerate(matches0):
            if j >= 0:
                precision.append(
                    [i, j] in gt_matches
                )  # CARE: this only works as expected after tolist()

        recall = np.mean(recall) if len(recall) > 0 else 0.0
        precision = np.mean(precision) if len(precision) > 0 else 0.0
        all_recalls.append(recall)
        all_precisions.append(precision)

    return np.mean(all_recalls), np.mean(all_precisions)


def calc_pose_error_intersect(objects, matches0, poses: List[Pose], directions):
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    errors = []
    for i_sample in range(batch_size):
        pose_prediction = get_pos_in_cell_intersect(
            objects[i_sample], matches0[i_sample], directions[i_sample]
        )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)


def calc_pose_error(
    objects, matches0, poses: List[Pose], offsets=None, use_mid_pred=False, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)
    else:
        offsets = np.zeros(
            (batch_size, pad_size, 2)
        )  # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    for i_sample in range(batch_size):
        if use_mid_pred:
            pose_prediction = np.array((0.5, 0.5))
        else:
            pose_prediction = get_pos_in_cell(
                objects[i_sample], matches0[i_sample], offsets[i_sample]
            )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))

    if return_samples:
        return errors
    else:
        return np.mean(errors)
    
def calc_pose_error2(
    objects, poses: List[Pose], offsets=None, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(poses)
    # assert isinstance(poses[0], Pose)

    # batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)
    else:
        raise TypeError
        offsets = np.zeros(
            (batch_size, pad_size, 2)
        )  # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    batch_size = len(poses)
    for i_sample in range(batch_size):
        # if use_mid_pred:
        #     pose_prediction = np.array((0.5, 0.5))
        # else:
        #     pose_prediction = get_pos_in_cell(
        #         objects[i_sample], matches0[i_sample], offsets[i_sample]
        #     )
        # if not is_degree:
        errors.append(np.linalg.norm(poses[i_sample] - offsets[i_sample]))
        # else:
        #     length, theta = offsets[i_sample]
        #     y = length * math.sin(theta) + 0.5
        #     x = length * math.cos(theta)
        #     errors.append(np.linalg.norm(poses[i_sample] - np.array([x, y])))

    if return_samples:
        return errors
    else:
        return np.mean(errors)


class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):  # Norming the input (as in paper) is actually not helpful
        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        # print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores,
        )
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores,
        )

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im)  # Take mean for batch-size stability


class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HardestRankingLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        assert images.shape == captions.shape and len(images.shape) == 2
        images = images / torch.norm(images, dim=1, keepdim=True)
        captions = captions / torch.norm(captions, dim=1, keepdim=True)
        num_samples = len(images)

        similarity_scores = torch.mm(images, captions.transpose(1, 0))  # [I x C]

        cost_images = (
            self.margin + similarity_scores - similarity_scores.diag().view((num_samples, 1))
        )
        cost_images.fill_diagonal_(0)
        cost_images = self.relu(cost_images)
        cost_images, _ = torch.max(cost_images, dim=1)
        cost_images = torch.mean(cost_images)

        cost_captions = (
            self.margin
            + similarity_scores.transpose(1, 0)
            - similarity_scores.diag().view((num_samples, 1))
        )
        cost_captions.fill_diagonal_(0)
        cost_captions = self.relu(cost_captions)
        cost_captions, _ = torch.max(cost_captions, dim=1)
        cost_captions = torch.mean(cost_captions)

        cost = cost_images + cost_captions
        return cost

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        # self.margin = margin
        self.temperature = temperature

    def forward(self, im, s):  # Norming the input (as in paper) is actually not helpful
        # Keep contrastive loss in float32 and use cross-entropy on logits directly
        # instead of exp/log ratios for better numerical stability.
        im = F.normalize(im.float(), p=2, dim=1, eps=1e-6)
        s = F.normalize(s.float(), p=2, dim=1, eps=1e-6)

        if not torch.isfinite(im).all():
            raise RuntimeError("ContrastiveLoss got non-finite image/submap embeddings.")
        if not torch.isfinite(s).all():
            raise RuntimeError("ContrastiveLoss got non-finite text embeddings.")

        logits = torch.mm(im, s.transpose(1, 0).contiguous())
        logits = logits / max(float(self.temperature), 1e-6)

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_i2t + loss_t2i)
# ================= CMMLoc++: MNCL-style multi-negative loss =================
class MNCLContrastiveLoss(nn.Module):
    """
    MNCL-style multi-negative contrastive loss for text–submap alignment.

    Assumes:
      - text_emb:   (B, K) L2-normalized
      - submap_emb: (B, K) L2-normalized

    Uses batch elements as negatives for each other (InfoNCE-style).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_emb: torch.Tensor, submap_emb: torch.Tensor) -> torch.Tensor:
        # Keep the MNCL branch in float32 for numerical stability, especially under AMP.
        text_emb = F.normalize(text_emb.float(), p=2, dim=-1, eps=1e-6)
        submap_emb = F.normalize(submap_emb.float(), p=2, dim=-1, eps=1e-6)

        if not torch.isfinite(text_emb).all():
            raise RuntimeError("MNCLContrastiveLoss got non-finite text embeddings.")
        if not torch.isfinite(submap_emb).all():
            raise RuntimeError("MNCLContrastiveLoss got non-finite submap embeddings.")

        # similarity matrix (B, B): text_emb @ submap_emb^T
        logits = torch.matmul(text_emb, submap_emb.t())
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        batch_size = text_emb.size(0)
        device = text_emb.device
        labels = torch.arange(batch_size, device=device)

        # text → submap
        loss_i2s = F.cross_entropy(logits, labels)

        # submap → text (symmetric)
        loss_s2i = F.cross_entropy(logits.t(), labels)

        return 0.5 * (loss_i2s + loss_s2i)
# =======================================================================


class SceneAwareHardNegativeLoss(nn.Module):
    """
    Contrastive loss that upweights same-scene negatives and the currently
    hardest negatives in-batch. This keeps the stable CE-on-logits formulation,
    but reshapes the softmax with log-weights on selected negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        same_scene_negative_weight: float = 2.0,
        hard_negative_topk: int = 4,
        hard_negative_weight: float = 1.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.same_scene_negative_weight = same_scene_negative_weight
        self.hard_negative_topk = hard_negative_topk
        self.hard_negative_weight = hard_negative_weight

    def _build_adjustment(self, logits: torch.Tensor, scene_names: List[str]) -> torch.Tensor:
        batch_size = logits.size(0)
        device = logits.device
        adjustment = torch.zeros_like(logits)
        eye_mask = torch.eye(batch_size, device=device, dtype=torch.bool)

        if self.same_scene_negative_weight > 1.0:
            same_scene = torch.tensor(
                [[scene_names[i] == scene_names[j] for j in range(batch_size)] for i in range(batch_size)],
                device=device,
                dtype=torch.bool,
            )
            same_scene = same_scene & (~eye_mask)
            adjustment = adjustment + same_scene.float() * math.log(self.same_scene_negative_weight)

        if self.hard_negative_topk > 0 and self.hard_negative_weight > 1.0 and batch_size > 1:
            masked_logits = logits.masked_fill(eye_mask, float("-inf"))
            topk = min(self.hard_negative_topk, batch_size - 1)
            hard_idx = torch.topk(masked_logits, k=topk, dim=1).indices
            hard_mask = torch.zeros_like(logits, dtype=torch.bool)
            hard_mask.scatter_(1, hard_idx, True)
            hard_mask = hard_mask & (~eye_mask)
            adjustment = adjustment + hard_mask.float() * math.log(self.hard_negative_weight)

        return adjustment

    def forward(self, text_emb: torch.Tensor, submap_emb: torch.Tensor, scene_names: List[str]) -> torch.Tensor:
        text_emb = F.normalize(text_emb.float(), p=2, dim=-1, eps=1e-6)
        submap_emb = F.normalize(submap_emb.float(), p=2, dim=-1, eps=1e-6)

        if not torch.isfinite(text_emb).all():
            raise RuntimeError("SceneAwareHardNegativeLoss got non-finite text embeddings.")
        if not torch.isfinite(submap_emb).all():
            raise RuntimeError("SceneAwareHardNegativeLoss got non-finite submap embeddings.")

        logits = torch.matmul(text_emb, submap_emb.t())
        logits = logits / max(float(self.temperature), 1e-6)
        adjustment = self._build_adjustment(logits, scene_names)
        adjusted_logits = logits + adjustment

        labels = torch.arange(adjusted_logits.size(0), device=adjusted_logits.device)
        loss_i2s = F.cross_entropy(adjusted_logits, labels)
        loss_s2i = F.cross_entropy(adjusted_logits.t(), labels)
        return 0.5 * (loss_i2s + loss_s2i)


class SemanticDistillationLoss(nn.Module):
    """
    Light semantic regularizer for coarse training.
    Aligns submap embeddings to a detached semantic teacher built from
    object label/color text, without directly changing the query text branch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
        student_emb = F.normalize(student_emb.float(), p=2, dim=-1, eps=1e-6)
        teacher_emb = F.normalize(teacher_emb.float(), p=2, dim=-1, eps=1e-6)

        if not torch.isfinite(student_emb).all():
            raise RuntimeError("SemanticDistillationLoss got non-finite student embeddings.")
        if not torch.isfinite(teacher_emb).all():
            raise RuntimeError("SemanticDistillationLoss got non-finite teacher embeddings.")

        cosine = torch.sum(student_emb * teacher_emb, dim=-1)
        loss = 1.0 - cosine
        return loss.mean()


class RetrievalDistillationLoss(nn.Module):
    """
    Distill retrieval geometry from a strong teacher coarse model.
    Matches the teacher's text->submap and submap->text similarity
    distributions over the current batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        student_text: torch.Tensor,
        student_submap: torch.Tensor,
        teacher_text: torch.Tensor,
        teacher_submap: torch.Tensor,
    ) -> torch.Tensor:
        student_text = F.normalize(student_text.float(), p=2, dim=-1, eps=1e-6)
        student_submap = F.normalize(student_submap.float(), p=2, dim=-1, eps=1e-6)
        teacher_text = F.normalize(teacher_text.float(), p=2, dim=-1, eps=1e-6)
        teacher_submap = F.normalize(teacher_submap.float(), p=2, dim=-1, eps=1e-6)

        if not torch.isfinite(student_text).all() or not torch.isfinite(student_submap).all():
            raise RuntimeError("RetrievalDistillationLoss got non-finite student embeddings.")
        if not torch.isfinite(teacher_text).all() or not torch.isfinite(teacher_submap).all():
            raise RuntimeError("RetrievalDistillationLoss got non-finite teacher embeddings.")

        tau = max(float(self.temperature), 1e-6)
        s_logits = torch.matmul(student_text, student_submap.t()) / tau
        t_logits = torch.matmul(teacher_text, teacher_submap.t()) / tau

        t_i2s = F.softmax(t_logits, dim=1)
        t_s2i = F.softmax(t_logits.t(), dim=1)
        s_i2s = F.log_softmax(s_logits, dim=1)
        s_s2i = F.log_softmax(s_logits.t(), dim=1)

        loss_i2s = F.kl_div(s_i2s, t_i2s, reduction="batchmean")
        loss_s2i = F.kl_div(s_s2i, t_s2i, reduction="batchmean")
        return 0.5 * (loss_i2s + loss_s2i)

class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0, scale = 64.0):
        super(HardestRankingLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        # assert images.shape == captions.shape and len(images.shape) == 2
        # images = images / torch.norm(images, dim=1, keepdim=True)
        # captions = captions / torch.norm(captions, dim=1, keepdim=True)
        # num_samples = len(images)

        # similarity_scores = torch.mm(images, captions.transpose(1, 0))  # [I x C]

        # cost_images = (
        #     self.margin + similarity_scores - similarity_scores.diag().view((num_samples, 1))
        # )
        # cost_images.fill_diagonal_(0)
        # cost_images = self.relu(cost_images)
        # cost_images, _ = torch.max(cost_images, dim=1)
        # cost_images = torch.mean(cost_images)

        # cost_captions = (
        #     self.margin
        #     + similarity_scores.transpose(1, 0)
        #     - similarity_scores.diag().view((num_samples, 1))
        # )
        # cost_captions.fill_diagonal_(0)
        # cost_captions = self.relu(cost_captions)
        # cost_captions, _ = torch.max(cost_captions, dim=1)
        # cost_captions = torch.mean(cost_captions)

        # cost = cost_images + cost_captions
        # return cost * self.scale
        im = images / torch.norm(images, dim=1, keepdim=True)
        s = captions / torch.norm(captions, dim=1, keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0).contiguous())
        # print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores,
        )
        # print(cost_s)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores,
        )

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0
        
        # import pdb; pdb.set_trace()
        
        cost_captions, _ = torch.max(cost_s, dim=1)
        cost_captions = torch.mean(cost_captions)

        cost_images, _ = torch.max(cost_im, dim=1)
        cost_images = torch.mean(cost_images)

        cost = cost_images + cost_captions
        return cost * self.scale


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/critic.py   
class LinearCritic(nn.Module):

    def __init__(self, temperature=1.):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2):
    # def forward(self, h1, h2):
        # z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        return self.criterion(raw_scores, targets)

if __name__ == "__main__":
    objects = [
        [
            EasyDict(center=np.array([0, 0])),
            EasyDict(center=np.array([10, 10])),
            EasyDict(center=np.array([99, 99])),
        ],
    ]
    matches0 = np.array((0, 1, -1)).reshape((1, 3))
    poses = np.array((0, 10)).reshape((1, 2))
    offsets = np.array([(2, 10), (-10, 0)]).reshape((1, 2, 2))

    err = calc_pose_error(objects, matches0, poses, offsets=None)
    print(err)
    err = calc_pose_error(objects, matches0, poses, offsets=offsets)
    print(err)

class ReConsLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.nb_joints = nb_joints
        self.motion_dim = (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        return loss
    
    def forward_vel(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])
        return loss
