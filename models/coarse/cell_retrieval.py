from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import accumulate
from models.coarse.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, CMMT_SC
from models.coarse.language_encoder import LanguageEncoder
from models.coarse.object_encoder import ObjectEncoder

from easydict import EasyDict as edict


def mask_logits(target, mask):
        return target * mask + (1 - mask) * (-1e10)


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Module for global place recognition.
        Implemented as a text branch (language encoder) and a 3D submap branch (object encoder).
        The 3D submap branch aggregates information about a varying count of multiple objects through Attention.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim

        """
        3D submap branch
        """

        # CARE: possibly handle variation in forward()!

        self.object_encoder = ObjectEncoder(args.coarse_embed_dim, known_classes, known_colors, args)
        self.object_size = args.object_size
        self.object_pos_embed = TrainablePositionalEncoding(max_position_embeddings=2000,
                                                          hidden_size=args.coarse_embed_dim,dropout=args.input_drop)
        self.cell_input_proj = LinearLayer(args.coarse_embed_dim, args.coarse_embed_dim, layer_norm=True,
                                            dropout=args.input_drop, relu=True)    
        self.cell_encoder1 = CMMT_SC(edict(hidden_size=args.coarse_embed_dim, intermediate_size=args.coarse_embed_dim,
                                                 hidden_dropout_prob=args.drop, num_attention_heads=args.n_heads,
                                                 attention_probs_dropout_prob=args.drop,object_size=args.object_size,sft_factor=args.sft_factor))        
        self.weight_token = nn.Parameter(torch.randn(1, 1, args.coarse_embed_dim))

        """
        Textual branch
        """
        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.intra_module_num_layers, 
                                                intra_module_num_heads = args.intra_module_num_heads, 
                                                is_fine = False,  
                                                inter_module_num_layers = args.inter_module_num_layers,
                                                inter_module_num_heads = args.inter_module_num_heads,
                                                text_max_length = args.text_max_length,
                                                ) 
        self.semantic_top_objects = getattr(args, "semantic_top_objects", 12)
        self.use_trainable_reranker = getattr(args, "use_trainable_reranker", False)
        self.reranker_hidden_dim = getattr(args, "reranker_hidden_dim", 32)
        self.known_classes = list(known_classes)
        # COLOR_NAMES contains a duplicate "gray" entry in this codebase.
        # Use a stable unique vocabulary so pairwise symbolic tensors and
        # key->index maps stay aligned.
        self.known_colors = list(dict.fromkeys(known_colors))
        self.label_to_idx = {label: idx for idx, label in enumerate(self.known_classes)}
        self.color_label_to_idx = {
            (color, label): idx
            for idx, (color, label) in enumerate(
                [(color, label) for color in self.known_colors for label in self.known_classes]
            )
        }


        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, dim: {args.coarse_embed_dim}, features: {args.use_features}"
        )
        # ================= CMMLoc++: MNCL-style projection heads =================
        proj_dim = getattr(args, "mncl_proj_dim", 256)

        # Use the same dimension that encode_text / encode_objects return
        in_dim = getattr(self, "embed_dim", proj_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=False),
            nn.Linear(proj_dim, proj_dim),
        )

        self.submap_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=False),
            nn.Linear(proj_dim, proj_dim),
        )

        self.semantic_proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=False),
            nn.Linear(proj_dim, proj_dim),
        )
        if self.use_trainable_reranker:
            self.reranker_head = nn.Sequential(
                nn.Linear(3, self.reranker_hidden_dim),
                nn.ReLU(inplace=False),
                nn.Linear(self.reranker_hidden_dim, 1),
            )
            # Start near identity: the reranker should begin as a tiny correction
            # to the strong base similarity rather than a competing scoring path.
            nn.init.zeros_(self.reranker_head[-1].weight)
            nn.init.zeros_(self.reranker_head[-1].bias)
        else:
            self.reranker_head = None
        # =======================================================================
    
    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer,weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
 
            return encoder_layer(feat, mask, weight_token)

        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)
    
    
    
    def encode_text(self, descriptions):

        description_encodings = self.language_encoder(descriptions)  # [B, DIM]

        description_encodings = F.normalize(description_encodings.float(), p=2, dim=-1, eps=1e-6)
        description_encodings = torch.nan_to_num(description_encodings, nan=0.0, posinf=0.0, neginf=0.0)

        return description_encodings
    # ================= CMMLoc++: projection to contrastive space =================
    def project_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        text_features: (B, D) from encode_text
        returns: (B, K) L2-normalized projected features for MNCL-style contrastive loss
        """
        z = self.text_proj(text_features)
        z = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def project_submaps(self, submap_features: torch.Tensor) -> torch.Tensor:
        """
        submap_features: (B, D) from encode_objects / encode_cells
        returns: (B, K) L2-normalized projected features
        """
        z = self.submap_proj(submap_features)
        z = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def project_semantics(self, semantic_features: torch.Tensor) -> torch.Tensor:
        z = self.semantic_proj(semantic_features)
        z = F.normalize(z.float(), p=2, dim=-1, eps=1e-6)
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    # =======================================================================

    def build_semantic_descriptions(self, objects):
        semantic_descriptions = []
        max_objects = getattr(self.language_encoder, "text_max_length", 128)
        max_objects = min(getattr(self, "semantic_top_objects", 12), max_objects)
        for objects_sample in objects:
            phrases = []
            for obj in objects_sample[:max_objects]:
                phrases.append(f"{obj.get_color_text()} {obj.label}")
            if len(phrases) == 0:
                semantic_descriptions.append("The cell is empty.")
            else:
                # Keep this as a single sentence so the existing language
                # encoder sees a consistent sentence count across the batch.
                semantic_descriptions.append("The cell contains " + ", ".join(phrases) + ".")
        return semantic_descriptions

    def encode_cell_semantics(self, objects):
        semantic_descriptions = self.build_semantic_descriptions(objects)
        semantic_encodings = self.language_encoder(semantic_descriptions)
        semantic_encodings = F.normalize(semantic_encodings.float(), p=2, dim=-1, eps=1e-6)
        semantic_encodings = torch.nan_to_num(semantic_encodings, nan=0.0, posinf=0.0, neginf=0.0)
        return semantic_encodings

    def _build_pose_symbolic_tensors(self, poses):
        device = self.device
        label_counts = torch.zeros((len(poses), len(self.known_classes)), device=device, dtype=torch.float32)
        color_label_counts = torch.zeros(
            (len(poses), len(self.color_label_to_idx)),
            device=device,
            dtype=torch.float32,
        )
        for i_pose, pose in enumerate(poses):
            for descr in pose.descriptions:
                label_idx = self.label_to_idx.get(descr.object_label)
                if label_idx is not None:
                    label_counts[i_pose, label_idx] += 1.0
                color_label_idx = self.color_label_to_idx.get(
                    (descr.object_color_text, descr.object_label)
                )
                if color_label_idx is not None:
                    color_label_counts[i_pose, color_label_idx] += 1.0
        return label_counts, color_label_counts

    def _build_cell_symbolic_tensors(self, objects):
        device = self.device
        label_counts = torch.zeros((len(objects), len(self.known_classes)), device=device, dtype=torch.float32)
        color_label_counts = torch.zeros(
            (len(objects), len(self.color_label_to_idx)),
            device=device,
            dtype=torch.float32,
        )
        for i_cell, objects_sample in enumerate(objects):
            for obj in objects_sample:
                label_idx = self.label_to_idx.get(obj.label)
                if label_idx is not None:
                    label_counts[i_cell, label_idx] += 1.0
                color_label_idx = self.color_label_to_idx.get((obj.get_color_text(), obj.label))
                if color_label_idx is not None:
                    color_label_counts[i_cell, color_label_idx] += 1.0
        return label_counts, color_label_counts

    def compute_pairwise_symbolic_coverages(self, poses, objects):
        pose_labels, pose_color_labels = self._build_pose_symbolic_tensors(poses)
        cell_labels, cell_color_labels = self._build_cell_symbolic_tensors(objects)

        label_match = torch.minimum(
            pose_labels.unsqueeze(1), cell_labels.unsqueeze(0)
        ).sum(dim=-1)
        label_total = pose_labels.sum(dim=-1, keepdim=True).clamp_min(1.0)
        label_cov = label_match / label_total

        color_match = torch.minimum(
            pose_color_labels.unsqueeze(1), cell_color_labels.unsqueeze(0)
        ).sum(dim=-1)
        color_total = pose_color_labels.sum(dim=-1, keepdim=True).clamp_min(1.0)
        color_cov = color_match / color_total

        return label_cov, color_cov

    def combine_rerank_scores(self, base_scores, label_cov, color_cov):
        if self.reranker_head is None:
            return base_scores.float()
        features = torch.stack(
            (base_scores.float(), label_cov.float(), color_cov.float()), dim=-1
        )
        raw_delta = self.reranker_head(features).squeeze(-1)
        delta = 0.5 * torch.tanh(raw_delta)
        return base_scores.float() + delta

    def compute_reranker_logits(self, base_logits, poses, objects):
        label_cov, color_cov = self.compute_pairwise_symbolic_coverages(poses, objects)
        return self.combine_rerank_scores(base_logits, label_cov, color_cov)

    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """
        
        batch = []  # Batch tensor to send into PyG


        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:

                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        embeddings, pos_postions= self.object_encoder(objects, object_points)
        object_size = self.object_size

        index_list = [0]
        last = 0
        
        x = torch.zeros(len(objects), object_size, self.embed_dim).to(self.device)
      

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)
        
        embeddings = F.normalize(embeddings.float(), p=2, dim=-1, eps=1e-6)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            x[idx,: num_object] = embeddings[start : (start + num_object)]
            
        mask = np.ones((len(objects),object_size), np.int_)
        mask = torch.from_numpy(mask)
        mask = mask.to(x.device)

        x = self.encode_input(x,mask,self.cell_input_proj,self.cell_encoder1,self.object_pos_embed,self.weight_token)
        x = torch.where(mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) == 1.0, \
                                                                        x, 0. * x)
        x = x.permute(1, 0, 2).contiguous()
        del embeddings, pos_postions
        
        x = x.max(dim = 0)[0]
        x = F.normalize(x.float(), p=2, dim=-1, eps=1e-6)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

