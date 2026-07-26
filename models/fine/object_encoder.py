from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from models.fine.language_encoder import get_mlp
from models.pointcloud.pointnet2 import PointNet2

from datapreparation.kitti360pose.imports import Object3d
from datapreparation.kitti360pose.utils import COLOR_NAMES


def safe_l2_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(tensor, p=2, dim=dim, eps=1e-6)

class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, known_classes: List[str], known_colors: List[str], args):
        """Module to encode a set of instances (objects / stuff)

        Args:
            embed_dim (int): Embedding dimension
            known_classes (List[str]): List of known classes
            known_colors (List[str]): List of known colors
            args: Global training arguments
        """
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.known_colors = {c: i for i, c in enumerate(COLOR_NAMES)}
        self.known_colors["<unk>"] = 0
        self.color_embedding = nn.Embedding(len(self.known_colors), embed_dim, padding_idx=0)

        self.pos_encoder = get_mlp([3, 64, embed_dim])  # OPTION: pos_encoder layers
        self.color_encoder = get_mlp([3, 64, embed_dim])  # OPTION: color_encoder layers
        if args.prealign_color_path:
            dict_color = torch.load(args.prealign_color_path, map_location="cpu")
            load_dict_color = {}
            for k, v in dict_color.items():
                if "batches" in k:
                    continue
                key = k.split("color_encoder.")[1]
                load_dict_color[key] = v
            self.color_encoder.load_state_dict(load_dict_color)
        self.num_encoder = get_mlp([1, 64, embed_dim])

        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817
        

        self.pointnet = PointNet2(
            len(known_classes), len(known_colors), args
        )  # The known classes are all the same now, at least for K360
        if args.prealign_pointnet_path:
            dict_pointnet = torch.load(args.prealign_pointnet_path, map_location="cpu")
            load_dict_pointnet = {}
            for k, v in dict_pointnet.items():
                if "batches" in k or "mlp_pointnet" in k:
                    continue
                key = k.split("pointnet.")[1]
                load_dict_pointnet[key] = v
            self.pointnet.load_state_dict(load_dict_pointnet)
 

        if args.pointnet_freeze:
            print("CARE: freezing PN")
            self.pointnet.requires_grad_(False)
        

        if args.pointnet_features == 0:
            self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
        elif args.pointnet_features == 1:
            self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])
        elif args.pointnet_features == 2:
            self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])
        self.mlp_merge = get_mlp([len(args.use_features) * embed_dim, embed_dim])

    def forward(self, objects: List[Object3d], object_points):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        if ("class_embed" in self.args and self.args.class_embed) or (
            "color_embed" in self.args and self.args.color_embed
        ):
            class_indices = []
            color_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
                    color_idx = self.known_colors[obj.get_color_text()]
                    color_indices.append(color_idx)

        if "class_embed" not in self.args or self.args.class_embed == False:
            # Void all colors for ablation
            if "color" not in self.args.use_features:
                for pyg_batch in object_points:
                    pyg_batch.x[:] = 0.0  # x is color, pos is xyz

            object_features = [
                self.pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in object_points
            ]  # [B, obj_counts, PN_dim]


            object_features = torch.cat(object_features, dim=0)  # [total_objects, PN_dim]
            object_features = self.mlp_pointnet(object_features)

        embeddings = []
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(
                    torch.tensor(class_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(safe_l2_normalize(class_embedding))
            else:
                embeddings.append(safe_l2_normalize(object_features))  # Use features from PointNet

        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                color_embedding = self.color_embedding(
                    torch.tensor(color_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(safe_l2_normalize(color_embedding))
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb() for obj in objects_sample])
                colors = np.asarray(colors, dtype=np.float32)
                color_embedding = self.color_encoder(
                    torch.tensor(colors, dtype=torch.float, device=self.get_device())
                )
                color_embedding = torch.nan_to_num(color_embedding, nan=0.0, posinf=0.0, neginf=0.0)
                embeddings.append(safe_l2_normalize(color_embedding))

        if "position" in self.args.use_features:
            positions = []
            relative_pos_dist = []
            relative_pos_x = []
            relative_pos_y = []
            for objects_sample in objects:                
                positions.extend([obj.get_center() for obj in objects_sample])
                object_num = len(objects_sample)
                relation_matrix = np.zeros([object_num, object_num]).astype(np.float32)
                for m in range(object_num):
                    pose = objects_sample[m].get_center()[0:2]
                    for n in range(object_num):
                        pose_for_dist = objects_sample[n].get_center()[0:2]
                        dist = np.linalg.norm(pose-pose_for_dist)
                        relation_matrix[m][n] = dist
                relative_pos_dist.append(safe_l2_normalize(torch.from_numpy(relation_matrix), dim=1))
            relative_position_embeds = torch.stack(relative_pos_dist,dim=0)


            positions = np.asarray(positions, dtype=np.float32)
            pos_positions = torch.tensor(positions, dtype=torch.float, device=self.get_device())
            pos_embedding = self.pos_encoder(pos_positions)
            pos_embedding = torch.nan_to_num(pos_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(safe_l2_normalize(pos_embedding))

        if "num" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                num_points.extend([len(obj.xyz) for obj in objects_sample])
            num_points = np.asarray(num_points, dtype=np.float32)
            normed_num_points = (
                torch.tensor(num_points, dtype=torch.float, device=self.get_device()).unsqueeze(-1)
                - self.num_mean
            ) / self.num_std
            normed_num_points = torch.clamp(normed_num_points, min=-10.0, max=10.0)
            num_points_embedding = self.num_encoder(
                normed_num_points
            )
            num_points_embedding = torch.nan_to_num(num_points_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            embeddings.append(safe_l2_normalize(num_points_embedding))


        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

        return embeddings,pos_embedding, relative_position_embeds

    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device
