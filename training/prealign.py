import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from typing import List
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
from training.args import parse_arguments
from training.checkpointing import build_training_state, save_training_state
from contextlib import nullcontext
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
# from models.language_encoder import get_mlp, LanguageEncoder
# from models.object_encoder import ObjectEncoder
from models.fine.language_encoder import get_mlp
from models.pointcloud.pointnet2 import PointNet2

from datapreparation.kitti360pose.imports import Object3d
from datapreparation.kitti360pose.utils import COLOR_NAMES
from nltk import tokenize as text_tokenize
# if nltk not work well add the following command
# nltk.download('punkt')
from transformers import AutoTokenizer, T5EncoderModel


def safe_l2_normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return F.normalize(tensor, p=2, dim=dim, eps=1e-6)


def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models without RELU in the final layer.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )

class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,  hungging_model = None, fixed_embedding=False, 
                 intra_module_num_layers=2, intra_module_num_heads=4, 
                 is_fine = False, inter_module_num_layers=2, inter_module_num_heads=4,
                 text_max_length=128,
                 ):
        """Language encoder to encode a set of hints for each sentence"""
        super(LanguageEncoder, self).__init__()

        self.is_fine = is_fine
        self.model_name = hungging_model or "t5-large"
        self.text_max_length = text_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.model_max_length = text_max_length
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.llm_model = T5EncoderModel.from_pretrained(self.model_name)
        if fixed_embedding:
            self.fixed_embedding = True
            for para in self.llm_model.parameters():
                para.requires_grad = False
        else:
            self.fixed_embedding = False

        input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]

        self.intra_module = nn.ModuleList([nn.TransformerEncoderLayer(input_dim, intra_module_num_heads,  dim_feedforward = input_dim * 4) for _ in range(intra_module_num_layers)])

        self.inter_mlp = get_mlp2([input_dim, embedding_dim], add_batchnorm=True)
        
        # if not is_fine:
        #     self.inter_module = nn.ModuleList([nn.TransformerEncoderLayer(embedding_dim, inter_module_num_heads,  dim_feedforward = embedding_dim * 4) for _ in range(inter_module_num_layers)])
            
    
    def forward(self, descriptions):

        split_union_sentences = []
        for description in descriptions:
            split_union_sentences.extend(text_tokenize.sent_tokenize(description))

        
        batch_size = len(descriptions)
        num_sentence = len(split_union_sentences) // batch_size

        inputs = self.tokenizer(
            split_union_sentences,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.text_max_length,
        )
        shorten_sentences_indices = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        shorten_sentences_indices = shorten_sentences_indices.to(self.device)
        attention_mask = attention_mask.to(self.device)
        llm_context = torch.no_grad if self.fixed_embedding else nullcontext
        with llm_context():
            with torch.cuda.amp.autocast(enabled=False):
                out = self.llm_model(input_ids = shorten_sentences_indices, 
                                attention_mask = attention_mask,
                                output_attentions = False)
                description_encodings = out.last_hidden_state.float()
        
        if self.fixed_embedding:
            description_encodings = description_encodings.detach()
        description_encodings = torch.nan_to_num(description_encodings, nan=0.0, posinf=0.0, neginf=0.0)
        description_encodings = description_encodings.max(dim = 1)[0]

        description_encodings = self.inter_mlp(description_encodings)
        description_encodings = torch.nan_to_num(description_encodings, nan=0.0, posinf=0.0, neginf=0.0)
        return description_encodings

    @property
    def device(self):
        return next(self.inter_mlp.parameters()).device
    

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
        self.num_encoder = get_mlp([1, 64, embed_dim])

        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817
        

        self.pointnet = PointNet2(
            len(known_classes), len(known_colors), args
        )  # The known classes are all the same now, at least for K360
        self.pointnet_dim = self.pointnet.lin2.weight.size(0)

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
            for objects_sample in objects:
                positions.extend([obj.get_center() for obj in objects_sample])
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
            normed_num_points = (torch.tensor(num_points, dtype=torch.float, device=self.get_device()).unsqueeze(-1) - self.num_mean) / self.num_std
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

        return (
            embeddings,
            pos_positions,
            safe_l2_normalize(object_features),
            safe_l2_normalize(color_embedding),
        )

    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device
    
class pretrain_object(nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Fine localization module.
        Consists of text branch (language encoder) and a 3D submap branch (object encoder) and
        cascaded cross-attention transformer (CCAT) module.

        Args:
            known_classes (List[str]): List of known classes
            known_colors (List[str]): List of known colors
            args: Global training args
        """
        super(pretrain_object, self).__init__()
        self.embed_dim = args.fine_embed_dim
        self.object_encoder = ObjectEncoder(args.fine_embed_dim, known_classes, known_colors, args)
        self.language_encoder = LanguageEncoder(args.fine_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.fine_intra_module_num_layers, 
                                                intra_module_num_heads = args.fine_intra_module_num_heads, 
                                                is_fine = True,  
                                                text_max_length = args.text_max_length,
                                                ) 
    
    def forward(self,objects,object_points):

        embeddings, pos_positions,object_feature,color_feature = self.object_encoder(objects,object_points)
        label_embed = []
        color_embed = []
        for i in range(len(objects)):
            color_list = []
            label_list = []
            object_list  =  objects[i]
            for n in range(len(object_list)):
                color = object_list[n].get_color_text()
                label = object_list[n].label
                color_list.append(f"{color}")
                label_list.append(f"{label}")
            label_feat = self.language_encoder(label_list)
            color_feat = self.language_encoder(color_list)
            label_embed.append(label_feat)
            color_embed.append(color_feat)
        t5_label_embed = torch.cat(label_embed,dim=0)
        t5_color_embed = torch.cat(color_embed,dim=0)
        return object_feature,color_feature,t5_label_embed,t5_color_embed
    
def train_epoch(model,dataloader,args):
    model.train()
    stats = EasyDict(
        loss=[],
    )
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            object_feature,color_feature,t5_label_embed,t5_color_embed = model(batch["objects"],batch["object_points"])
            if not torch.isfinite(object_feature).all() or not torch.isfinite(color_feature).all():
                raise RuntimeError("Non-finite pre-align object/color features detected.")
            if not torch.isfinite(t5_label_embed).all() or not torch.isfinite(t5_color_embed).all():
                raise RuntimeError("Non-finite pre-align text features detected.")
            loss = loss_label(object_feature,t5_label_embed) + loss_color(color_feature,t5_color_embed)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pre-align loss detected before backward.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        stats.loss.append(loss.item())
        with torch.no_grad():
            loss_color_value = loss_color(color_feature, t5_color_embed).item()
            loss_label_value = loss_label(object_feature, t5_label_embed).item()
        pbar.set_postfix(loss_color = loss_color_value,loss_label = loss_label_value,loss = loss.item())
    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


@torch.no_grad()
def eval_epoch(model, dataloader, args):
    model.eval()
    stats = EasyDict(loss=[])
    pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    for i_batch, batch in pbar:
        with torch.cuda.amp.autocast(enabled=use_amp):
            object_feature, color_feature, t5_label_embed, t5_color_embed = model(
                batch["objects"], batch["object_points"]
            )
            if not torch.isfinite(object_feature).all() or not torch.isfinite(color_feature).all():
                raise RuntimeError("Non-finite pre-align object/color features detected during evaluation.")
            if not torch.isfinite(t5_label_embed).all() or not torch.isfinite(t5_color_embed).all():
                raise RuntimeError("Non-finite pre-align text features detected during evaluation.")
            loss = loss_label(object_feature, t5_label_embed) + loss_color(color_feature, t5_color_embed)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pre-align validation loss detected.")
        stats.loss.append(loss.item())
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
    dataset_path = args.base_path.rstrip("/\\")       # remove trailing / or \
    dataset_name = osp.basename(dataset_path)         # e.g. "k360_30-10_scG_pd10_pc4_spY_all"
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
    torch.autograd.set_detect_anomaly(True)
    use_amp = bool(args.use_amp and device.type == "cuda")
    model = pretrain_object(dataset_train.get_known_classes(),COLOR_NAMES_K360,args)
    model.to(device)
    start_epoch = 1
    if args.continue_path:
        print("Loading pre-align checkpoint from", args.continue_path)
        loaded_payload = torch.load(args.continue_path, map_location=device)
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            model.load_state_dict(loaded_payload["model_state"], strict=False)
        else:
            model.load_state_dict(loaded_payload, strict=False)
    loss_label = nn.MSELoss()
    loss_color = nn.MSELoss()
    # model_dic = model.state_dict()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    lr = args.learning_rate
    num_epoch_warmup = 3
    best_val_loss = float("inf")
    if args.continue_path:
        loaded_payload = torch.load(args.continue_path, map_location=device)
        if isinstance(loaded_payload, dict) and "model_state" in loaded_payload:
            start_epoch = int(loaded_payload.get("epoch", 0)) + 1
            best_val_loss = loaded_payload.get("best_metric", best_val_loss)
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
            print(f"Resuming pre-align training from epoch {start_epoch}.")

    resume_state_path = osp.join(checkpoint_dir, "resume_training_state.pth")
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
        val_out = eval_epoch(model, dataloader_val, args,)
        if scheduler:
            scheduler.step()

        print(
            (
                f"\t lr {lr:0.6} epoch {epoch} train-loss {train_out.loss:0.3f} "
                f"val-loss {val_out.loss:0.3f} "
            ),
            flush=True,
        )
        val_loss = np.mean(val_out.loss)

        full_model_path = f"./checkpoints/{dataset_name}/{folder_name}/resume_full_model.pth"
        torch.save(model.state_dict(), full_model_path)

        if val_loss < best_val_loss:
            pointnet_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_pointnet.pth"
            color_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_color_encoder.pth"
            mlp_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_mlp.pth"
            best_pointnet_path = f"./checkpoints/{dataset_name}/{folder_name}/best_pointnet.pth"
            best_color_path = f"./checkpoints/{dataset_name}/{folder_name}/best_color_encoder.pth"
            best_mlp_path = f"./checkpoints/{dataset_name}/{folder_name}/best_mlp.pth"
            
            model_dic = model.state_dict()
                # out = collections.OrderedDict()
                # for item in model_dic:
                #     if "llm_model" not in item:
                #         out[item] = model_dic[item]
            pointnet_dict = {k:v for k,v in model_dic.items() if "pointnet" in k}
            color_dict = {k:v for k,v in model_dic.items() if "color_encoder" in k}
            mlp_dict = {k:v for k,v in model_dic.items() if "inter_mlp" in k}
            torch.save(pointnet_dict, pointnet_path)
            torch.save(color_dict, color_path)
            torch.save(mlp_dict, mlp_path)
            torch.save(pointnet_dict, best_pointnet_path)
            torch.save(color_dict, best_color_path)
            torch.save(mlp_dict, best_mlp_path)
                
            best_val_loss = val_loss

        current_phase = "main" if epoch >= num_epoch_warmup else "warmup"
        resume_payload = build_training_state(
            model_state=model.state_dict(),
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_val_loss,
            extra_state={"phase": current_phase},
        )
        save_training_state(resume_state_path, resume_payload)

