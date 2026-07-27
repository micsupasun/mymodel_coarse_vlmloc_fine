"""Audited CMMLoc public-release coarse backend.

The released ``coarse.pth`` contains three module families that are absent
from the public constructor.  The official evaluation script consequently
ignores those tensors with ``strict=False``.  This module reproduces that
*published inference behavior* in a separate backend: the public constructor
is retained, the checkpoint-only families are explicitly enumerated by the
caller, and every missing/unexpected key is audited before loading.

This is intentionally not the architecture used by ``my_model``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from models.coarse.language_encoder import LanguageEncoder
from models.coarse.model_components import (
    CMMT_SC,
    LinearLayer,
    TrainablePositionalEncoding,
)
from models.coarse.object_encoder import ObjectEncoder


CMMLOC_RELEASE_IGNORED_PREFIXES = (
    "cell_encoder2.",
    "modular_vector_mapping.",
    "obj_inter_module.",
)
CMMLOC_RELEASE_IGNORED_PREFIX_COUNTS = {
    "cell_encoder2": 130,
    "modular_vector_mapping": 1,
    "obj_inter_module": 24,
}
CMMLOC_RELEASE_IGNORED_KEY_COUNT = sum(
    CMMLOC_RELEASE_IGNORED_PREFIX_COUNTS.values()
)


class CMMLocReleaseLanguageEncoder(LanguageEncoder):
    """Public CMMLoc text forward with only model-path compatibility added."""

    def forward(self, descriptions):
        # This is the pinned public forward path. In particular, it does not
        # silently truncate text. The inherited constructor only replaces the
        # literal public placeholder PATH_TO_T5 with the requested T5 path.
        from nltk import tokenize as text_tokenize

        split_union_sentences = []
        for description in descriptions:
            split_union_sentences.extend(
                text_tokenize.sent_tokenize(description)
            )

        batch_size = len(descriptions)
        if batch_size == 0:
            raise ValueError("CMMLoc release text batch is empty")
        if len(split_union_sentences) % batch_size:
            raise RuntimeError(
                "CMMLoc release requires the same sentence count per query"
            )
        num_sentence = len(split_union_sentences) // batch_size

        inputs = self.tokenizer(
            split_union_sentences,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        out = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
        )
        description_encodings = out.last_hidden_state
        if self.fixed_embedding:
            description_encodings = description_encodings.detach()

        description_encodings = description_encodings.permute(1, 0, 2)
        for layer in self.intra_module:
            description_encodings = layer(description_encodings)
        description_encodings = (
            description_encodings.permute(1, 0, 2)
            .contiguous()
            .max(dim=1)[0]
        )
        description_encodings = self.inter_mlp(description_encodings)
        description_encodings = description_encodings.view(
            batch_size, num_sentence, -1
        )
        if self.is_fine:
            return description_encodings

        description_encodings = description_encodings.permute(1, 0, 2)
        for layer in self.inter_module:
            description_encodings += layer(description_encodings)
        return description_encodings.max(dim=0)[0]


class CMMLocReleaseObjectEncoder(ObjectEncoder):
    """Pinned public object forward with outer-checkpoint PointNet loading."""

    def forward(self, objects, object_points):
        if (
            ("class_embed" in self.args and self.args.class_embed)
            or ("color_embed" in self.args and self.args.color_embed)
        ):
            class_indices = []
            color_indices = []
            for objects_sample in objects:
                for obj in objects_sample:
                    class_indices.append(
                        self.known_classes.get(obj.label, 0)
                    )
                    color_indices.append(
                        self.known_colors[obj.get_color_text()]
                    )

        if "class_embed" not in self.args or not self.args.class_embed:
            if "color" not in self.args.use_features:
                for pyg_batch in object_points:
                    pyg_batch.x[:] = 0.0
            object_features = [
                self.pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in object_points
            ]
            object_features = torch.cat(object_features, dim=0)
            object_features = self.mlp_pointnet(object_features)

        embeddings = []
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args
                and self.args.class_embed
            ):
                class_embedding = self.class_embedding(
                    torch.tensor(
                        class_indices,
                        dtype=torch.long,
                        device=self.get_device(),
                    )
                )
                embeddings.append(F.normalize(class_embedding, dim=-1))
            else:
                embeddings.append(F.normalize(object_features, dim=-1))

        if "color" in self.args.use_features:
            if (
                "color_embed" in self.args
                and self.args.color_embed
            ):
                color_embedding = self.color_embedding(
                    torch.tensor(
                        color_indices,
                        dtype=torch.long,
                        device=self.get_device(),
                    )
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend(
                        obj.get_color_rgb() for obj in objects_sample
                    )
                color_embedding = self.color_encoder(
                    torch.tensor(
                        colors,
                        dtype=torch.float,
                        device=self.get_device(),
                    )
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))

        if "position" in self.args.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend(
                    obj.get_center() for obj in objects_sample
                )
            pos_positions = torch.tensor(
                positions, dtype=torch.float, device=self.get_device()
            )
            embeddings.append(
                F.normalize(self.pos_encoder(pos_positions), dim=-1)
            )

        if "num" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                num_points.extend(len(obj.xyz) for obj in objects_sample)
            values = torch.tensor(
                num_points, dtype=torch.float, device=self.get_device()
            ).unsqueeze(-1)
            num_embedding = self.num_encoder(
                (values - self.num_mean) / self.num_std
            )
            embeddings.append(F.normalize(num_embedding, dim=-1))

        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]
        return embeddings, pos_positions


class CMMLocReleaseCoarseNetwork(torch.nn.Module):
    """The exact constructor and forward graph exposed by CMMLoc's release."""

    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        super().__init__()
        self.embed_dim = args.coarse_embed_dim
        self.object_encoder = CMMLocReleaseObjectEncoder(
            args.coarse_embed_dim, known_classes, known_colors, args
        )
        self.object_size = args.object_size
        self.object_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=2000,
            hidden_size=args.coarse_embed_dim,
            dropout=args.input_drop,
        )
        self.cell_input_proj = LinearLayer(
            args.coarse_embed_dim,
            args.coarse_embed_dim,
            layer_norm=True,
            dropout=args.input_drop,
            relu=True,
        )
        self.cell_encoder1 = CMMT_SC(
            edict(
                hidden_size=args.coarse_embed_dim,
                intermediate_size=args.coarse_embed_dim,
                hidden_dropout_prob=args.drop,
                num_attention_heads=args.n_heads,
                attention_probs_dropout_prob=args.drop,
                object_size=args.object_size,
                sft_factor=args.sft_factor,
            )
        )
        self.weight_token = torch.nn.Parameter(
            torch.randn(1, 1, args.coarse_embed_dim)
        )
        self.language_encoder = CMMLocReleaseLanguageEncoder(
            args.coarse_embed_dim,
            hungging_model=args.hungging_model,
            fixed_embedding=args.fixed_embedding,
            intra_module_num_layers=args.intra_module_num_layers,
            intra_module_num_heads=args.intra_module_num_heads,
            is_fine=False,
            inter_module_num_layers=args.inter_module_num_layers,
            inter_module_num_heads=args.inter_module_num_heads,
            text_max_length=args.text_max_length,
        )
        print(
            "CMMLocReleaseCoarseNetwork, "
            f"class embed {args.class_embed}, "
            f"color embed {args.color_embed}, "
            f"dim: {args.coarse_embed_dim}, "
            f"features: {args.use_features}"
        )

    @staticmethod
    def encode_input(
        feat,
        mask,
        input_proj_layer,
        encoder_layer,
        pos_embed_layer,
        weight_token=None,
    ):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)
        if weight_token is not None:
            return encoder_layer(feat, mask, weight_token)
        return encoder_layer(feat, mask)

    def encode_text(self, descriptions):
        return F.normalize(self.language_encoder(descriptions))

    def encode_objects(self, objects, object_points):
        embeddings, pos_positions = self.object_encoder(
            objects, object_points
        )
        index_list = [0]
        last = 0
        for sample_objects in objects:
            last += len(sample_objects)
            index_list.append(last)

        x = torch.zeros(
            len(objects),
            self.object_size,
            self.embed_dim,
            device=self.device,
        )
        embeddings = F.normalize(embeddings, dim=-1)
        for index in range(len(index_list) - 1):
            start = index_list[index]
            count = min(
                index_list[index + 1] - start, self.object_size
            )
            x[index, :count] = embeddings[start : start + count]

        # The all-one mask is part of the pinned public implementation.
        mask = torch.from_numpy(
            np.ones((len(objects), self.object_size), np.int_)
        ).to(x.device)
        x = self.encode_input(
            x,
            mask,
            self.cell_input_proj,
            self.cell_encoder1,
            self.object_pos_embed,
            self.weight_token,
        )
        x = torch.where(
            mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) == 1.0,
            x,
            0.0 * x,
        )
        x = x.permute(1, 0, 2).contiguous()
        del embeddings, pos_positions
        return F.normalize(x.max(dim=0)[0])

    def forward(self):
        raise RuntimeError("CMMLoc release coarse forward is not implemented")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device
