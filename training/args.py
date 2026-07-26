import argparse
from argparse import ArgumentParser
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Training")

    # General
    parser.add_argument("--train", default=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="K360", help="Currently only K360")
    parser.add_argument("--base_path", type=str, help="Root path of Kitti360Pose")

    # Model
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")

    parser.add_argument(
        "--continue_path", type=str, help="Set to continue from a previous checkpoint"
    )

    parser.add_argument("--no_pc_augment", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--offset_lambda", type=float, default=5)
    parser.add_argument("--pmc_prob", type=float, default=0.0)
    parser.add_argument("--pmc_threshold", type=float, default=0.4)

    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)
    parser.add_argument("--block_head", type=int, default=8)
    parser.add_argument("--fuse_layer", type=int, default=1)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    # Loss
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--temperatures", type=float, default=1.0) 
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--ranking_loss", type=str, default="contrastive")

    # ================= CMMLoc++ (MNCL) specific args =================
    parser.add_argument(
        "--lambda_mncl",
        type=float,
        default=0.3,
        help="Weight for MNCL-style contrastive loss in CMMLoc++ (0 disables it).",
    )

    parser.add_argument(
        "--mncl_proj_dim",
        type=int,
        default=256,
        help="Projection dimension for MNCL-style contrastive head.",
    )
    parser.add_argument(
        "--lambda_semantic_mncl",
        type=float,
        default=0.0,
        help="Weight for semantic-composition contrastive loss in coarse training.",
    )
    parser.add_argument(
        "--lambda_scene_hardneg",
        type=float,
        default=0.0,
        help="Weight for scene-aware hard negative contrastive loss in coarse training.",
    )
    parser.add_argument(
        "--same_scene_negative_weight",
        type=float,
        default=2.0,
        help="Extra multiplier for negatives from the same scene in the hard-negative loss.",
    )
    parser.add_argument(
        "--hard_negative_topk",
        type=int,
        default=4,
        help="Number of strongest off-diagonal negatives to upweight per row/column.",
    )
    parser.add_argument(
        "--hard_negative_weight",
        type=float,
        default=1.5,
        help="Extra multiplier for hardest negatives in the hard-negative loss.",
    )
    parser.add_argument(
        "--semantic_top_objects",
        type=int,
        default=12,
        help="Maximum number of object label-color phrases used to build semantic cell descriptions.",
    )
    parser.add_argument(
        "--lambda_semantic_distill",
        type=float,
        default=0.0,
        help="Weight for detached semantic teacher regularization on submap embeddings.",
    )
    parser.add_argument(
        "--teacher_coarse_path",
        type=str,
        default="",
        help="Optional teacher coarse checkpoint for retrieval distillation.",
    )
    parser.add_argument(
        "--lambda_teacher_distill",
        type=float,
        default=0.0,
        help="Weight for coarse teacher distillation loss.",
    )
    parser.add_argument(
        "--teacher_temperature",
        type=float,
        default=0.07,
        help="Temperature for teacher/student retrieval distillation.",
    )
    parser.add_argument(
        "--use_trainable_reranker",
        action="store_true",
        help="Train a lightweight symbolic reranker head on top of coarse similarities.",
    )
    parser.add_argument(
        "--lambda_reranker",
        type=float,
        default=0.0,
        help="Weight for the trainable reranker cross-entropy objective.",
    )
    parser.add_argument(
        "--reranker_hidden_dim",
        type=int,
        default=32,
        help="Hidden size of the trainable coarse reranker MLP.",
    )
    parser.add_argument(
        "--trainable_rerank_topn",
        type=int,
        default=50,
        help="Top-N coarse candidates to rerank with the trainable reranker during evaluation.",
    )
    parser.add_argument(
        "--skip_train_eval",
        action="store_true",
        help="Skip full training-set retrieval evaluation after each epoch. This speeds up training and does not affect optimization or val/test model selection.",
    )
    parser.add_argument(
        "--train_eval_frequency",
        type=int,
        default=1,
        help="Run full training-set retrieval evaluation every N epochs. Set 0 to skip it. --skip_train_eval also skips it. Validation/test evaluation and checkpoint selection are unchanged.",
    )
    # ======================================================================

    # Object-encoder / PointNet
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument(
        # "--pointnet_path", type=str, default="E:\\Github storage\\CMMLocPP\\checkpoints\\pointnet_acc0.86_lr1_p256.pth"
        "--pointnet_path", type=str,default="/share/nas/cs-nas/zh932237/CMMLoc_MNCLv4/checkpoints/pointnet_acc0.86_lr1_p256.pth"
    )
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)
    parser.add_argument("--input_drop", default=0.2)
    parser.add_argument("--drop", default=0.2)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_position_embed", type=int, default=60)
    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--num_objects", type=int, default=16)
    parser.add_argument("--sft_factor", default=0.6)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)
    parser.add_argument("--projection_dim", default=256)
    # Language Encoder
    parser.add_argument("--hungging_model", type=str, help="hugging face model")
    parser.add_argument("--fixed_embedding", action="store_true")
    parser.add_argument(
        "--text_max_length",
        type=int,
        default=128,
        help="Maximum tokenizer length for all text encoders.",
    )
    parser.add_argument(
        "--prealign_pointnet_path",
        type=str,
        default="",
        help="Optional fine-stage PointNet checkpoint from pre-align.",
    )
    parser.add_argument(
        "--prealign_color_path",
        type=str,
        default="",
        help="Optional fine-stage color encoder checkpoint from pre-align.",
    )
    parser.add_argument(
        "--prealign_mlp_path",
        type=str,
        default="",
        help="Optional fine-stage language MLP checkpoint from pre-align.",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable automatic mixed precision on CUDA.",
    )

    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)

    # Variations which tranlations are fed into the network for training/evaluation.
    # NOTE: These variations did not make much difference.
    parser.add_argument("--regressor_cell", type=str, default="pose")  # Pose or best
    parser.add_argument("--regressor_learn", type=str, default="center")  # Center or closest
    parser.add_argument("--regressor_eval", type=str, default="center")  # Center or closest

    # Others
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="exponential")
    parser.add_argument("--lr_step", type=float, default=10)
    parser.add_argument("--folder_name", type=str, default="folder_name")

    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")


    args = parser.parse_args()

    if bool(args.continue_path):
        assert osp.isfile(args.continue_path)

    assert args.regressor_cell in ("pose", "best", "all")
    assert args.regressor_learn in ("center", "closest")
    assert args.regressor_eval in ("center", "closest")

    args.dataset = args.dataset.upper()
    assert args.dataset in ("S3D", "K360")

    assert args.ranking_loss in ("triplet", "pairwise", "hardest", "contrastive")

    for feat in args.use_features:
        assert feat in ["class", "color", "position", "num"], "Unexpected feature"

    if args.pointnet_path:
        assert osp.isfile(args.pointnet_path)

    if args.prealign_pointnet_path:
        assert osp.isfile(args.prealign_pointnet_path)
    if args.prealign_color_path:
        assert osp.isfile(args.prealign_color_path)
    if args.prealign_mlp_path:
        assert osp.isfile(args.prealign_mlp_path)
    if args.teacher_coarse_path:
        assert osp.isfile(args.teacher_coarse_path)

    assert osp.isdir(args.base_path)

    assert args.describe_by in ("closest", "class", "direction", "random", "all")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
