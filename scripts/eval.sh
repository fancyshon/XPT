#/!bin/bash

python eval_interpretability.py \
    --gpuid=0 \
    --base_architecture=deit_tiny_patch16_224 \
    --reserve_layers=11 \
    --data_path=datasets/CUB_200_2011/ \
    --prototype_shape 2000 192 1 1 \
    --use_global=True \
    --global_proto_per_class=10 \
    --resume=output_cosine/CUB2011U/add_orthogonal_loss/1028--adamw-0.05-200-protopformer/checkpoints/epoch-best.pth \
    --global_coe=0.5 \
    --reserve_token_nums=81 \
    --use_ppc_loss=True \
    --batch_size=32 \
    --out_dir=output_view