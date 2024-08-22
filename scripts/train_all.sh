#!/bin/bash

sh scripts/fsl_cub.sh deit_tiny_patch16_224 256 5 5
sh scripts/fsl_miniImagenet.sh deit_tiny_patch16_224 256 5 5
sh scripts/fsl_tieredImagenet.sh deit_tiny_patch16_224 256 5 5

sh scripts/fsl_cub.sh deit_tiny_patch16_224 256 5 1
sh scripts/fsl_miniImagenet.sh deit_tiny_patch16_224 256 5 1
sh scripts/fsl_tieredImagenet.sh deit_tiny_patch16_224 256 5 1

# sh scripts/fsl_cifarfs.sh deit_tiny_patch16_224 256 5 5
# sh scripts/fsl_cifarfs.sh deit_tiny_patch16_224 256 5 1

# sh scripts/fsl_fc100.sh deit_tiny_patch16_224 256 5 5
# sh scripts/fsl_fc100.sh deit_tiny_patch16_224 256 5 1