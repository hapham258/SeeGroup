#!/bin/bash
set -e

DATA=/media/hapq/TRAINING_DATA/LayeredFlow/real
CKPT=checkpoints/seegroup.pth
CFG=config/val.py
OUT=predictions/LayeredFlowReal

JOBS=8
export CKPT CFG OUT
find "$DATA/val" "$DATA/test" -name '0_0.png' -print0 |
xargs -0 -P "$JOBS" -I {} bash -c '
    image="$1"
    split=$(echo "$image" | sed "s|.*/real/||" | cut -d/ -f1)
    scene=$(basename "$(dirname "$image")")
    python demo.py \
        --image "$image" \
        --checkpoint-path "$CKPT" \
        --config "$CFG" \
        --output-dir "$OUT/$split/$scene"
' _ {}
