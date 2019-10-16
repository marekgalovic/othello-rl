#!/bin/bash

JOB_ID="othello_rl_v1_train_$(date +%Y%m%d_%H%M%S)"
JOB_DIR="gs://othello_rl/models/v1"

IMAGE_NAME="gcr.io/othello-rl/github.com/marekgalovic/othello-rl"
IMAGE_LAST_TAG=$(gcloud container images list-tags $IMAGE_NAME --sort-by "~TIMESTAMP" --limit 1 | tail -n 1 | awk '{ print $2 }')
IMAGE_URI="${IMAGE_NAME}:${IMAGE_LAST_TAG}"

echo "JOB_ID: ${JOB_ID}"
echo "IMAGE_URI: ${IMAGE_URI}"

gcloud ai-platform jobs submit training $JOB_ID \
    --config gcp_train_config.yaml \
    --region us-central1 \
    --master-image-uri $IMAGE_URI \
    --job-dir $JOB_DIR \
    -- \
    --epochs 100 \
    --epoch-games 100 \
    --mcts-iter 50 \
    --batch-size 256 \
    --lr 1e-4 \
    --lr-decay 0.96

gcloud ai-platform jobs stream-logs $JOB_ID
