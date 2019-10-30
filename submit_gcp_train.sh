#!/bin/bash

USE_LAST=0

while getopts ':l' opt; do
    case $opt in
        l)
            USE_LAST=1
        ;;
    esac
done

JOB_ID="othello_rl_ce_train_$(date +%Y%m%d_%H%M%S)"
JOBS_DIR="gs://othello_rl/models/"

IMAGE_NAME="gcr.io/othello-rl/github.com/marekgalovic/othello-rl"
IMAGE_LAST_TAG=$(gcloud container images list-tags $IMAGE_NAME --sort-by "~TIMESTAMP" --limit 1 | tail -n 1 | awk '{ print $2 }')
IMAGE_URI="${IMAGE_NAME}:${IMAGE_LAST_TAG}"

LATEST_COMMIT=$(git rev-parse HEAD)
LATEST_COMMIT_TAG=${LATEST_COMMIT:0:7}

if [[ $USE_LAST -eq 1 ]] && [[ $LATEST_COMMIT_TAG != $IMAGE_LAST_TAG ]]; then
    echo "Latest commit does not match latest image."
    exit 1
fi

echo "JOB_ID: ${JOB_ID}"
echo "IMAGE_URI: ${IMAGE_URI}"

gcloud ai-platform jobs submit training $JOB_ID \
    --config gcp_train_config.yaml \
    --region us-central1 \
    --master-image-uri $IMAGE_URI \
    --job-dir "${JOBS_DIR}${JOB_ID}" \
    -- \
    --epochs 100 \
    --epoch-games 160 \
    --mcts-iter 30 \
    --batch-size 256 \
    --lr 1e-4 \
    --lr-decay 0.98 \
    --reward-gamma 1.0

gcloud ai-platform jobs stream-logs $JOB_ID
