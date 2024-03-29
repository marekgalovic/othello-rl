#!/bin/bash

USE_LAST=0
STREAM_LOGS=0

while getopts ':ls' opt; do
    case $opt in
        l)
            USE_LAST=1
        ;;
        s)
            STREAM_LOGS=1
        ;;
    esac
done

JOB_ID="othello_rl_v3_gs_$(date +%Y%m%d_%H%M%S)"

GCP_PROJECT=$(gcloud config get-value project)
JOBS_DIR="gs://mg_rl_1/othello/models/"

IMAGE_NAME="gcr.io/${GCP_PROJECT}/github.com/marekgalovic/othello-rl"
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
    --epochs 50 \
    --epoch-games 160 \
    --benchmark-games 10 \
    --batch-size 256 \
    --lr 1e-4 \
    --lr-decay 1.0 \
    --reward-gamma 1.0 \
    --checkpoint-gamma 0.2 \
    --agent-net-size 256 \
    --agent-net-conv 5 \
    --mcts-iter 50

if [[ $STREAM_LOGS -eq 1 ]]; then
    gcloud ai-platform jobs stream-logs $JOB_ID
fi
