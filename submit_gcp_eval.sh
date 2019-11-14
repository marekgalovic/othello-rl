#!/bin/bash

JOB_ID=""

while getopts 'j:' opt; do
    case $opt in
        j)
            JOB_ID="${OPTARG}"
        ;;
    esac
done
if [ -z "${JOB_ID}" ]; then
    echo "No JOB ID provided"
    exit 1
fi

GCP_PROJECT=$(gcloud config get-value project)
JOBS_DIR="gs://mg_rl_1/othello/models/"

IMAGE_NAME="gcr.io/${GCP_PROJECT}/github.com/marekgalovic/othello-rl"
IMAGE_LAST_TAG=$(gcloud container images list-tags $IMAGE_NAME --sort-by "~TIMESTAMP" --limit 1 | tail -n 1 | awk '{ print $2 }')
IMAGE_URI="${IMAGE_NAME}:${IMAGE_LAST_TAG}"

LATEST_COMMIT=$(git rev-parse HEAD)
LATEST_COMMIT_TAG=${LATEST_COMMIT:0:7}

if [[ $LATEST_COMMIT_TAG != $IMAGE_LAST_TAG ]]; then
    echo "Latest commit does not match latest image."
    exit 1
fi

echo "JOB_ID: ${JOB_ID}"
echo "IMAGE_URI: ${IMAGE_URI}"

gcloud ai-platform jobs submit training "${JOB_ID}_eval" \
    --config gcp_train_config.yaml \
    --region us-central1 \
    --master-image-uri $IMAGE_URI \
    --job-dir "${JOBS_DIR}${JOB_ID}" \
    -- \
    --eval True \
    --benchmark-games 30 \
    --agent-net-size 256 \
    --agent-net-conv 5 \
    --mcts-iter 50
