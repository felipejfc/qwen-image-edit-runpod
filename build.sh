#!/bin/bash
# Build script for cross-platform Docker image (M1 Mac -> x86_64 RunPod)

set -e

IMAGE_NAME=${1:-"qwen-image-edit-runpod"}
IMAGE_TAG=${2:-"latest"}

echo "Building Docker image for linux/amd64 platform..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Build for amd64 platform (required for RunPod x86_64 GPUs)
docker buildx build \
    --platform linux/amd64 \
    --file .runpod/Dockerfile \
    --tag ${IMAGE_NAME}:${IMAGE_TAG} \
    --load \
    .

echo ""
echo "✅ Build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to a registry:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  docker push your-registry/${IMAGE_NAME}:${IMAGE_TAG}"

