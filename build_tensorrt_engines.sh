#!/bin/bash

# TensorRT Engine Builder Runner
# Runs the builder container as a one-time job to compile TensorRT engines

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=========================================="
echo "TensorRT-LLM Engine Builder"
echo "=========================================="
echo

# Configuration (can be overridden via environment variables)
MODEL_PATH=${MODEL_PATH:-"openai/gpt-oss-20b"}
ENGINE_OUTPUT_DIR=${ENGINE_OUTPUT_DIR:-"/app/models/tensorrt-engines/gpt-oss-20b"}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-"8"}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-"2048"}
MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN:-"1024"}
DTYPE=${DTYPE:-"float16"}
TP_SIZE=${TP_SIZE:-"2"}
PP_SIZE=${PP_SIZE:-"1"}

print_status "Build Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output Directory: $ENGINE_OUTPUT_DIR"
echo "  Max Batch Size: $MAX_BATCH_SIZE"
echo "  Max Input Length: $MAX_INPUT_LEN"
echo "  Max Output Length: $MAX_OUTPUT_LEN"
echo "  Data Type: $DTYPE"
echo "  Tensor Parallelism: $TP_SIZE GPUs"
echo "  Pipeline Parallelism: $PP_SIZE"
echo

print_warning "This process will take 10-30 minutes depending on model size and hardware."
echo

# Check if image exists
if ! podman image exists llm-tensorrt-builder:latest; then
    print_error "Builder image not found!"
    print_status "Please build it first with:"
    echo "  ./run_containers.sh → Build specific image → TensorRT Builder"
    exit 1
fi

print_status "Starting TensorRT engine build..."
echo

# Run builder container
# - Mounts models directory (shared with inference pod)
# - Uses all GPUs
# - Runs as one-time job (--rm removes container after completion)
# - Passes configuration via environment variables
podman run \
    --rm \
    --name llm-tensorrt-builder-job \
    -v ./models:/app/models:Z \
    --device nvidia.com/gpu=all \
    -e MODEL_PATH="$MODEL_PATH" \
    -e ENGINE_OUTPUT_DIR="$ENGINE_OUTPUT_DIR" \
    -e MAX_BATCH_SIZE="$MAX_BATCH_SIZE" \
    -e MAX_INPUT_LEN="$MAX_INPUT_LEN" \
    -e MAX_OUTPUT_LEN="$MAX_OUTPUT_LEN" \
    -e DTYPE="$DTYPE" \
    -e TP_SIZE="$TP_SIZE" \
    -e PP_SIZE="$PP_SIZE" \
    -e CUDA_VISIBLE_DEVICES=0,1 \
    llm-tensorrt-builder:latest

BUILD_EXIT_CODE=$?

echo
echo "=========================================="

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    print_success "Engine build completed successfully!"
    echo
    print_status "Engines saved to: ./models/tensorrt-engines/gpt-oss-20b/"
    echo
    print_status "Next steps:"
    echo "  1. Verify engines exist:"
    echo "     ls -lh ./models/tensorrt-engines/gpt-oss-20b/"
    echo
    echo "  2. Update TensorRT inference pod to use pre-compiled engines"
    echo "     (Set engine_dir in config or provider initialization)"
    echo
    echo "  3. Restart TensorRT inference container:"
    echo "     ./run_containers.sh → Start specific container → TensorRT"
else
    print_error "Engine build failed!"
    print_status "Check the logs above for error details"
    exit 1
fi

echo "=========================================="