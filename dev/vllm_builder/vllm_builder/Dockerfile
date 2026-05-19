FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone vLLM repository and checkout v0.10.1rc1
WORKDIR /workspace
RUN git clone https://github.com/vllm-project/vllm.git
WORKDIR /workspace/vllm
RUN git checkout v0.10.1rc1

# Install build dependencies
RUN pip install -r requirements-build.txt

# Set CUDA architecture (adjust for your GPU)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Copy build script
COPY build_script.py /workspace/build_script.py

# Create output directory
RUN mkdir -p /wheels

# Run the build script
CMD ["python", "/workspace/build_script.py"]