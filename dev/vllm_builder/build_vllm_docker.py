#!/usr/bin/env python3
"""
Docker-based vLLM v0.10.1rc1 wheel builder with MXFP4 support
This script helps build or download vLLM v0.10.1rc1 wheel files for offline installation
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(f"SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True

def download_gptoss_wheel():
    """Download pre-built vLLM wheel with GPT-OSS support"""
    print("\n🚀 Option 1: Downloading pre-built vLLM v0.10.1+gptoss wheel")
    
    # Create wheels directory
    wheels_dir = Path("../wheels")
    wheels_dir.mkdir(exist_ok=True)
    
    # Download the wheel using pip download
    cmd = [
        "pip", "download",
        "--pre", "vllm==0.10.1+gptoss",
        "--extra-index-url", "https://wheels.vllm.ai/gpt-oss/",
        "--extra-index-url", "https://download.pytorch.org/whl/nightly/cu128",
        "--index-strategy", "unsafe-best-match",
        "--dest", "../wheels/",
        "--no-deps"  # Only download vLLM wheel
    ]
    
    if run_command(cmd, "Download vLLM v0.10.1+gptoss wheel"):
        print("\n✅ Successfully downloaded vLLM wheel to wheels/ directory")
        print("You can now use this for offline installation")
        return True
    else:
        print("\n❌ Failed to download pre-built wheel")
        return False

def build_with_docker():
    """Build vLLM v0.10.1rc1 from source using Docker"""
    print("\n🐳 Option 2: Building vLLM v0.10.1rc1 from source using Docker")
    
    dockerfile_content = '''
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3.12 \\
    python3.12-dev \\
    python3.12-venv \\
    python3-pip \\
    git \\
    build-essential \\
    cmake \\
    ninja-build \\
    ccache \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.4
RUN pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \\
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

# Build wheel with MXFP4 support
RUN python setup.py bdist_wheel

# Create output directory
RUN mkdir -p /wheels
RUN cp dist/*.whl /wheels/

CMD ["ls", "-la", "/wheels/"]
'''
    
    # Write Dockerfile
    with open("Dockerfile.vllm", "w") as f:
        f.write(dockerfile_content)
    
    # Build Docker image
    build_cmd = [
        "docker", "build",
        "-f", "Dockerfile.vllm",
        "-t", "vllm-builder:v0.10.1rc1",
        "."
    ]
    
    if not run_command(build_cmd, "Build vLLM Docker image"):
        return False
    
    # Run container to build wheel
    wheels_dir = Path("../wheels").absolute()
    wheels_dir.mkdir(exist_ok=True)
    
    run_cmd = [
        "docker", "run",
        "--rm",
        "-v", f"{wheels_dir}:/output",
        "vllm-builder:v0.10.1rc1",
        "sh", "-c", "cp /wheels/*.whl /output/"
    ]
    
    if run_command(run_cmd, "Extract wheel from Docker container"):
        print("\n✅ Successfully built vLLM wheel using Docker")
        print(f"Wheel file saved to: {wheels_dir}")
        return True
    else:
        print("\n❌ Failed to extract wheel from Docker container")
        return False

def verify_wheel():
    """Verify the downloaded/built wheel"""
    wheels_dir = Path("../wheels")
    vllm_wheels = list(wheels_dir.glob("vllm*.whl"))
    
    if not vllm_wheels:
        print("\n❌ No vLLM wheels found in wheels/ directory")
        return False
    
    print(f"\n✅ Found vLLM wheel(s):")
    for wheel in vllm_wheels:
        print(f"  - {wheel.name}")
    
    return True

def main():
    """Main function to coordinate wheel building/downloading"""
    print("vLLM v0.10.1rc1 Wheel Builder with MXFP4 Support")
    print("=" * 60)
    
    print("\nChoose your approach:")
    print("1) Download pre-built vLLM v0.10.1+gptoss wheel (Recommended)")
    print("2) Build vLLM v0.10.1rc1 from source using Docker")
    print("3) Both (download first, build as backup)")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")
    
    success = False
    
    if choice in ["1", "3"]:
        success = download_gptoss_wheel()
        if success and choice == "1":
            verify_wheel()
            return
    
    if choice in ["2", "3"]:
        if choice == "3" and not success:
            print("\n📦 Download failed, trying Docker build...")
        
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n❌ Docker is not available. Please install Docker first.")
            return
        
        success = build_with_docker()
    
    if success:
        verify_wheel()
        print("\n🎉 Success! You can now use the wheel for offline installation")
        print("Update your install.sh to use the wheel from wheels/ directory")
    else:
        print("\n❌ All attempts failed. Consider using the standard vLLM v0.10.0 installation")

if __name__ == "__main__":
    main()