# vLLM v0.10.1rc1 Docker Builder

This directory contains a clean, modular approach to building vLLM v0.10.1rc1 wheels with MXFP4 quantization support using Docker.

## Structure

```
vllm_builder/
├── Dockerfile              # Docker container definition
├── build_script.py         # Script that runs inside the container
├── run_build.py            # Main orchestration script
└── README.md              # This file
```

## Files

### `Dockerfile`
- Defines the build environment with CUDA 12.4.1
- Installs all necessary dependencies
- Sets up Python 3.12 virtual environment
- Clones vLLM repository and checks out v0.10.1rc1

### `build_script.py` 
- Runs inside the Docker container
- Handles the actual wheel building process
- Provides detailed logging and error handling
- Copies built wheels to output directory

### `run_build.py`
- Main orchestration script run from project root
- Manages Docker image building and container execution
- Handles file extraction and verification
- Provides user-friendly progress updates

## Usage

### Prerequisites
- Docker installed and running
- NVIDIA Docker runtime (for GPU access during build)
- Run from the project root directory

### Build Process

1. **From the project root directory**, run:
   ```bash
   python utils/vllm_builder/run_build.py
   ```

2. The script will:
   - Check Docker availability
   - Build the Docker image with vLLM v0.10.1rc1
   - Run the container to compile the wheel
   - Extract the wheel to `wheels/` directory

3. Upon success, you'll find the wheel file in `wheels/vllm-*.whl`

### Manual Debugging

If the automated build fails, you can debug manually:

```bash
# Build the image
cd utils/vllm_builder
docker build -t vllm-builder:v0.10.1rc1 .

# Run container interactively for debugging
docker run -it --rm --gpus all vllm-builder:v0.10.1rc1 /bin/bash

# Inside container, run build script manually
python /workspace/build_script.py
```

## Troubleshooting

### Common Issues

1. **Docker not found**: Install Docker Desktop or Docker Engine
2. **NVIDIA runtime not available**: Install nvidia-docker2 package
3. **Build fails**: Check CUDA compatibility with your GPU
4. **Out of memory**: Increase Docker memory limits

### Build Logs

All build steps provide detailed logging. Look for:
- CUDA version compatibility
- PyTorch installation success  
- Git checkout confirmation
- Wheel file creation
- Output file verification

## Output

Successfully built wheels will be saved to the project's `wheels/` directory and can be used for offline installation in your server environment.