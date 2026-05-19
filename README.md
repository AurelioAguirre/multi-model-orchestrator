# LLM Tensor Server

A **microservices-based** high-performance inference server supporting multiple ML frameworks with OpenAI-compatible APIs.

## Supported Engines

- **🤗 Transformers**: HuggingFace Transformers with PyTorch
- **⚡ vLLM**: High-performance inference with advanced optimizations
- **🚀 TensorRT-LLM**: NVIDIA's optimized inference engine

## Architecture

### Clean Microservices Design
Each ML framework runs as an independent application with isolated dependencies:

```
┌─────────────────┐    HTTP    ┌─────────────────┐
│   Orchestrator  │ ────────→ │  Transformers   │
│   (API Gateway) │            │   Application   │
└─────────────────┘            └─────────────────┘
         │                              
         │         HTTP     ┌─────────────────┐
         ├─────────────────→│     vLLM        │
         │                  │  Application    │
         │                  └─────────────────┘
         │                              
         │         HTTP     ┌─────────────────┐
         └─────────────────→│   TensorRT-LLM  │
                            │   Application   │
                            └─────────────────┘
```

### Key Benefits
- **🔒 Dependency Isolation**: No version conflicts between frameworks
- **🎯 Independent Scaling**: Scale individual services based on demand  
- **🔧 Easy Maintenance**: Update one framework without affecting others
- **📦 Clean Deployment**: Each service in its own container

## Quick Start

### Prerequisites
- **Python 3.12+**
- **Podman** (rootless containers with GPU support)
- **NVIDIA GPU** with CUDA support (for ML inference)

### Option 1: Local Development (Orchestrator Only)
```bash
# Install orchestrator dependencies
./install.sh

# Run orchestrator locally (requires microservices running separately)  
python src/orchestrator/main.py
```

### Option 2: Full Microservices (Recommended)
```bash
# Interactive container management
./run_containers.sh

# Select:
# 1. Build all services
# 3. Start all services
```

### Manual Container Commands
```bash
# Build all services
cd containers/
podman build -f Containerfile.orchestrator -t llm-orchestrator .
podman build -f Containerfile.transformers -t llm-transformers .
podman build -f Containerfile.vllm -t llm-vllm .
podman build -f Containerfile.tensorrt -t llm-tensorrt .

# Start with compose (if available)
podman-compose -f compose.yml --profile full up -d
```

## API Usage

### Health Checks
```bash
curl http://localhost:8011/health      # Orchestrator
curl http://localhost:8012/health      # Transformers service  
curl http://localhost:8013/health      # vLLM service
curl http://localhost:8014/health      # TensorRT service
```

### Load Models
```bash
# Load models via orchestrator (routes to microservices)
curl -X POST "http://localhost:8011/load_transformers"
curl -X POST "http://localhost:8011/load_vllm" 
curl -X POST "http://localhost:8011/load_tensorrt"
```

### OpenAI-Compatible Inference
```bash
# Chat completions
curl -X POST "http://localhost:8011/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Text completions  
curl -X POST "http://localhost:8011/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b", 
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

## Project Structure

### Source Code
```
src/
├── orchestrator/           # API gateway application
│   ├── main.py            # FastAPI entry point
│   ├── routes.py          # API endpoints
│   ├── handlers/          # Business logic
│   └── clients/           # HTTP clients for microservices
├── transformers/          # Transformers microservice
│   └── main.py           # Transformers FastAPI service
├── vllm/                 # vLLM microservice  
│   └── main.py           # vLLM FastAPI service
├── tensorrt/             # TensorRT microservice
│   └── main.py           # TensorRT FastAPI service
└── shared/               # Common utilities
    ├── schemas.py        # OpenAI-compatible models
    ├── config.py         # Configuration loading
    └── *_provider.py     # ML framework providers
```

### Container Definitions
```
containers/
├── Containerfile.orchestrator    # Minimal orchestrator (CPU-only)
├── Containerfile.transformers    # Transformers + dependencies
├── Containerfile.vllm            # vLLM + custom wheels
├── Containerfile.tensorrt        # TensorRT-LLM + dependencies
└── compose.yml                   # Multi-container orchestration
```

### Configuration & Requirements
```
requirements/
├── orchestrator.txt       # Minimal deps for orchestrator
└── podman/               # Microservice-specific requirements
    ├── transformers.txt  # Standalone transformers deps
    ├── vllm.txt         # Standalone vLLM deps  
    └── tensorrt.txt     # Standalone TensorRT deps

resources/
└── config.yml           # Main configuration file
```

### Development Tools
```
dev/
├── package_testing/      # Package compatibility testing
│   └── run_import_tests.sh  # Test framework dependencies
└── legacy/               # Reference implementations
```

## Container Structure Reference

### Orchestrator Container (`/app/`)
```
/app/
├── requirements/
│   └── orchestrator.txt           # Minimal web framework deps
├── src/                          # Full source tree
│   ├── orchestrator/            # ← RUNS THIS
│   │   ├── main.py              # Entry point
│   │   ├── routes.py            # API routes
│   │   ├── handlers/            # Business logic
│   │   └── clients/             # HTTP clients
│   ├── shared/                  # ← USES THIS
│   │   ├── schemas.py           # API models
│   │   └── config.py            # Config loading
│   └── [other apps - present but unused]
├── resources/
│   └── config.yml               # Configuration
└── models/                      # Model storage (volume)

Entry Point: python src/orchestrator/main.py
Dependencies: FastAPI, httpx, pydantic (CPU-only)
```

### Transformers Container (`/app/`)
```
/app/
├── requirements/
│   └── transformers.txt         # Full ML stack + transformers
├── wheels/                      # Custom wheels (if any)
├── src/                        # Full source tree  
│   ├── transformers/           # ← RUNS THIS
│   │   └── main.py             # Entry point
│   ├── shared/                 # ← USES THIS
│   │   ├── schemas.py          # API models
│   │   ├── transformers_provider.py # Transformers implementation
│   │   └── config.py           # Config loading
│   └── [other apps - present but unused]
├── resources/
│   └── config.yml              # Configuration
└── models/                     # Model storage (volume)

Entry Point: python src/transformers/main.py
Dependencies: PyTorch, transformers, accelerate, etc.
```

### vLLM Container (`/app/`)
```
/app/
├── requirements/
│   └── vllm.txt                # Full ML stack + vLLM deps
├── wheels/                     # Custom vLLM wheels
│   └── vllm-*.whl             # Custom vLLM with GPT-OSS support
├── src/                       # Full source tree
│   ├── vllm/                  # ← RUNS THIS  
│   │   └── main.py            # Entry point (with streaming)
│   ├── shared/                # ← USES THIS
│   │   ├── schemas.py         # API models
│   │   ├── vllm_provider.py   # vLLM implementation
│   │   └── config.py          # Config loading
│   └── [other apps - present but unused]
├── resources/
│   └── config.yml             # Configuration
└── models/                    # Model storage (volume)

Entry Point: python src/vllm/main.py  
Dependencies: Custom vLLM wheel, PyTorch, triton, etc.
```

### TensorRT Container (`/app/`)
```
/app/
├── requirements/
│   └── tensorrt.txt           # Full ML stack + TensorRT deps
├── wheels/                    # Custom TensorRT wheels
│   └── tensorrt_llm-*.whl    # Custom TensorRT-LLM wheel
├── src/                      # Full source tree
│   ├── tensorrt/             # ← RUNS THIS
│   │   └── main.py           # Entry point  
│   ├── shared/               # ← USES THIS
│   │   ├── schemas.py        # API models
│   │   ├── tensorrt_provider.py # TensorRT implementation
│   │   └── config.py         # Config loading
│   └── [other apps - present but unused]
├── resources/
│   └── config.yml            # Configuration
└── models/                   # Model storage (volume)

Entry Point: python src/tensorrt/main.py
Dependencies: Custom TensorRT-LLM wheel, numpy<2, PyTorch, etc.
```

## Development

### Package Compatibility Testing
```bash
# Test all frameworks for package compatibility
cd dev/package_testing/
./run_import_tests.sh

# Test individual framework
./run_import_tests.sh transformers
./run_import_tests.sh vllm
./run_import_tests.sh tensorrt
```

### Adding New ML Frameworks
1. Create provider in `src/shared/[framework]_provider.py`
2. Create application in `src/[framework]/main.py`
3. Test requirements in `dev/package_testing/`
4. Create frozen requirements in `requirements/podman/[framework].txt`
5. Create container definition in `containers/Containerfile.[framework]`
6. Add HTTP client to orchestrator

### Configuration
- **Service URLs**: Set via environment variables
  - `TRANSFORMERS_SERVICE_URL=http://transformers:8012`
  - `VLLM_SERVICE_URL=http://vllm:8013` 
  - `TENSORRT_SERVICE_URL=http://tensorrt:8014`
- **Model Storage**: Shared `./models` volume across containers
- **GPU Allocation**: Configure via `CUDA_VISIBLE_DEVICES`

## Features

- ✅ **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- ✅ **Multiple ML Frameworks**: Transformers, vLLM, TensorRT-LLM
- ✅ **Streaming Support**: Real-time response streaming (vLLM)
- ✅ **Custom Model Support**: GPT-OSS and other specialized models
- ✅ **Container Orchestration**: Full microservices with Podman
- ✅ **Dependency Isolation**: No version conflicts between frameworks
- ✅ **Health Monitoring**: Comprehensive health checks
- ✅ **GPU Resource Management**: Efficient GPU allocation per service

## License

Copyright 2026 Aurelio Aguirre

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
