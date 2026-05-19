#!/usr/bin/env python3
"""
Main orchestration script to build vLLM v0.10.1rc1 wheel using Docker
Run this from the project root directory
"""
import subprocess
import sys
import os
from pathlib import Path
import shutil

def run_command(cmd, description="", cwd=None, env=None):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")
    if env and "DOCKER_BUILDKIT" in env:
        print("Environment: DOCKER_BUILDKIT=1")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True)
    
    if result.returncode != 0:
        print(f"❌ ERROR: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        return True

def check_requirements():
    """Check if Docker is available"""
    print("🔍 Checking requirements...")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
        else:
            print("❌ Docker command failed")
            return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
        return False
    
    # Check if NVIDIA Docker runtime is available
    try:
        result = subprocess.run(["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.4.1-runtime-ubuntu22.04", "nvidia-smi"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA Docker runtime is working")
        else:
            print("⚠️  NVIDIA Docker runtime test failed. Build may still work without GPU access.")
    except:
        print("⚠️  Could not test NVIDIA Docker runtime")
    
    return True

def main():
    """Main build orchestration"""
    print("🐳 vLLM v0.10.1rc1 Docker Build Orchestrator")
    print("=" * 60)
    
    # Determine project root (should be run from project root)
    current_dir = Path.cwd()
    if not (current_dir / "src").exists() or not (current_dir / "utils").exists():
        print("❌ Please run this script from the project root directory")
        print("   (The directory containing src/ and utils/ folders)")
        sys.exit(1)
    
    project_root = current_dir
    builder_dir = project_root / "utils" / "vllm_builder"
    
    if not builder_dir.exists():
        print(f"❌ Builder directory not found: {builder_dir}")
        sys.exit(1)
    
    print(f"📂 Project root: {project_root}")
    print(f"📂 Builder directory: {builder_dir}")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create wheels directory in project root
    wheels_dir = project_root / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    print(f"📦 Wheels directory: {wheels_dir}")
    
    # Approach 1: Try using official vLLM Dockerfile
    print("\n🏗️  Building Docker image using official vLLM approach...")
    print("Option 1: Using official vLLM Dockerfile from GitHub...")
    
    # Clone vLLM repo temporarily to get official Dockerfile
    temp_vllm_dir = project_root / "temp_vllm"
    if temp_vllm_dir.exists():
        shutil.rmtree(temp_vllm_dir)
    
    if not run_command([
        "git", "clone", "--depth", "1", "--branch", "v0.10.1rc1",
        "https://github.com/vllm-project/vllm.git",
        "temp_vllm"
    ], "Clone vLLM repository", cwd=project_root):
        print("❌ Failed to clone vLLM repository!")
        print("📦 Falling back to custom Dockerfile...")
        # Fallback to our custom approach
        if not run_command([
            "docker", "build",
            "-t", "vllm-builder:v0.10.1rc1",
            "."
        ], "Build vLLM Docker image (custom)", cwd=builder_dir):
            print("❌ Docker image build failed!")
            sys.exit(1)
    else:
        # Use official Dockerfile with BuildKit enabled
        build_env = os.environ.copy()
        build_env["DOCKER_BUILDKIT"] = "1"
        
        if not run_command([
            "docker", "build",
            "--target", "build",
            "-t", "vllm-builder:v0.10.1rc1",
            "-f", "docker/Dockerfile",
            "--build-arg", "PYTHON_VERSION=3.12",
            "--build-arg", "RUN_WHEEL_CHECK=false",
            "."
        ], "Build vLLM Docker image (official with BuildKit)", cwd=temp_vllm_dir, env=build_env):
            print("❌ Official Docker build failed!")
            # Cleanup and exit
            shutil.rmtree(temp_vllm_dir)
            sys.exit(1)
        
        # Cleanup temp directory
        shutil.rmtree(temp_vllm_dir)
    
    # Extract wheel from built image
    print("\n📦 Extracting wheel from Docker image...")
    
    # The official build creates wheels in /workspace/dist/ 
    if not run_command([
        "docker", "run",
        "--rm",
        "-v", f"{wheels_dir}:/output",
        "vllm-builder:v0.10.1rc1",
        "sh", "-c", "find /workspace -name '*.whl' -exec cp {} /output/ \\; || find /opt/vllm -name '*.whl' -exec cp {} /output/ \\; || echo 'Searching for wheels in container...'"
    ], "Extract wheel from container"):
        print("⚠️  Wheel extraction command failed, trying alternative method...")
        
        # Alternative: Create a temporary container to explore
        print("🔍 Exploring container for wheel files...")
        run_command([
            "docker", "run",
            "--rm",
            "vllm-builder:v0.10.1rc1",
            "find", "/", "-name", "*.whl", "-type", "f"
        ], "Find wheel files in container")
    
    # Verify output
    print("\n🔍 Checking output...")
    wheel_files = list(wheels_dir.glob("vllm*.whl"))
    
    if wheel_files:
        print("🎉 Success! Built wheel files:")
        for wheel in wheel_files:
            file_size = wheel.stat().st_size / (1024 * 1024)  # MB
            print(f"  📄 {wheel.name} ({file_size:.1f} MB)")
        
        print(f"\n✅ Wheels saved to: {wheels_dir}")
        print("   You can now use these wheels for offline installation!")
        
        # Show install command
        print(f"\n💡 To install the built wheel:")
        for wheel in wheel_files:
            print(f"   pip install {wheels_dir}/{wheel.name}")
    else:
        print("❌ No vLLM wheel files found in output directory")
        print("   Check the Docker build logs above for errors")
        sys.exit(1)

if __name__ == "__main__":
    main()