#!/usr/bin/env python3
"""
Build script that runs inside the Docker container to compile vLLM
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description="", cwd=None):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=cwd, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"SUCCESS: {description}")
        return True

def main():
    """Main build process"""
    print("🔧 Starting vLLM v0.10.1rc1 build process")
    print("=" * 60)
    
    # Verify we're in the right directory
    vllm_dir = Path("/workspace/vllm")
    if not vllm_dir.exists():
        print("❌ vLLM directory not found!")
        sys.exit(1)
    
    print(f"📂 Working in: {vllm_dir}")
    
    # Show git status
    if not run_command(["git", "status"], "Check git status", cwd=vllm_dir):
        print("⚠️  Git status check failed, continuing anyway...")
    
    # Show current branch/commit
    if not run_command(["git", "log", "--oneline", "-1"], "Show current commit", cwd=vllm_dir):
        print("⚠️  Git log check failed, continuing anyway...")
    
    # Clean any previous builds
    print("\n🧹 Cleaning previous builds...")
    dist_dir = vllm_dir / "dist"
    if dist_dir.exists():
        run_command(["rm", "-rf", "dist"], "Remove dist directory", cwd=vllm_dir)
    
    build_dir = vllm_dir / "build"
    if build_dir.exists():
        run_command(["rm", "-rf", "build"], "Remove build directory", cwd=vllm_dir)
    
    # Show Python and CUDA info
    print("\n🔍 Environment information:")
    run_command(["python", "--version"], "Check Python version")
    run_command(["nvcc", "--version"], "Check CUDA version")
    run_command(["pip", "list"], "Show installed packages")
    
    # Build the wheel
    print("\n🏗️  Building vLLM wheel...")
    if not run_command(["python", "setup.py", "bdist_wheel"], "Build vLLM wheel", cwd=vllm_dir):
        print("❌ Wheel build failed!")
        sys.exit(1)
    
    # Copy wheel to output directory
    print("\n📦 Copying wheel to output directory...")
    dist_files = list((vllm_dir / "dist").glob("*.whl"))
    
    if not dist_files:
        print("❌ No wheel files found in dist/ directory!")
        sys.exit(1)
    
    wheels_dir = Path("/wheels")
    for wheel_file in dist_files:
        dest_file = wheels_dir / wheel_file.name
        if not run_command(["cp", str(wheel_file), str(dest_file)], f"Copy {wheel_file.name}"):
            print(f"❌ Failed to copy {wheel_file.name}")
            sys.exit(1)
        print(f"✅ Copied: {wheel_file.name}")
    
    # List final output
    print("\n🎉 Build complete! Output files:")
    run_command(["ls", "-la", "/wheels"], "List output wheels")
    
    print("\n✅ vLLM v0.10.1rc1 wheel build completed successfully!")

if __name__ == "__main__":
    main()