#!/usr/bin/env python3
"""
HuggingFace Model Downloader

A standalone utility to download complete HuggingFace models with all files.
Ensures models are fully downloaded and ready for local inference.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Default download directory (relative to project root)
DOWNLOAD_DIR = Path(__file__).parent.parent / "models"

# HuggingFace access token for private/gated models
# Set your token here for models that require authentication
HF_TOKEN = None  # Replace with your token: "hf_your_token_here"


def list_downloaded_models() -> List[str]:
    """List all models that have been downloaded to the default directory."""
    if not DOWNLOAD_DIR.exists():
        return []
    
    models = []
    for provider_dir in DOWNLOAD_DIR.iterdir():
        if provider_dir.is_dir() and not provider_dir.name.startswith('.'):
            for model_dir in provider_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    # Check if it looks like a model (has config.json)
                    if (model_dir / "config.json").exists():
                        models.append(f"{provider_dir.name}/{model_dir.name}")
    
    return sorted(models)


def download_model(model_name: str, download_dir: Optional[Path] = None) -> bool:
    """Download a complete HuggingFace model."""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub library not available.")
        print("Install with: pip install huggingface_hub")
        return False
    
    if download_dir is None:
        download_dir = DOWNLOAD_DIR
    
    # Parse model name (e.g., "openai/gpt-oss-20b")
    if '/' not in model_name:
        print(f"Error: Model name must be in format 'provider/model', got: {model_name}")
        return False
    
    provider, model = model_name.split('/', 1)
    local_dir = download_dir / provider / model
    
    print(f"Downloading {model_name} to {local_dir}")
    print("This may take a while for large models...")
    
    try:
        # Create directory if it doesn't exist
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the complete model
        snapshot_download(
            repo_id=model_name,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
            resume_download=True,          # Resume if interrupted
            token=HF_TOKEN,                # Use authentication token if provided
        )
        
        print(f"✓ Successfully downloaded {model_name}")
        print(f"  Location: {local_dir}")
        
        # Verify download by checking for key files
        required_files = ["config.json"]
        missing_files = [f for f in required_files if not (local_dir / f).exists()]
        
        if missing_files:
            print(f"⚠ Warning: Some expected files are missing: {missing_files}")
            return False
        
        # Check for model weights
        weight_patterns = ["*.safetensors", "*.bin"]
        has_weights = any(
            list(local_dir.glob(pattern)) 
            for pattern in weight_patterns
        )
        
        if not has_weights:
            print("⚠ Warning: No model weight files found")
            return False
        
        print("✓ Model verification passed")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models or list downloaded models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # List downloaded models
  %(prog)s openai/gpt-oss-20b        # Download model to default location
  %(prog)s openai/gpt-oss-20b ./custom_models  # Download to custom location
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs='?',
        help="Model to download in format 'provider/model' (e.g., 'openai/gpt-oss-20b')"
    )
    
    parser.add_argument(
        "download_dir",
        nargs='?',
        type=Path,
        help=f"Download directory (default: {DOWNLOAD_DIR})"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List downloaded models"
    )
    
    args = parser.parse_args()
    
    # If no arguments or --list flag, show downloaded models
    if not args.model_name or args.list:
        print("Downloaded models:")
        models = list_downloaded_models()
        
        if not models:
            print("  No models found in", DOWNLOAD_DIR)
            print(f"  Download a model with: {sys.argv[0]} <provider/model>")
        else:
            for model in models:
                print(f"  {model}")
        return
    
    # Download model
    success = download_model(args.model_name, args.download_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()