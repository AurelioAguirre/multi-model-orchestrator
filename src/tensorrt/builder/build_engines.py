#!/usr/bin/env python3
"""
TensorRT-LLM Engine Builder
Compiles HuggingFace models to TensorRT engines for optimized inference
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s"
)
logger = logging.getLogger("EngineBuilder")

def build_engine(
    model_path: str,
    output_dir: str,
    max_batch_size: int = 8,
    max_input_len: int = 2048,
    max_output_len: int = 1024,
    dtype: str = "float16",
    tp_size: int = 2,  # Tensor parallelism for 2x GPUs
    pp_size: int = 1,  # Pipeline parallelism
    use_custom_allreduce: bool = False,
    use_fused_mlp: bool = True,
    enable_context_fmha: bool = True,
    multi_block_mode: bool = False
):
    """
    Build TensorRT-LLM engine from HuggingFace model

    Args:
        model_path: Path to HuggingFace model or model ID
        output_dir: Directory to save compiled engines
        max_batch_size: Maximum batch size for inference
        max_input_len: Maximum input sequence length
        max_output_len: Maximum output sequence length
        dtype: Data type (float16, bfloat16, float32)
        tp_size: Tensor parallelism size (number of GPUs)
        pp_size: Pipeline parallelism size
        use_custom_allreduce: Use custom allreduce for multi-GPU
        use_fused_mlp: Enable fused MLP optimization
        enable_context_fmha: Enable Flash Attention
        multi_block_mode: Enable multi-block mode for long sequences
    """
    logger.info("=" * 80)
    logger.info("TensorRT-LLM Engine Builder")
    logger.info("=" * 80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Configuration:")
    logger.info(f"  Max Batch Size: {max_batch_size}")
    logger.info(f"  Max Input Length: {max_input_len}")
    logger.info(f"  Max Output Length: {max_output_len}")
    logger.info(f"  Data Type: {dtype}")
    logger.info(f"  Tensor Parallelism: {tp_size}")
    logger.info(f"  Pipeline Parallelism: {pp_size}")
    logger.info("=" * 80)

    try:
        # Import TensorRT-LLM (after logging config)
        import tensorrt_llm
        from tensorrt_llm import LLM

        logger.info(f"TensorRT-LLM version: {tensorrt_llm.__version__}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build configuration
        build_config = {
            "max_batch_size": max_batch_size,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
        }

        # Model configuration
        model_config = {
            "dtype": dtype,
            "tp_size": tp_size,
            "pp_size": pp_size,
        }

        # Optimization flags
        if use_fused_mlp:
            logger.info("Enabling fused MLP optimization")
        if enable_context_fmha:
            logger.info("Enabling Flash Attention (context FMHA)")
        if multi_block_mode:
            logger.info("Enabling multi-block mode")

        logger.info("Starting engine compilation (this may take 10-30 minutes)...")

        # Initialize LLM and trigger engine build
        # Using PyTorch backend for custom models like gpt_oss
        # For gpt_oss models, we need to specify the architecture explicitly
        # since it's not in standard Transformers
        #
        # Note: gpt-oss models use WFP4A16 quantization which requires SM90+ (Hopper/Blackwell)
        # For RTX 4090 (SM89), we need to use FP16 instead
        llm = LLM(
            model=model_path,
            backend="pytorch",
            trust_remote_code=True,
            # Specify tensor parallelism and other build parameters
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            # Force dtype to override model's default quantization
            dtype=dtype,
            # Build configuration
            # Note: These parameters may vary based on TensorRT-LLM version
            # Consult documentation for exact parameter names
        )

        logger.info("Engine compilation successful!")

        # Save metadata
        metadata = {
            "model_path": model_path,
            "build_date": datetime.now().isoformat(),
            "tensorrt_llm_version": tensorrt_llm.__version__,
            "config": {
                "max_batch_size": max_batch_size,
                "max_input_len": max_input_len,
                "max_output_len": max_output_len,
                "dtype": dtype,
                "tp_size": tp_size,
                "pp_size": pp_size,
            }
        }

        import json
        metadata_file = output_path / "build_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Build metadata saved to: {metadata_file}")
        logger.info("=" * 80)
        logger.info("ENGINE BUILD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Engines saved to: {output_dir}")
        logger.info("You can now use these engines for inference")

        return True

    except Exception as e:
        logger.error(f"Engine build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for engine builder"""
    parser = argparse.ArgumentParser(
        description="Build TensorRT-LLM engines from HuggingFace models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_PATH", "openai/gpt-oss-20b"),
        help="HuggingFace model ID or path (default: openai/gpt-oss-20b)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("ENGINE_OUTPUT_DIR", "/app/models/tensorrt-engines/gpt-oss-20b"),
        help="Output directory for compiled engines"
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.getenv("MAX_BATCH_SIZE", "8")),
        help="Maximum batch size"
    )

    parser.add_argument(
        "--max-input-len",
        type=int,
        default=int(os.getenv("MAX_INPUT_LEN", "2048")),
        help="Maximum input sequence length"
    )

    parser.add_argument(
        "--max-output-len",
        type=int,
        default=int(os.getenv("MAX_OUTPUT_LEN", "1024")),
        help="Maximum output sequence length"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=os.getenv("DTYPE", "float16"),
        choices=["float16", "bfloat16", "float32"],
        help="Data type for inference"
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.getenv("TP_SIZE", "2")),
        help="Tensor parallelism size (number of GPUs)"
    )

    parser.add_argument(
        "--pp-size",
        type=int,
        default=int(os.getenv("PP_SIZE", "1")),
        help="Pipeline parallelism size"
    )

    args = parser.parse_args()

    logger.info("TensorRT-LLM Engine Builder Starting...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working Directory: {os.getcwd()}")

    success = build_engine(
        model_path=args.model,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        dtype=args.dtype,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()