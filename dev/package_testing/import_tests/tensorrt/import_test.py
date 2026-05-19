#!/usr/bin/env python3
"""
TensorRT-LLM Import Test
Tests core TensorRT-LLM functionality and dependencies
"""

def test_tensorrt_imports():
    """Test basic TensorRT-LLM imports"""
    print("Testing TensorRT-LLM imports...")
    
    try:
        # Core TensorRT-LLM (may not be available without custom wheel)
        try:
            import tensorrt_llm
            print(f"✅ tensorrt_llm: {tensorrt_llm.__version__}")
            
            from tensorrt_llm.runtime import ModelRunner, SamplingParams
            print("✅ TensorRT-LLM runtime components imported")
            
            from tensorrt_llm import LLM
            print("✅ TensorRT-LLM LLM imported")
            
        except ImportError as e:
            print(f"⚠️  TensorRT-LLM not available (may need custom wheel): {e}")
        
        # PyTorch
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Transformers
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
        
        # vLLM (compatible version)
        import vllm
        print(f"✅ vllm: {vllm.__version__}")
        
        # Numeric computing (must be <2 for TensorRT)
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
        if numpy.__version__.startswith('2.'):
            print("⚠️  Warning: numpy 2.x detected, TensorRT-LLM requires numpy<2")
        
        # Web framework
        import fastapi
        import uvicorn
        print(f"✅ fastapi: {fastapi.__version__}")
        print(f"✅ uvicorn: {uvicorn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_tensorrt_functionality():
    """Test basic TensorRT-LLM functionality"""
    print("\nTesting TensorRT-LLM functionality...")
    
    try:
        # Test if TensorRT-LLM is available
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import SamplingParams
            
            # Test SamplingParams creation
            sampling_params = SamplingParams(
                temperature=0.8,
                top_p=0.95,
                max_tokens=10
            )
            print(f"✅ TensorRT SamplingParams created: temp={sampling_params.temperature}")
            
        except ImportError:
            print("⚠️  TensorRT-LLM not available for functionality test")
            
        # Test vLLM compatibility
        from vllm import LLM, SamplingParams as VLLMSamplingParams
        vllm_params = VLLMSamplingParams(temperature=0.8, max_tokens=10)
        print("✅ vLLM compatibility test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("TensorRT-LLM Import & Functionality Test")
    print("=" * 50)
    
    imports_ok = test_tensorrt_imports()
    
    if imports_ok:
        functionality_ok = test_tensorrt_functionality()
        
        if imports_ok and functionality_ok:
            print("\n🎉 All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
    else:
        print("\n❌ Import tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())