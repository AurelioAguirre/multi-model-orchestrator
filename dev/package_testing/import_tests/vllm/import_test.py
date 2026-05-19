#!/usr/bin/env python3
"""
vLLM Import Test
Tests core vLLM functionality and dependencies
"""

def test_vllm_imports():
    """Test basic vLLM imports"""
    print("Testing vLLM imports...")
    
    try:
        # Core vLLM
        import vllm
        print(f"✅ vllm: {vllm.__version__}")
        
        # PyTorch
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Transformers
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
        
        # vLLM specific dependencies
        import triton
        print(f"✅ triton: {triton.__version__}")
        
        try:
            import xformers
            print(f"✅ xformers: {xformers.__version__}")
        except ImportError:
            print("⚠️  xformers not available (optional)")
        
        # Ray for distributed
        import ray
        print(f"✅ ray: {ray.__version__}")
        
        # Numeric computing
        import numpy
        print(f"✅ numpy: {numpy.__version__}")
        
        # Web framework
        import fastapi
        import uvicorn
        print(f"✅ fastapi: {fastapi.__version__}")
        print(f"✅ uvicorn: {uvicorn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_vllm_functionality():
    """Test basic vLLM functionality"""
    print("\nTesting vLLM functionality...")
    
    try:
        from vllm import LLM, SamplingParams
        
        print("✅ vLLM LLM and SamplingParams imported")
        
        # Test SamplingParams creation
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)
        print(f"✅ SamplingParams created: temp={sampling_params.temperature}")
        
        # Test basic vLLM imports that are used in the service
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        print("✅ AsyncLLMEngine imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("vLLM Import & Functionality Test")
    print("=" * 50)
    
    imports_ok = test_vllm_imports()
    
    if imports_ok:
        functionality_ok = test_vllm_functionality()
        
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