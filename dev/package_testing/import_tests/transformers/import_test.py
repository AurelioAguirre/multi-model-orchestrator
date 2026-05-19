#!/usr/bin/env python3
"""
Transformers Import Test
Tests core HuggingFace Transformers functionality
"""

def test_transformers_imports():
    """Test basic transformers imports"""
    print("Testing Transformers imports...")
    
    try:
        # Core transformers
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
        
        # PyTorch
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Tokenizers
        import tokenizers
        print(f"✅ tokenizers: {tokenizers.__version__}")
        
        # HuggingFace Hub
        import huggingface_hub
        print(f"✅ huggingface_hub: {huggingface_hub.__version__}")
        
        # Safetensors
        import safetensors
        print(f"✅ safetensors: {safetensors.__version__}")
        
        # Accelerate
        import accelerate
        print(f"✅ accelerate: {accelerate.__version__}")
        
        # Web framework
        import fastapi
        import uvicorn
        print(f"✅ fastapi: {fastapi.__version__}")
        print(f"✅ uvicorn: {uvicorn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_transformers_functionality():
    """Test basic transformers functionality"""
    print("\nTesting Transformers functionality...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test tokenizer loading (small model)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        print(f"✅ Tokenizer loaded: {len(tokenizer.vocab)} vocab size")
        
        # Test basic tokenization
        text = "Hello world"
        tokens = tokenizer.encode(text)
        print(f"✅ Tokenization works: '{text}' -> {tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Transformers Import & Functionality Test")
    print("=" * 50)
    
    imports_ok = test_transformers_imports()
    
    if imports_ok:
        functionality_ok = test_transformers_functionality()
        
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