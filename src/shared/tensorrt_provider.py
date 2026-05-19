import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

# TensorRT-LLM availability check
_TENSORRT_LLM_AVAILABLE = False
_TENSORRT_IMPORT_ERROR = None

try:
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    import torch
    _TENSORRT_LLM_AVAILABLE = True
    _TENSORRT_IMPORT_ERROR = None
except ImportError as e:
    _TENSORRT_LLM_AVAILABLE = False
    _TENSORRT_IMPORT_ERROR = str(e)

from .base import BaseModelProvider
# from src.api.utils import SharedUtils, ChatTemplates  # TODO: Implement these utilities
from src.shared.schemas import ChatMessage


class TensorRTProvider(BaseModelProvider):
    """Model provider using TensorRT-LLM for optimized inference"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.tokenizer_path = None
        
        if not _TENSORRT_LLM_AVAILABLE:
            error_msg = f"TensorRT-LLM not available: {_TENSORRT_IMPORT_ERROR}"
            self.logger.error(error_msg)
            raise ImportError(error_msg)
        
        # TensorRT-LLM 1.0.0 can work with either pre-compiled engines or HuggingFace models
        self.engine_dir = getattr(config, 'engine_dir', None)
        if self.engine_dir:
            engine_path = Path(self.engine_dir)
            if not engine_path.exists():
                self.logger.warning(f"TensorRT engine directory not found: {engine_path}")
                self.logger.info("Will attempt to use HuggingFace model path for automatic compilation")
                self.engine_dir = None
    
    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """Simple chat message formatting (fallback for missing ChatTemplates)"""
        formatted_parts = []
        for message in messages:
            role = message.role.value if hasattr(message.role, 'value') else str(message.role)
            formatted_parts.append(f"{role.title()}: {message.content}")
        return "\n".join(formatted_parts) + "\nAssistant:"
    
    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (fallback for missing SharedUtils)"""
        # Rough estimation: ~4 characters per token for most models
        return len(text) // 4
    
    async def load_model(self, model_path: str = None) -> None:
        """Load TensorRT-LLM model"""
        try:
            # Set environment variables to avoid CUDA conflicts
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')  # RTX 4090 architecture
            if self.engine_dir:
                # Use pre-compiled TensorRT engines
                self.logger.info(f"Loading TensorRT-LLM model from engine directory: {self.engine_dir}")
                model_to_load = self.engine_dir
            else:
                # Use provided model_path or fall back to config default
                model_to_load = model_path or getattr(self.config, 'model_path', None) or getattr(self.config, 'default_model', None)
                if not model_to_load:
                    raise ValueError("No model_path specified in request or configuration")

                self.logger.info(f"Loading TensorRT-LLM model from HuggingFace: {model_to_load}")
                self.logger.info("TensorRT-LLM will compile engines automatically (this may take a while on first run)")

            # Initialize LLM with multiple fallback strategies
            initialization_successful = False

            # Strategy 1: Try TensorRT backend
            if not initialization_successful:
                try:
                    self.llm = LLM(
                        model=model_to_load,
                        tokenizer=getattr(self.config, 'tokenizer_path', None),
                        backend="tensorrt",
                        trust_remote_code=True
                    )
                    self.logger.info("Successfully loaded with TensorRT backend")
                    initialization_successful = True
                except Exception as e:
                    self.logger.warning(f"TensorRT backend failed: {e}")

            # Strategy 2: Try PyTorch backend
            if not initialization_successful:
                try:
                    self.llm = LLM(
                        model=model_to_load,
                        tokenizer=getattr(self.config, 'tokenizer_path', None),
                        backend="pytorch",
                        trust_remote_code=True
                    )
                    self.logger.info("Successfully loaded with PyTorch backend")
                    initialization_successful = True
                except Exception as e:
                    self.logger.warning(f"PyTorch backend failed: {e}")

            # Strategy 3: Try minimal initialization (auto-detect backend)
            if not initialization_successful:
                try:
                    self.llm = LLM(
                        model=model_to_load,
                        trust_remote_code=True
                    )
                    self.logger.info("Successfully loaded with auto-detected backend")
                    initialization_successful = True
                except Exception as e:
                    self.logger.error(f"All initialization strategies failed. Final error: {e}")
                    raise RuntimeError(f"Failed to initialize TensorRT-LLM with any backend: {e}")

            if not initialization_successful:
                raise RuntimeError("All TensorRT-LLM initialization strategies failed")

            self.tokenizer_path = getattr(self.config, 'tokenizer_path', None) or model_to_load
            self._model_loaded = True

            self.logger.info("TensorRT-LLM model loaded successfully")
            self.logger.info(f"Model path: {model_to_load}")
            self.logger.info(f"Tokenizer path: {self.tokenizer_path}")

        except Exception as e:
            self.logger.error(f"Failed to load TensorRT-LLM model: {e}")
            raise RuntimeError(f"TensorRT model loading failed: {e}")
    
    async def unload_model(self) -> None:
        """Unload TensorRT-LLM model"""
        try:
            if self.llm is not None:
                # TensorRT-LLM doesn't have explicit cleanup, but we can clear references
                del self.llm
                self.llm = None
                self._model_loaded = False
                
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info("TensorRT-LLM model unloaded successfully")
                
        except Exception as e:
            self.logger.error(f"Error unloading TensorRT-LLM model: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using TensorRT-LLM"""
        if not self._model_loaded:
            await self.load_model()
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.config.default_temperature),
                top_p=kwargs.get('top_p', self.config.default_top_p),
                top_k=kwargs.get('top_k', self.config.default_top_k),
                max_tokens=kwargs.get('max_new_tokens', self.config.default_max_tokens),
            )
            
            # Apply stop sequences if provided
            stop_sequences = kwargs.get('stop_sequences', [])
            if stop_sequences:
                sampling_params.stop = stop_sequences
            
            # Generate text
            outputs = await asyncio.to_thread(
                self.llm.generate,
                prompts=[prompt],
                sampling_params=sampling_params
            )
            
            if not outputs:
                raise RuntimeError("No output generated")
            
            output = outputs[0]
            generated_text = output.outputs[0].text
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"TensorRT-LLM text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages using TensorRT-LLM"""
        # Use chat template
        template_name = kwargs.get('chat_template', 'default')
        # template_func = ChatTemplates.get_template(template_name)
        # TODO: Implement proper chat templates
        prompt = self._format_chat_messages(messages)
        return await self.generate_text(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens - TensorRT-LLM doesn't expose tokenizer directly"""
        # For now, use estimation - could be improved by loading tokenizer separately
        return self._estimate_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get TensorRT-LLM model information"""
        info = {
            "provider": "tensorrt-llm",
            "engine_dir": getattr(self.config, 'engine_dir', None),
            "tokenizer_path": self.tokenizer_path,
            "loaded": self._model_loaded
        }
        
        if self._model_loaded:
            try:
                engine_path = Path(self.config.engine_dir)
                if engine_path.exists():
                    # Look for config files that might contain model info
                    config_files = list(engine_path.glob("*.json"))
                    info["config_files"] = [str(f.name) for f in config_files]
                    
                    # Try to read basic engine info
                    engine_files = list(engine_path.glob("*.engine"))
                    info["engine_files"] = [str(f.name) for f in engine_files]
                    
            except Exception as e:
                self.logger.warning(f"Could not get detailed engine info: {e}")
        
        return info