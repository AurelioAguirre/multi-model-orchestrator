import asyncio
from typing import Dict, Any, List, AsyncGenerator
import logging
from pathlib import Path

try:
    from vllm import LLM, SamplingParams as VLLMSamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from .base import BaseModelProvider
# from src.api.utils import SharedUtils, ChatTemplates  # TODO: Implement these utilities
from src.shared.schemas import ChatMessage


class VLLMProvider(BaseModelProvider):
    """Model provider using vLLM for high-performance inference"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.async_engine = None
        
        if not _VLLM_AVAILABLE:
            raise ImportError("vLLM not available")
    
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
        """Load model using vLLM"""
        try:
            # Use provided model_path or fall back to config default
            model_to_load = model_path or self.config.model_path
            self.logger.info(f"Loading vLLM model: {model_to_load}")

            # Determine tensor parallel size based on available GPUs
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            tensor_parallel_size = min(num_gpus, getattr(self.config, 'world_size', 1))

            self.logger.info(f"Using tensor_parallel_size: {tensor_parallel_size}")

            # Load model with vLLM
            self.llm = await asyncio.to_thread(
                LLM,
                model=str(Path(model_to_load).resolve()),
                tokenizer=getattr(self.config, 'tokenizer_path', None) or model_to_load,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=getattr(self.config, 'gpu_memory_fraction', 0.9),
                max_num_batched_tokens=getattr(self.config, 'max_num_batched_tokens', 4096),
                enable_prefix_caching=getattr(self.config, 'enable_prefix_caching', True),
                trust_remote_code=True,
            )
            
            self._model_loaded = True

            # Log model info
            self.logger.info("vLLM model loaded successfully")
            self.logger.info(f"Model: {model_to_load}")
            self.logger.info(f"Tensor parallel size: {tensor_parallel_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to load vLLM model: {e}")
            raise RuntimeError(f"vLLM model loading failed: {e}")
    
    async def unload_model(self) -> None:
        """Unload vLLM model to free memory"""
        try:
            if self.async_engine:
                # Stop async engine if running
                del self.async_engine
                self.async_engine = None
            
            if self.llm:
                del self.llm
                self.llm = None
            
            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._model_loaded = False
            self.logger.info("vLLM model unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error unloading vLLM model: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM"""
        if not self._model_loaded:
            await self.load_model()
        
        try:
            # Create sampling parameters
            sampling_params = VLLMSamplingParams(
                temperature=kwargs.get('temperature', self.config.default_temperature),
                top_p=kwargs.get('top_p', self.config.default_top_p),
                top_k=kwargs.get('top_k', self.config.default_top_k),
                max_tokens=kwargs.get('max_new_tokens', self.config.default_max_tokens),
                stop=kwargs.get('stop_sequences', []) or None,
            )
            
            # Generate using vLLM
            outputs = await asyncio.to_thread(
                self.llm.generate,
                [prompt],
                sampling_params
            )
            
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("No output generated")
            
            generated_text = outputs[0].outputs[0].text
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"vLLM text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages using vLLM"""
        # Use chat template or fallback to simple formatting
        # template_name = kwargs.get('chat_template', 'default')
        # template_func = ChatTemplates.get_template(template_name)
        # TODO: Implement proper chat templates
        
        prompt = self._format_chat_messages(messages)
        return await self.generate_text(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using vLLM tokenizer"""
        if self.llm and hasattr(self.llm, 'get_tokenizer'):
            tokenizer = self.llm.get_tokenizer()
            if hasattr(tokenizer, 'encode'):
                return len(tokenizer.encode(text))
        
        # Fallback estimation
        return self._estimate_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information"""
        info = {
            "provider": "vllm",
            "model_path": self.config.model_path,
            "loaded": self._model_loaded
        }
        
        if self._model_loaded and self.llm:
            try:
                # Get vLLM-specific information
                import torch
                info.update({
                    "tensor_parallel_size": min(torch.cuda.device_count(), getattr(self.config, 'world_size', 1)),
                    "gpu_memory_utilization": getattr(self.config, 'gpu_memory_fraction', 0.9),
                    "enable_prefix_caching": getattr(self.config, 'enable_prefix_caching', True)
                })
            except Exception as e:
                self.logger.warning(f"Could not get detailed vLLM info: {e}")
        
        return info


class VLLMAsyncProvider(BaseModelProvider):
    """Async vLLM provider for streaming support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.async_engine = None
        
        if not _VLLM_AVAILABLE:
            raise ImportError("vLLM not available")
    
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
        """Load model using vLLM AsyncEngine"""
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine

            # Use provided model_path or fall back to config default
            model_to_load = model_path or self.config.model_path
            self.logger.info(f"Loading vLLM async model: {model_to_load}")

            # Determine tensor parallel size
            import torch
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            tensor_parallel_size = min(num_gpus, getattr(self.config, 'world_size', 1))

            # Create engine args
            engine_args = AsyncEngineArgs(
                model=str(Path(model_to_load).resolve()),
                tokenizer=getattr(self.config, 'tokenizer_path', None) or model_to_load,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=getattr(self.config, 'gpu_memory_fraction', 0.9),
                max_num_batched_tokens=getattr(self.config, 'max_num_batched_tokens', 4096),
                enable_prefix_caching=getattr(self.config, 'enable_prefix_caching', True),
                trust_remote_code=True,
            )
            
            # Create async engine
            self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            self._model_loaded = True
            
            self.logger.info("vLLM async model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load vLLM async model: {e}")
            raise RuntimeError(f"vLLM async model loading failed: {e}")
    
    async def unload_model(self) -> None:
        """Unload vLLM async model"""
        try:
            if self.async_engine:
                del self.async_engine
                self.async_engine = None
            
            # Clear CUDA cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._model_loaded = False
            self.logger.info("vLLM async model unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error unloading vLLM async model: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM async engine"""
        if not self._model_loaded:
            await self.load_model()
        
        try:
            # Create sampling parameters
            sampling_params = VLLMSamplingParams(
                temperature=kwargs.get('temperature', self.config.default_temperature),
                top_p=kwargs.get('top_p', self.config.default_top_p),
                top_k=kwargs.get('top_k', self.config.default_top_k),
                max_tokens=kwargs.get('max_new_tokens', self.config.default_max_tokens),
                stop=kwargs.get('stop_sequences', []) or None,
            )
            
            # Generate using async engine
            request_id = f"req_{asyncio.current_task().get_name()}_{id(prompt)}"
            
            # Add request to engine
            await self.async_engine.add_request(request_id, prompt, sampling_params)
            
            # Get results
            final_output = None
            async for request_output in self.async_engine.generate():
                if request_output.request_id == request_id:
                    final_output = request_output
                    break
            
            if not final_output or not final_output.outputs:
                raise RuntimeError("No output generated")
            
            generated_text = final_output.outputs[0].text
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"vLLM async text generation failed: {e}")
            raise RuntimeError(f"Async text generation failed: {e}")
    
    async def generate_text_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming text using vLLM async engine"""
        if not self._model_loaded:
            await self.load_model()
        
        try:
            # Create sampling parameters
            sampling_params = VLLMSamplingParams(
                temperature=kwargs.get('temperature', self.config.default_temperature),
                top_p=kwargs.get('top_p', self.config.default_top_p),
                top_k=kwargs.get('top_k', self.config.default_top_k),
                max_tokens=kwargs.get('max_new_tokens', self.config.default_max_tokens),
                stop=kwargs.get('stop_sequences', []) or None,
            )
            
            # Generate streaming
            request_id = f"stream_{asyncio.current_task().get_name()}_{id(prompt)}"
            await self.async_engine.add_request(request_id, prompt, sampling_params)
            
            previous_text = ""
            async for request_output in self.async_engine.generate():
                if request_output.request_id == request_id:
                    if request_output.outputs:
                        current_text = request_output.outputs[0].text
                        # Yield only the new part
                        new_text = current_text[len(previous_text):]
                        if new_text:
                            yield new_text
                        previous_text = current_text
                    
                    if request_output.finished:
                        break
                        
        except Exception as e:
            self.logger.error(f"vLLM streaming generation failed: {e}")
            raise RuntimeError(f"Streaming generation failed: {e}")
    
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages using vLLM async"""
        template_name = kwargs.get('chat_template', 'default')
        # template_func = ChatTemplates.get_template(template_name)
        # TODO: Implement proper chat templates
        prompt = self._format_chat_messages(messages)
        return await self.generate_text(prompt, **kwargs)
    
    async def generate_chat_stream(self, messages: List[ChatMessage], **kwargs):
        """Generate streaming response for chat messages"""
        template_name = kwargs.get('chat_template', 'default')
        # template_func = ChatTemplates.get_template(template_name)
        # TODO: Implement proper chat templates
        prompt = self._format_chat_messages(messages)
        async for chunk in self.generate_text_stream(prompt, **kwargs):
            yield chunk
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for async engine"""
        return self._estimate_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM async model information"""
        return {
            "provider": "vllm-async",
            "model_path": self.config.model_path,
            "loaded": self._model_loaded,
            "supports_streaming": True
        }