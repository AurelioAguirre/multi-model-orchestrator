import asyncio, os
from typing import Dict, Any, List
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

from .base import BaseModelProvider
# from src.api.utils import SharedUtils, ChatTemplates  # TODO: Implement these utilities
from src.shared.schemas import ChatMessage


class TransformersProvider(BaseModelProvider):
    """Model provider using HuggingFace Transformers (PyTorch)"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Debug GPU detection
        if torch.cuda.is_available():
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            self.logger.info(f"GPU count: {self.num_gpus}")
            for i in range(self.num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            self.logger.info("CUDA not available - using CPU")
        
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not available")
    
    def _format_chat_messages(self, messages: List[ChatMessage]) -> str:
        """Simple chat message formatting (fallback for missing ChatTemplates)"""
        formatted_parts = []
        for message in messages:
            role = message.role.value if hasattr(message.role, 'value') else str(message.role)
            formatted_parts.append(f"{role.title()}: {message.content}")
        return "\n".join(formatted_parts) + "\nAssistant:"
    
    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation (fallback for missing SharedUtils)"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        # Rough estimation: ~4 characters per token for most models
        return len(text) // 4
    
    def _find_free_port(self) -> int:
        """Find a free port for distributed coordination"""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 0))
            return s.getsockname()[1]
    
    def _init_distributed(self) -> bool:
        """Initialize torch.distributed for tensor parallelism"""
        if self.num_gpus <= 1 or torch.distributed.is_initialized():
            return False
        
        try:
            # Set environment variables for distributed processing
            master_port = self._find_free_port()
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(master_port)
            os.environ['WORLD_SIZE'] = str(self.num_gpus)
            os.environ['RANK'] = '0'
            
            # Initialize the default distributed process group
            torch.distributed.init_process_group(backend='nccl')
            
            self.logger.info(f"Initialized distributed training with {self.num_gpus} GPUs")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize distributed processing: {e}")
            return False
    
    async def load_model(self, model_path: str = None) -> None:
        """Load HuggingFace Transformers model"""
        try:
            # Use provided model_path or fall back to config default
            model_to_load = model_path or self.config.model_path
            self.logger.info(f"Loading Transformers model: {model_to_load}")
            
            # Initialize distributed processing if multi-GPU
            use_distributed = self._init_distributed()
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained,
                model_to_load,
                local_files_only=False
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate device mapping
            self.logger.info("Loading model...")
            if self.num_gpus > 1 and not use_distributed:
                # Use device_map for multi-GPU without distributed
                self.logger.info(f"Loading model with device_map='auto' for {self.num_gpus} GPUs")
                self.model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_to_load,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=False
                )
            elif self.device == "cuda":
                # Single GPU
                self.logger.info(f"Loading model on single GPU: {self.device}")
                self.model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_to_load,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=False
                )
                self.model = self.model.to(self.device)
            else:
                # CPU
                self.logger.info("Loading model on CPU")
                self.model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    model_to_load,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    local_files_only=False
                )
            
            # Create pipeline for easier generation
            self.logger.info("Creating generation pipeline...")
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and self.num_gpus == 1 else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            self._model_loaded = True
            
            # Log model info
            if hasattr(self.model, 'config'):
                model_config = self.model.config
                self.logger.info(f"Model architecture: {getattr(model_config, 'architectures', 'Unknown')}")
                self.logger.info(f"Model size: ~{self.model.num_parameters() / 1e9:.1f}B parameters")
                self.logger.info(f"Vocab size: {getattr(model_config, 'vocab_size', 'Unknown')}")
            
            self.logger.info("Transformers model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Transformers model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def unload_model(self) -> None:
        """Unload Transformers model to free memory"""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if self.model:
                del self.model
                self.model = None
            
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Cleanup distributed if initialized
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
            
            self._model_loaded = False
            self.logger.info("Transformers model unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error unloading Transformers model: {e}")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using Transformers"""
        if not self._model_loaded:
            await self.load_model()
        
        try:
            # Prepare generation parameters
            generation_kwargs = {
                "max_new_tokens": kwargs.get('max_new_tokens', self.config.default_max_tokens),
                "temperature": kwargs.get('temperature', self.config.default_temperature),
                "top_p": kwargs.get('top_p', self.config.default_top_p),
                "do_sample": kwargs.get('temperature', self.config.default_temperature) > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
            }
            
            # Handle stop sequences
            stop_sequences = kwargs.get('stop_sequences', [])
            
            # Generate using pipeline
            results = await asyncio.to_thread(
                self.pipeline,
                prompt,
                **generation_kwargs
            )
            
            generated_text = results[0]["generated_text"]
            
            # Apply stop sequences if provided
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Transformers text generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages using Transformers"""
        # Use chat template or fallback to simple formatting
        template_name = kwargs.get('chat_template', 'default')
        # template_func = ChatTemplates.get_template(template_name)
        # TODO: Implement proper chat templates
        prompt = self._format_chat_messages(messages)
        return await self.generate_text(prompt, **kwargs)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using the loaded tokenizer"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback estimation
            return self._estimate_tokens(text)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Transformers model information"""
        info = {
            "provider": "transformers",
            "model_path": self.config.model_path,
            "device": self.device,
            "num_gpus": self.num_gpus,
            "loaded": self._model_loaded
        }
        
        if self._model_loaded and self.model:
            try:
                config = self.model.config
                info.update({
                    "architecture": getattr(config, 'architectures', ['Unknown'])[0],
                    "parameters": f"~{self.model.num_parameters() / 1e9:.1f}B",
                    "vocab_size": getattr(config, 'vocab_size', 'Unknown'),
                    "max_position_embeddings": getattr(config, 'max_position_embeddings', 'Unknown')
                })
            except Exception as e:
                self.logger.warning(f"Could not get detailed model info: {e}")
        
        return info