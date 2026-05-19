from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional, Dict, Any
import logging

from src.shared.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ChatMessage, Role
)


class BaseModelProvider(ABC):
    """Abstract base class for model providers - handles model loading and inference"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model_loaded = False
    
    @property
    def model_loaded(self) -> bool:
        return self._model_loaded
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the model to free memory"""
        pass
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        pass
    
    @abstractmethod
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    def format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to a single prompt string - can be overridden"""
        prompt_parts = []
        
        for message in messages:
            if message.role == Role.SYSTEM:
                prompt_parts.append(f"System: {message.content}")
            elif message.role == Role.USER:
                prompt_parts.append(f"User: {message.content}")
            elif message.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    def create_sampling_params(self, request) -> Dict[str, Any]:
        """Create sampling parameters from request"""
        return {
            "temperature": getattr(request, 'temperature', self.config.default_temperature),
            "top_p": getattr(request, 'top_p', self.config.default_top_p),
            "top_k": getattr(request, 'top_k', self.config.default_top_k),
            "max_new_tokens": getattr(request, 'max_tokens', self.config.default_max_tokens),
            "stop_sequences": getattr(request, 'stop', []) or [],
        }