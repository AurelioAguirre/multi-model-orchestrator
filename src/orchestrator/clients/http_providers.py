import httpx
import logging
from typing import Dict, Any, List

from src.shared.base import BaseModelProvider
from src.shared.schemas import ChatMessage


class HTTPModelProvider(BaseModelProvider):
    """Base HTTP client for engine microservices"""
    
    def __init__(self, service_url: str, engine_name: str, config=None):
        # Don't call super().__init__ as we don't need full config for HTTP clients
        self.config = config
        self.service_url = service_url
        self.engine_name = engine_name
        self.logger = logging.getLogger(f"{__name__}.{engine_name}")
        self._model_loaded = False
        
        # HTTP client with longer timeout for ML operations
        self.http_client = httpx.AsyncClient(
            base_url=service_url,
            timeout=httpx.Timeout(300.0)  # 5 minutes for model loading/inference
        )
        
        self.logger.info(f"Initialized HTTP provider for {engine_name} at {service_url}")
    
    async def load_model(self, model_path: str = None) -> None:
        """Load model via HTTP service"""
        try:
            self.logger.info(f"Loading model via {self.engine_name} service")

            # Send proper JSON body with ModelLoadRequest schema
            payload = {
                "model_name": model_path,
                "config_overrides": None
            }

            response = await self.http_client.post("/load_model", json=payload)
            response.raise_for_status()

            result = response.json()
            if result.get("status") == "success" or result.get("success"):
                self._model_loaded = True
                self.logger.info(f"{self.engine_name} model loaded successfully")
            else:
                raise RuntimeError(f"Service load failed: {result.get('message', 'Unknown error')}")
                
        except httpx.RequestError as e:
            self.logger.error(f"Failed to connect to {self.engine_name} service: {e}")
            raise RuntimeError(f"Service connection failed: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load {self.engine_name} model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def unload_model(self) -> None:
        """Unload model via HTTP service"""
        try:
            response = await self.http_client.post("/unload_model")
            response.raise_for_status()
            
            result = response.json()
            if result.get("status") == "success":
                self._model_loaded = False
                self.logger.info(f"{self.engine_name} model unloaded successfully")
            else:
                self.logger.warning(f"Service unload warning: {result.get('message', 'Unknown')}")
                self._model_loaded = False
                
        except Exception as e:
            self.logger.error(f"Failed to unload {self.engine_name} model: {e}")
            self._model_loaded = False
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text via HTTP service"""
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": kwargs.get('max_new_tokens', 512),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50),
                "stop_sequences": kwargs.get('stop_sequences', [])
            }
            
            response = await self.http_client.post("/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("text", "")
            
            # Log performance info if available
            if "tokens_per_second" in result:
                self.logger.info(f"{self.engine_name} generation: {result['tokens_per_second']:.1f} tokens/sec")
            
            return generated_text.strip()
            
        except httpx.RequestError as e:
            self.logger.error(f"{self.engine_name} service request failed: {e}")
            raise RuntimeError(f"Service generation failed: {e}")
        except Exception as e:
            self.logger.error(f"{self.engine_name} generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {e}")
    
    async def generate_chat(self, messages: List[ChatMessage], **kwargs) -> str:
        """Generate response for chat messages via HTTP service"""
        try:
            payload = {
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "max_tokens": kwargs.get('max_new_tokens', 512),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50),
                "stop_sequences": kwargs.get('stop_sequences', [])
            }
            
            response = await self.http_client.post("/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result.get("text", "").strip()
            
        except Exception as e:
            self.logger.error(f"{self.engine_name} chat generation failed: {e}")
            raise RuntimeError(f"Chat generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (simplified for HTTP providers)"""
        # Simple estimation - could call service for exact count
        return len(text.split()) * 1.3  # Rough approximation
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information from service"""
        return {
            "provider": self.engine_name,
            "service_url": self.service_url,
            "loaded": self._model_loaded,
            "type": "http_client"
        }
    
    async def close(self):
        """Clean up HTTP client"""
        await self.http_client.aclose()


class TransformersHTTPProvider(HTTPModelProvider):
    """HTTP provider for Transformers microservice"""
    
    def __init__(self, service_url: str, config=None):
        super().__init__(service_url, "transformers", config)


class VLLMHTTPProvider(HTTPModelProvider):
    """HTTP provider for vLLM microservice"""
    
    def __init__(self, service_url: str, config=None):
        super().__init__(service_url, "vllm", config)


class TensorRTHTTPProvider(HTTPModelProvider):
    """HTTP provider for TensorRT-LLM microservice"""
    
    def __init__(self, service_url: str, config=None):
        super().__init__(service_url, "tensorrt", config)