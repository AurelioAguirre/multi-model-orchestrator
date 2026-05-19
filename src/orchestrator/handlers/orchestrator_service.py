import os
import logging
from typing import Dict, Any, Optional
from enum import Enum

from src.shared.config import AppConfig
from src.orchestrator.clients.http_providers import (
    TransformersHTTPProvider, 
    VLLMHTTPProvider, 
    TensorRTHTTPProvider
)


class ProviderType(str, Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TENSORRT = "tensorrt"


class OrchestratorService:
    """Service that orchestrates requests to different model providers"""
    
    def __init__(self):
        self.config = AppConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize HTTP providers for microservices
        self.providers = {}
        self._initialize_providers()
        
        # Current active provider
        self.active_provider = None
        self.active_provider_type = None
        
        self.logger.info("OrchestratorService initialized")
    
    def _initialize_providers(self):
        """Initialize HTTP providers for each microservice"""
        # Get service URLs from environment or use defaults
        transformers_url = os.getenv('TRANSFORMERS_SERVICE_URL', 'http://transformers-service:8021')
        vllm_url = os.getenv('VLLM_SERVICE_URL', 'http://vllm-service:8022')
        tensorrt_url = os.getenv('TENSORRT_SERVICE_URL', 'http://tensorrt-service:8023')
        
        self.providers = {
            ProviderType.TRANSFORMERS: TransformersHTTPProvider(transformers_url, self.config),
            ProviderType.VLLM: VLLMHTTPProvider(vllm_url, self.config),
            ProviderType.TENSORRT: TensorRTHTTPProvider(tensorrt_url, self.config)
        }
        
        self.logger.info(f"Initialized providers: {list(self.providers.keys())}")
    
    async def load_model_with_provider(self, provider_type: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Load a model with specified provider"""
        try:
            provider_enum = ProviderType(provider_type)
            provider = self.providers.get(provider_enum)
            
            if not provider:
                raise ValueError(f"Unknown provider type: {provider_type}")
            
            self.logger.info(f"Loading model with {provider_type} provider")

            # Load the model via HTTP service, passing model_name if provided
            await provider.load_model(model_path=model_name)

            # Set as active provider
            self.active_provider = provider
            self.active_provider_type = provider_enum
            
            model_info = provider.get_model_info()
            model_info.update({
                'model_name': model_name or self.config.default_model,
                'provider_type': provider_type,
                'status': 'loaded'
            })
            
            self.logger.info(f"Model loaded successfully with {provider_type} provider")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load model with {provider_type} provider: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def unload_current_model(self) -> None:
        """Unload the currently loaded model"""
        try:
            if self.active_provider:
                await self.active_provider.unload_model()
                self.logger.info(f"Model unloaded from {self.active_provider_type.value} provider")
            
            self.active_provider = None
            self.active_provider_type = None
            
        except Exception as e:
            self.logger.error(f"Failed to unload model: {e}")
            raise RuntimeError(f"Model unloading failed: {e}")
    
    def get_active_provider(self):
        """Get the currently active provider"""
        if not self.active_provider:
            raise RuntimeError("No model loaded. Load a model first.")
        return self.active_provider
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status including active provider info"""
        status = {
            "active_provider": self.active_provider_type.value if self.active_provider_type else None,
            "model_loaded": self.active_provider._model_loaded if self.active_provider else False,
            "available_providers": list(self.providers.keys()),
            "providers_status": {}
        }

        # Add individual provider status (without actually calling them to avoid timeouts)
        for provider_type, provider in self.providers.items():
            status["providers_status"][provider_type.value] = {
                "service_url": provider.service_url,
                "loaded": provider._model_loaded
            }
        
        return status
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the active provider"""
        if not self.active_provider:
            return {
                "active_provider": None,
                "model_loaded": False,
                "provider_info": None
            }

        return {
            "active_provider": self.active_provider_type.value,
            "model_loaded": self.active_provider._model_loaded,
            "provider_info": self.active_provider.get_model_info()
        }
    
    async def close(self):
        """Clean up resources"""
        try:
            # Close all HTTP providers
            for provider in self.providers.values():
                if hasattr(provider, 'close'):
                    await provider.close()
            
            self.logger.info("OrchestratorService closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing OrchestratorService: {e}")


# Singleton instance
_orchestrator_instance = None

def get_orchestrator() -> OrchestratorService:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorService()
    return _orchestrator_instance