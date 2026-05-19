#!/usr/bin/env python3
"""
Transformers Service Routes
API endpoints for HuggingFace Transformers inference service
"""

import logging
from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any

from src.shared.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    HealthResponse, ModelLoadRequest, ModelLoadResponse
)
from src.shared.transformers_provider import TransformersProvider

logger = logging.getLogger("TransformersRoutes")

# Create router
router = APIRouter()

# Global provider instance (will be set by main.py)
provider: TransformersProvider = None

def get_provider() -> TransformersProvider:
    """Dependency to get the global provider instance"""
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return provider

def set_provider(transformers_provider: TransformersProvider):
    """Set the global provider instance"""
    global provider
    provider = transformers_provider

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Transformers service is running",
        timestamp=int(__import__("time").time())
    )

@router.post("/load_model", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest, provider: TransformersProvider = Depends(get_provider)):
    """Load Transformers model"""
    try:
        # Pass model_name from request if provided, otherwise use config default
        model_path = request.model_name if request.model_name else None
        await provider.load_model(model_path=model_path)
        return ModelLoadResponse(
            status="success",
            message="Transformers model loaded successfully",
            model_info=provider.get_model_info()
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model loading failed: {str(e)}"
        )

@router.post("/unload_model")
async def unload_model(provider: TransformersProvider = Depends(get_provider)):
    """Unload Transformers model"""
    try:
        await provider.unload_model()
        return {"success": True, "message": "Transformers model unloaded successfully"}
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model unloading failed: {str(e)}"
        )

@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest, provider: TransformersProvider = Depends(get_provider)):
    """Generate text completion"""
    try:
        if not provider._model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load model first."
            )

        generated_text = await provider.generate_text(
            prompt=request.prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            stop_sequences=request.stop
        )

        return CompletionResponse(
            id=f"cmpl-{__import__('uuid').uuid4().hex[:12]}",
            object="text_completion",
            created=int(__import__("time").time()),
            model=request.model,
            choices=[{
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }]
        )

    except Exception as e:
        logger.error(f"Completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest, provider: TransformersProvider = Depends(get_provider)):
    """Generate chat completion"""
    try:
        if not provider._model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load model first."
            )

        generated_text = await provider.generate_chat(
            messages=request.messages,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            stop_sequences=request.stop
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{__import__('uuid').uuid4().hex[:12]}",
            object="chat.completion",
            created=int(__import__("time").time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }]
        )

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}"
        )

@router.get("/generate")
async def generate(prompt: str, max_tokens: int = 100, provider: TransformersProvider = Depends(get_provider)):
    """Simple generation endpoint for testing"""
    try:
        if not provider._model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please load model first."
            )

        generated_text = await provider.generate_text(
            prompt=prompt,
            max_new_tokens=max_tokens
        )

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "model": provider.config.model_path
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text generation failed: {str(e)}"
        )
