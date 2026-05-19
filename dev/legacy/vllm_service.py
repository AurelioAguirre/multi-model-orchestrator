#!/usr/bin/env python3
"""
vLLM Microservice
FastAPI service for vLLM inference
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from src.utils.config import load_config
from src.utils.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    HealthResponse, ModelLoadRequest, ModelLoadResponse
)
from src.providers.vllm_provider import VLLMAsyncProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VLLMService")

# FastAPI app
app = FastAPI(
    title="vLLM Microservice",
    description="vLLM high-performance inference service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = None
provider = None


@app.on_event("startup")
async def startup_event():
    """Initialize the service"""
    global config, provider
    
    try:
        # Load configuration
        config_path = project_root / "resources" / "config.yml"
        config = load_config(str(config_path))
        
        # Initialize provider (but don't load model yet)
        provider = VLLMAsyncProvider(config)
        
        logger.info("vLLM service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="vLLM service is running",
        timestamp=int(__import__("time").time())
    )


@app.post("/load_model", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest):
    """Load a model"""
    try:
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        # Update model path if provided
        if request.model_name:
            config.model_path = request.model_name
        
        await provider.load_model()
        
        return ModelLoadResponse(
            status="success",
            message="Model loaded successfully",
            model_info=provider.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/unload_model")
async def unload_model():
    """Unload the current model"""
    try:
        if provider:
            await provider.unload_model()
        
        return {
            "status": "success",
            "message": "Model unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )


@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    try:
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        return provider.get_model_info()
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Generate chat completions"""
    try:
        if not provider or not provider._model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded"
            )
        
        if request.stream:
            # Streaming response
            async def stream_generator():
                try:
                    async for chunk in provider.generate_chat_stream(
                        messages=request.messages,
                        temperature=request.temperature,
                        max_new_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stop_sequences=request.stop
                    ):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            response_text = await provider.generate_chat(
                messages=request.messages,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=request.stop
            )
            
            # Count tokens
            prompt_text = " ".join([msg.content for msg in request.messages])
            prompt_tokens = provider.count_tokens(prompt_text)
            completion_tokens = provider.count_tokens(response_text)
            
            # Create response
            from src.utils.schemas import ChatMessage, Role, Choice, Usage
            import time
            import uuid
            
            choice = Choice(
                index=0,
                message=ChatMessage(role=Role.ASSISTANT, content=response_text),
                finish_reason="stop"
            )
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            response = ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
            return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Generate text completions"""
    try:
        if not provider or not provider._model_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No model loaded"
            )
        
        if request.stream:
            # Streaming response
            async def stream_generator():
                try:
                    async for chunk in provider.generate_text_stream(
                        prompt=request.prompt,
                        temperature=request.temperature,
                        max_new_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stop_sequences=request.stop
                    ):
                        yield f"data: {chunk}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/plain"
            )
        else:
            # Non-streaming response
            response_text = await provider.generate_text(
                prompt=request.prompt,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                stop_sequences=request.stop
            )
            
            # Count tokens
            prompt_tokens = provider.count_tokens(request.prompt)
            completion_tokens = provider.count_tokens(response_text)
            
            # Create response
            from src.utils.schemas import CompletionChoice, Usage
            import time
            import uuid
            
            choice = CompletionChoice(
                index=0,
                text=response_text,
                finish_reason="stop"
            )
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            
            response = CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:8]}",
                object="text_completion",
                created=int(time.time()),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
            return response
        
    except Exception as e:
        logger.error(f"Text completion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text completion failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment or default
    port = int(os.getenv("VLLM_SERVICE_PORT", 8013))
    
    logger.info(f"Starting vLLM service on port {port}")
    
    uvicorn.run(
        "vllm_service:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )