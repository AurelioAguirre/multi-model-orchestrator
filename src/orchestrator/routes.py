
from fastapi import APIRouter, HTTPException, Depends, status, Query
from fastapi.responses import StreamingResponse
from typing import Optional
import time

from src.shared.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ModelsResponse, HealthResponse, SystemStatus,
    ErrorResponse, ChatMessage, Role
)
from src.orchestrator.handlers.orchestrator_service import get_orchestrator
from src.orchestrator.handlers.completion_service import CompletionService

router = APIRouter()

# Get singleton services
def get_completion_service() -> CompletionService:
    return CompletionService()

def get_orchestrator_service():
    return get_orchestrator()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="Server is running",
        timestamp=int(time.time())
    )


@router.get("/gpu", tags=["System"])
async def gpu_status(orchestrator = Depends(get_orchestrator_service)):
    """Get GPU status and system information"""
    try:
        return orchestrator.get_system_status()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get GPU status: {str(e)}"
        )


@router.post("/test", tags=["Testing"])
async def test_endpoint(
    orchestrator = Depends(get_orchestrator_service),
    completion_service: CompletionService = Depends(get_completion_service)
):
    """Test model inference with a predefined question and performance metrics"""
    try:
        # Get active provider
        provider = orchestrator.get_active_provider()
        config = orchestrator.config
        test_question = config.test_question
        
        # Create a simple chat completion request with the test question
        test_request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=Role.USER, content=test_question)],
            max_tokens=100,
            temperature=0.7
        )
        
        # Performance timing
        start_time = time.time()
        start_timestamp = int(start_time)
        
        # Generate response using completion service
        response = await completion_service.chat_completions(test_request, provider)
        
        # Calculate performance metrics
        end_time = time.time()
        end_timestamp = int(end_time)
        total_time = end_time - start_time
        
        # Extract response content and calculate tokens
        response_content = response.choices[0].message.content
        
        # Count tokens (rough estimate based on response)
        prompt_tokens = response.usage.prompt_tokens if response.usage else len(test_question.split()) * 1.3
        completion_tokens = response.usage.completion_tokens if response.usage else len(response_content.split()) * 1.3
        total_tokens = response.usage.total_tokens if response.usage else prompt_tokens + completion_tokens
        
        # Calculate tokens per second
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
        total_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        return {
            "status": "ok",
            "message": "Model inference test successful",
            "test_question": test_question,
            "response": response_content,
            "model_info": orchestrator.get_provider_info(),
            
            # Performance metrics
            "performance": {
                "total_time_seconds": round(total_time, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "total_tokens_per_second": round(total_tokens_per_second, 2),
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
            },
            
            # Timestamps
            "timing": {
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
                "end_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model test failed: {str(e)}"
        )


@router.get("/engine", tags=["System"])
async def engine_info(orchestrator = Depends(get_orchestrator_service)):
    """Get information about the current inference provider"""
    try:
        return orchestrator.get_provider_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get engine info: {str(e)}"
        )


@router.post("/load_transformers", tags=["Model Management"])
async def load_transformers(
    model_name: Optional[str] = Query(None, description="Model name/path to load"),
    orchestrator = Depends(get_orchestrator_service)
):
    """Load a model using Transformers provider"""
    try:
        result = await orchestrator.load_model_with_provider(
            provider_type="transformers",
            model_name=model_name
        )
        return {
            "status": "success",
            "message": "Model loaded with Transformers provider",
            "provider": "transformers",
            "model_info": result,
            "timestamp": int(time.time())
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load Transformers model: {str(e)}"
        )


@router.post("/load_vllm", tags=["Model Management"])
async def load_vllm(
    model_name: Optional[str] = Query(None, description="Model name/path to load"),
    orchestrator = Depends(get_orchestrator_service)
):
    """Load a model using vLLM provider"""
    try:
        result = await orchestrator.load_model_with_provider(
            provider_type="vllm",
            model_name=model_name
        )
        return {
            "status": "success",
            "message": "Model loaded with vLLM provider",
            "provider": "vllm",
            "model_info": result,
            "timestamp": int(time.time())
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load vLLM model: {str(e)}"
        )


@router.post("/load_tensorrt", tags=["Model Management"])
async def load_tensorrt(
    model_name: Optional[str] = Query(None, description="Model name/path to load"),
    orchestrator = Depends(get_orchestrator_service)
):
    """Load a model using TensorRT-LLM provider"""
    try:
        result = await orchestrator.load_model_with_provider(
            provider_type="tensorrt",
            model_name=model_name
        )
        return {
            "status": "success",
            "message": "Model loaded with TensorRT-LLM provider",
            "provider": "tensorrt",
            "model_info": result,
            "timestamp": int(time.time())
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load TensorRT model: {str(e)}"
        )


@router.post("/unload_model", tags=["Model Management"])
async def unload_model(orchestrator = Depends(get_orchestrator_service)):
    """Unload the currently loaded model"""
    try:
        await orchestrator.unload_current_model()
        return {
            "status": "success",
            "message": "Model unloaded successfully",
            "timestamp": int(time.time())
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.get("/v1/models", response_model=ModelsResponse, tags=["OpenAI Compatible"])
async def list_models(orchestrator = Depends(get_orchestrator_service)):
    """List available models (OpenAI compatible)"""
    try:
        # Create a simple models response based on active provider
        provider_info = orchestrator.get_provider_info()
        
        from src.shared.schemas import ModelInfo
        models = []
        
        if provider_info["active_provider"]:
            model_info = ModelInfo(
                id=f"{provider_info['active_provider']}-model",
                created=int(time.time()),
                owned_by=f"llm-tensor-server-{provider_info['active_provider']}"
            )
            models.append(model_info)
        
        return ModelsResponse(data=models)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(
    request: ChatCompletionRequest,
    orchestrator = Depends(get_orchestrator_service),
    completion_service: CompletionService = Depends(get_completion_service)
):
    """Generate chat completions (OpenAI compatible)"""
    try:
        provider = orchestrator.get_active_provider()
        
        if request.stream:
            return StreamingResponse(
                completion_service.chat_completions_stream(request, provider),
                media_type="text/plain"
            )
        else:
            response = await completion_service.chat_completions(request, provider)
            return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/v1/completions", tags=["OpenAI Compatible"])
async def completions(
    request: CompletionRequest,
    orchestrator = Depends(get_orchestrator_service),
    completion_service: CompletionService = Depends(get_completion_service)
):
    """Generate text completions (OpenAI compatible)"""
    try:
        provider = orchestrator.get_active_provider()
        
        if request.stream:
            return StreamingResponse(
                completion_service.completions_stream(request, provider),
                media_type="text/plain"
            )
        else:
            response = await completion_service.completions(request, provider)
            return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )