from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None
    
    @validator('stop')
    def validate_stop(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("stop sequences cannot exceed 4 items")
        return v


class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, ge=1)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Usage


class StreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class StreamCompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[StreamCompletionChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tensorrt-llm"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: int


class GPUStatus(BaseModel):
    gpu_id: int
    name: str
    memory_used: int  # MB
    memory_total: int  # MB
    memory_free: int  # MB
    utilization: float  # percentage
    temperature: Optional[int] = None  # Celsius


class SystemStatus(BaseModel):
    status: str
    gpus: List[GPUStatus]
    model_loaded: bool
    model_name: Optional[str] = None
    timestamp: int


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


# Microservice schemas
class ModelLoadRequest(BaseModel):
    model_name: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None


class ModelLoadResponse(BaseModel):
    status: str
    message: str
    model_info: Optional[Dict[str, Any]] = None