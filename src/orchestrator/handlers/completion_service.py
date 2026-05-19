import time
import logging
from typing import Dict, Any, AsyncGenerator
from enum import Enum

from src.shared.schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    CompletionRequest, CompletionResponse,
    ChatCompletionStreamResponse, CompletionStreamResponse,
    StreamChoice, StreamCompletionChoice,
    Choice, CompletionChoice, ChatMessage, Usage
)


class CompletionService:
    """Service for handling completion and chat completion requests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _generate_completion_id(self, prefix: str = "cmpl") -> str:
        """Generate unique completion ID"""
        import uuid
        return f"{prefix}-{uuid.uuid4().hex[:24]}"
    
    def _get_current_timestamp(self) -> int:
        """Get current Unix timestamp"""
        return int(time.time())
    
    def _create_usage(self, prompt_text: str, generated_text: str, prompt_tokens: int = None, completion_tokens: int = None) -> Usage:
        """Create usage statistics"""
        if prompt_tokens is None:
            prompt_tokens = len(prompt_text.split()) * 1.3  # Rough estimate
        if completion_tokens is None:
            completion_tokens = len(generated_text.split()) * 1.3  # Rough estimate
        
        return Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens)
        )
    
    async def chat_completions(self, request: ChatCompletionRequest, provider) -> ChatCompletionResponse:
        """Handle chat completions request"""
        try:
            self.logger.info(f"Processing chat completion with {provider.engine_name if hasattr(provider, 'engine_name') else 'provider'}")
            
            # Ensure model is loaded
            if not provider.model_loaded:
                raise RuntimeError("No model loaded. Load a model first.")
            
            # Generate response using the provider
            sampling_params = provider.create_sampling_params(request) if hasattr(provider, 'create_sampling_params') else {}
            generated_text = await provider.generate_chat(
                request.messages,
                **sampling_params
            )
            
            # Create response choice
            choice = Choice(
                index=0,
                message=Message(
                    role="assistant",
                    content=generated_text
                ),
                finish_reason="stop"
            )
            
            # Calculate token usage
            prompt_text = provider.format_chat_prompt(request.messages) if hasattr(provider, 'format_chat_prompt') else str(request.messages)
            prompt_tokens = provider.count_tokens(prompt_text) if hasattr(provider, 'count_tokens') else None
            completion_tokens = provider.count_tokens(generated_text) if hasattr(provider, 'count_tokens') else None
            usage = self._create_usage(prompt_text, generated_text, prompt_tokens, completion_tokens)
            
            return ChatCompletionResponse(
                id=self._generate_completion_id("chatcmpl"),
                created=self._get_current_timestamp(),
                model=request.model,
                choices=[choice],
                usage=usage
            )
            
        except Exception as e:
            self.logger.error(f"Chat completion failed: {e}")
            raise RuntimeError(f"Chat completion failed: {e}")
    
    async def completions(self, request: CompletionRequest, provider) -> CompletionResponse:
        """Handle text completions request"""
        try:
            self.logger.info(f"Processing completion with {provider.engine_name if hasattr(provider, 'engine_name') else 'provider'}")
            
            # Ensure model is loaded
            if not provider.model_loaded:
                raise RuntimeError("No model loaded. Load a model first.")
            
            # Handle single prompt or list of prompts
            prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
            
            choices = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            # Generate for each prompt
            sampling_params = provider.create_sampling_params(request) if hasattr(provider, 'create_sampling_params') else {}
            
            for i, prompt in enumerate(prompts):
                generated_text = await provider.generate_text(
                    prompt,
                    **sampling_params
                )
                
                choice = CompletionChoice(
                    index=i,
                    text=generated_text,
                    finish_reason="stop"
                )
                choices.append(choice)
                
                # Count tokens
                prompt_tokens = provider.count_tokens(prompt) if hasattr(provider, 'count_tokens') else len(prompt.split()) * 1.3
                completion_tokens = provider.count_tokens(generated_text) if hasattr(provider, 'count_tokens') else len(generated_text.split()) * 1.3
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
            
            usage = Usage(
                prompt_tokens=int(total_prompt_tokens),
                completion_tokens=int(total_completion_tokens),
                total_tokens=int(total_prompt_tokens + total_completion_tokens)
            )
            
            return CompletionResponse(
                id=self._generate_completion_id("cmpl"),
                created=self._get_current_timestamp(),
                model=request.model,
                choices=choices,
                usage=usage
            )
            
        except Exception as e:
            self.logger.error(f"Text completion failed: {e}")
            raise RuntimeError(f"Text completion failed: {e}")
    
    async def chat_completions_stream(self, request: ChatCompletionRequest, provider) -> AsyncGenerator[str, None]:
        """Handle streaming chat completions"""
        try:
            # Check if provider supports streaming
            if hasattr(provider, 'generate_text_stream'):
                prompt_text = provider.format_chat_prompt(request.messages) if hasattr(provider, 'format_chat_prompt') else str(request.messages)
                sampling_params = provider.create_sampling_params(request) if hasattr(provider, 'create_sampling_params') else {}
                
                completion_id = self._generate_completion_id("chatcmpl")
                timestamp = self._get_current_timestamp()
                
                async for text_chunk in provider.generate_text_stream(prompt_text, **sampling_params):
                    stream_response = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=timestamp,
                        model=request.model,
                        choices=[
                            StreamChoice(
                                index=0,
                                delta={"content": text_chunk},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {stream_response.model_dump_json()}\n\n"
                
                # Send final chunk
                final_response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=timestamp,
                    model=request.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop"
                        )
                    ]
                )
                yield f"data: {final_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            
            else:
                # Fallback to non-streaming
                response = await self.chat_completions(request, provider)
                
                stream_response = ChatCompletionStreamResponse(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={"content": response.choices[0].message.content},
                            finish_reason="stop"
                        )
                    ]
                )
                
                yield f"data: {stream_response.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                
        except Exception as e:
            self.logger.error(f"Streaming chat completion failed: {e}")
            raise RuntimeError(f"Streaming failed: {e}")
    
    async def completions_stream(self, request: CompletionRequest, provider) -> AsyncGenerator[str, None]:
        """Handle streaming text completions"""
        try:
            # Similar implementation to chat streaming
            # For now, implementing basic fallback
            response = await self.completions(request, provider)
            
            for choice in response.choices:
                stream_response = CompletionStreamResponse(
                    id=response.id,
                    created=response.created,
                    model=response.model,
                    choices=[
                        StreamCompletionChoice(
                            index=choice.index,
                            text=choice.text,
                            finish_reason="stop"
                        )
                    ]
                )
                
                yield f"data: {stream_response.model_dump_json()}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            self.logger.error(f"Streaming completion failed: {e}")
            raise RuntimeError(f"Streaming failed: {e}")