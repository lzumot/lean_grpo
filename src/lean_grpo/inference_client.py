"""API endpoint inference client for proof generation."""

import asyncio
from typing import Any, AsyncIterator, Optional

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient


class InferenceClient:
    """Client for inference API endpoints.
    
    This client supports both OpenAI-compatible endpoints and
    custom vLLM endpoints for proof generation.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize the inference client.
        
        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            model_name: Default model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or "dummy-key"
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create OpenAI-compatible client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=DefaultAsyncHttpxClient(
                timeout=httpx.Timeout(timeout=timeout, connect=5.0),
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
            ),
            max_retries=max_retries,
        )
    
    async def generate_tactic(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        stop: Optional[list[str]] = None,
        extra_body: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate a tactic given a conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            extra_body: Additional parameters for vLLM
            
        Returns:
            Dict with 'content', 'logprobs', 'tokens', etc.
        """
        model = model or self.model_name
        if not model:
            raise ValueError("Model name must be provided")
        
        # Prepare request
        request_body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        
        if stop:
            request_body["stop"] = stop
        
        # Add extra_body for vLLM-specific features like logprobs
        if extra_body:
            request_body.update(extra_body)
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = await self._client.chat.completions.create(
                    **request_body
                )
                
                choice = response.choices[0]
                content = choice.message.content or ""
                
                result = {
                    "content": content,
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
                
                # Extract logprobs if available
                if choice.logprobs and choice.logprobs.content:
                    result["logprobs"] = [
                        token_logprob.logprob
                        for token_logprob in choice.logprobs.content
                    ]
                    result["token_ids"] = [
                        token_logprob.token
                        for token_logprob in choice.logprobs.content
                    ]
                
                return result
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        raise RuntimeError("Max retries exceeded")
    
    async def generate_batch(
        self,
        prompts: list[list[dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        stop: Optional[list[str]] = None,
        extra_body: Optional[dict] = None,
        max_concurrent: int = 10,
    ) -> list[dict[str, Any]]:
        """Generate tactics for multiple prompts in parallel.
        
        Args:
            prompts: List of conversation histories
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            extra_body: Additional parameters
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of generation results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_limit(messages):
            async with semaphore:
                return await self.generate_tactic(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body=extra_body,
                )
        
        tasks = [generate_with_limit(msgs) for msgs in prompts]
        return await asyncio.gather(*tasks)
    
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
    ) -> AsyncIterator[str]:
        """Stream generate a tactic.
        
        Args:
            messages: List of message dicts
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Generated text chunks
        """
        model = model or self.model_name
        if not model:
            raise ValueError("Model name must be provided")
        
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def check_health(self) -> bool:
        """Check if the inference endpoint is healthy."""
        try:
            # Try to list models (OpenAI compatible)
            await self._client.models.list()
            return True
        except Exception:
            return False


class VLLMClient(InferenceClient):
    """Specialized client for vLLM endpoints with GRPO support."""
    
    async def generate_with_logprobs(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        top_logprobs: int = 20,
    ) -> dict[str, Any]:
        """Generate with detailed logprobs for GRPO.
        
        Args:
            messages: List of message dicts
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_logprobs: Number of top logprobs to return
            
        Returns:
            Generation result with logprobs
        """
        extra_body = {
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
        
        return await self.generate_tactic(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
    
    async def generate_group(
        self,
        messages: list[dict[str, str]],
        group_size: int,
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
    ) -> list[dict[str, Any]]:
        """Generate a group of responses for GRPO.
        
        Args:
            messages: The prompt messages
            group_size: Number of responses to generate
            model: Model name override
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            List of generation results
        """
        # Create prompts (same prompt repeated)
        prompts = [messages] * group_size
        
        # Generate in parallel
        return await self.generate_batch(
            prompts=prompts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=group_size,
        )


class MockInferenceClient(InferenceClient):
    """Mock client for testing without an actual API endpoint."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock client without making real connections."""
        self.base_url = "http://mock"
        self.api_key = "mock-key"
        self.model_name = "mock-model"
        self.timeout = 30.0
        self.max_retries = 0
        self._client = None
    
    async def generate_tactic(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        stop: Optional[list[str]] = None,
        extra_body: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate a mock tactic."""
        # Simple heuristic based on message content
        last_message = messages[-1]["content"] if messages else ""
        
        # Mock tactics based on context
        mock_tactics = [
            "intro h",
            "simp",
            "apply Nat.add_comm",
            "exact h",
            "rw [Nat.add_zero]",
            "cases n",
            "induction n",
        ]
        
        import random
        content = random.choice(mock_tactics)
        
        return {
            "content": content,
            "finish_reason": "stop",
            "prompt_tokens": len(str(messages).split()),
            "completion_tokens": len(content.split()),
            "logprobs": [-0.5] * len(content.split()),
        }
    
    async def check_health(self) -> bool:
        """Always healthy."""
        return True
