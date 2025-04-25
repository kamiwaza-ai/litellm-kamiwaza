# LiteLLM Router Integration for Kamiwaza AI

This package provides a custom router for [LiteLLM](https://github.com/BerriAI/litellm) that integrates with [Kamiwaza AI](https://kamiwaza.ai) model deployments. The `KamiwazaRouter` extends LiteLLM's `Router` class to enable efficient routing of requests to Kamiwaza-deployed models.

## Features

- **Dynamic Model Discovery**: Automatically discovers available models from Kamiwaza deployments
- **Multi-Instance Support**: Connect to multiple Kamiwaza instances simultaneously 
- **Caching**: Efficient caching of model lists with configurable TTL
- **Model Pattern Filtering**: Filter models based on name patterns (e.g., only use "qwen" or "gemma" models)
- **Static Model Configuration**: Support for static model configurations alongside Kamiwaza models
- **Fallback Routing**: Automatic fallback between models in case of failures

## Installation

```bash
pip install litellm-kamiwaza
```

For running the examples, you'll also need:

```bash
pip install python-dotenv
```

## Requirements

- Python 3.7+
- litellm>=1.0.0
- kamiwaza-client>=0.1.0

## Usage

### Basic Usage

```python
from litellm_kamiwaza import KamiwazaRouter

# Initialize router with automatic Kamiwaza discovery
router = KamiwazaRouter()

# Use the router like a standard litellm Router
response = router.completion(
    model="deployed-model-name",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

### Configuration Options

#### Environment Variables

- `KAMIWAZA_API_URL`: URL for the Kamiwaza API (e.g., "https://localhost/api")
- `KAMIWAZA_URL_LIST`: Comma-separated list of Kamiwaza URLs (e.g., "https://instance1/api,https://instance2/api")
- `KAMIWAZA_VERIFY_SSL`: Set to "true" to enable SSL verification (default: "false")

#### Router Configuration

```python
# Initialize with specific Kamiwaza URL
router = KamiwazaRouter(
    kamiwaza_api_url="https://my-kamiwaza-server.com/api",
    cache_ttl_seconds=600,  # Cache model list for 10 minutes
    model_pattern="72b",    # Only use models with "72b" in their name
)

# Initialize with multiple Kamiwaza instances
router = KamiwazaRouter(
    kamiwaza_uri_list="https://instance1.com/api,https://instance2.com/api",
    cache_ttl_seconds=300
)

# Initialize with static model list alongside Kamiwaza models
router = KamiwazaRouter(
    kamiwaza_api_url="https://my-kamiwaza-server.com/api",
    model_list=[
        {
            "model_name": "my-static-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "sk-your-api-key",
                "api_base": "https://api.openai.com/v1"
            },
            "model_info": {
                "id": "my-static-model",
                "provider": "static",
                "description": "Static model configuration"
            }
        }
    ]
)
```

### Pattern Matching Examples

You can filter models by name patterns:

```python
# Only use models with "qwen" in their name
router = KamiwazaRouter(
    kamiwaza_api_url="https://my-kamiwaza-server.com/api",
    model_pattern="qwen"
)

# Only use gemma models
router = KamiwazaRouter(
    kamiwaza_uri_list="https://instance1.com/api,https://instance2.com/api",
    model_pattern="gemma"
)

# Only use static models
router = KamiwazaRouter(
    model_pattern="static"
)
```

### Static Models Configuration

For more organized static model configurations, you can create a `static_models_conf.py` file in your project root:

```python
# static_models_conf.py
from typing import List, Dict, Any, Optional

def get_static_model_configs() -> List[Dict[str, Any]]:
    """Returns a list of statically defined model configurations."""
    return [
        {
            "model_name": "static-custom-model", 
            "litellm_params": {
                "model": "openai/model",
                "api_key": "your-api-key",
                "api_base": "https://your-endpoint.com/v1"
            },
            "model_info": {
                "id": "static-custom-model",
                "provider": "static",
                "description": "Static model configuration"
            }
        }
    ]
```

The `KamiwazaRouter` will automatically detect and use these static models.

## Automatic Model Selection

If no `model` is specified in the `completion` call, the router will automatically select an available model from its list.

### Example 1: Single Available Model

```python
# Assuming the router discovers only one compatible model
response = router.completion(
    messages=[{"role": "user", "content": "Hello, world!"}]
)

# Example log output (timestamps and some details may vary):
# 2025-04-25 06:58:38,079 - asyncio - INFO - Cache expired or not used. Fetching fresh model list (TTL: 300s).
# 2025-04-25 06:58:38,198 - asyncio - INFO - Successfully fetched and processed 1 models from https://my-kamiwaza-server.com/api
# 2025-04-25 06:58:38,198 - asyncio - INFO - No static model configurations returned or defined.
# 2025-04-25 06:58:38,198 - asyncio - INFO - Updated model cache. Total unique models: 1 (1 from Kamiwaza, 0 static).
# 06:58:38 - LiteLLM:INFO: utils.py:3085 - 
# LiteLLM completion() model= model; provider = openai
# 2025-04-25 06:58:38,204 - LiteLLM - INFO - 
# LiteLLM completion() model= model; provider = openai
# 2025-04-25 06:58:39,701 - httpx - INFO - HTTP Request: POST http://my-kamiwaza-server.com/v1/chat/completions "HTTP/1.1 200 OK"
# 06:58:39 - LiteLLM:INFO: utils.py:1177 - Wrapper: Completed Call, calling success_handler
# 2025-04-25 06:58:39,713 - LiteLLM - INFO - Wrapper: Completed Call, calling success_handler
# 06:58:39 - LiteLLM:INFO: cost_calculator.py:636 - selected model name for cost calculation: openai/model
# 2025-04-25 06:58:39,713 - LiteLLM - INFO - selected model name for cost calculation: openai/model
# 06:58:39 - LiteLLM:INFO: cost_calculator.py:636 - selected model name for cost calculation: openai/model
# 2025-04-25 06:58:39,714 - LiteLLM - INFO - selected model name for cost calculation: openai/model
# 06:58:39 - LiteLLM:INFO: cost_calculator.py:636 - selected model name for cost calculation: openai/model
# 2025-04-25 06:58:39,715 - LiteLLM - INFO - selected model name for cost calculation: openai/model
# 06:58:39 - LiteLLM Router:INFO: router.py:853 - litellm.completion(model=openai/model) 200 OK
# 2025-04-25 06:58:39,715 - LiteLLM Router - INFO - litellm.completion(model=openai/model) 200 OK

print(response)
# Output:
# ModelResponse(id='chatcmpl-ccd95ce8-ca2c-4731-ae42-7ca5d5d130d4', created=1745585918, model='model', object='chat.completion', system_fingerprint=None, choices=[Choices(finish_reason='stop', index=0, message=Message(content='Hello! How can I assist you today? Is there something you would like to discuss or ask about? I am here to answer any questions you may have.', role='assistant', tool_calls=None, function_call=None, provider_specific_fields={'refusal': None}))], usage=Usage(completion_tokens=33, prompt_tokens=23, total_tokens=56, completion_tokens_details=None, prompt_tokens_details=None), service_tier=None, prompt_logprobs=None)

```

### Example 2: Verifying and Using Automatic Selection

```python
# Initialize the router (it will fetch models)
router = KamiwazaRouter(kamiwaza_api_url="https://your-kamiwaza-server.com/api") 

# Ensure models are loaded (optional, router handles this)
# router.model_list is dynamically populated

if len(router.model_list) > 0:
    print(f"Router has {len(router.model_list)} models available. Proceeding with auto-selection.")
    try:
        response = router.completion(
            messages=[{"role": "user", "content": "Tell me a joke."}] 
        )
        print("Completion successful using automatically selected model:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred during completion: {e}")
else:
    print("No models available in the router.")

```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
