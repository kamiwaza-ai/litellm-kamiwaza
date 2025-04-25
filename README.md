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

### Using a Discovered Model

After initialization, the `router.model_list` attribute contains the discovered models. You can use this list to select a model for your completion request.

```python
from litellm_kamiwaza import KamiwazaRouter

# Initialize the router (will fetch models from Kamiwaza)
router = KamiwazaRouter(kamiwaza_api_url="https://my-kamiwaza-server.com/api")

# Wait for models to be loaded or ensure they are loaded
# For example, you might check router.model_list

if router.model_list:
    # Select the first discovered model
    selected_model = router.model_list[0]['model_name']
    print(f"Using discovered model: {selected_model}")

    response = router.completion(
        model=selected_model,
        messages=[{"role": "user", "content": "Hello, world!"}]
    )
    print(response)

    # Example log output and response (details may vary):
    # 2025-04-25 06:58:38,079 - asyncio - INFO - Cache expired or not used. Fetching fresh model list (TTL: 300s).
    # 2025-04-25 06:58:38,198 - asyncio - INFO - Successfully fetched and processed 1 models from https://my-kamiwaza-server.com/api
    # 2025-04-25 06:58:38,198 - asyncio - INFO - No static model configurations returned or defined.
    # 2025-04-25 06:58:38,198 - asyncio - INFO - Updated model cache. Total unique models: 1 (1 from Kamiwaza, 0 static).
    # ... (LiteLLM logs) ...
    # 2025-04-25 06:58:39,701 - httpx - INFO - HTTP Request: POST http://my-kamiwaza-server.com/v1/chat/completions "HTTP/1.1 200 OK"
    # ... (LiteLLM logs) ...
    # 2025-04-25 06:58:39,715 - LiteLLM Router - INFO - litellm.completion(model=openai/model) 200 OK
    #
    # ModelResponse(id='chatcmpl-ccd95ce8-ca2c-4731-ae42-7ca5d5d130d4', ...)

else:
    print("No models found by the router.")

### Example: Checking Length and Randomly Selecting

This example shows how to check if models are available and then randomly select one for the request:

```python
import random
from litellm_kamiwaza import KamiwazaRouter

# Initialize the router 
router = KamiwazaRouter(kamiwaza_api_url="https://my-kamiwaza-server.com/api")

# Check if the model list has been populated and is not empty
if router.model_list and len(router.model_list) > 0:
    # Randomly select an available model dictionary
    selected_model_dict = random.choice(router.model_list)
    model_to_use = selected_model_dict["model_name"]
    
    print(f"Found {len(router.model_list)} models. Randomly selected: {model_to_use}")
    
    try:
        response = router.completion(
            model=model_to_use,
            messages=[{"role": "user", "content": "Translate 'hello' to French."}]
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error during completion: {e}")
else:
    print("Router initialization failed or no models were discovered.")

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
