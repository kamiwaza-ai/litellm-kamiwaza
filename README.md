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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
