# LiteLLM Router Integration for Kamiwaza AI

This package provides a custom router for [LiteLLM](https://github.com/BerriAI/litellm) that integrates with [Kamiwaza AI](https://kamiwaza.ai) model deployments. The `KamiwazaRouter` extends LiteLLM's `Router` class to enable efficient routing of requests to Kamiwaza-deployed models.

## Features

- **Dynamic Model Discovery**: Automatically discovers available models from Kamiwaza deployments
- **Caching**: Efficient caching of model lists with configurable TTL
- **Model Pattern Filtering**: Filter models based on name patterns (e.g., only use "72b" models)
- **Static Model Configuration**: Support for static model configurations alongside Kamiwaza models
- **Fallback Routing**: Automatic fallback between models in case of failures

## Installation

```bash
pip install litellm-kamiwaza
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

- `KAMIWAZA_API_URL`: URL for the Kamiwaza API
- `KAMIWAZA_URL_LIST`: Comma-separated list of Kamiwaza URLs
- `KAMIWAZA_VERIFY_SSL`: Set to "true" to enable SSL verification (default: "false")

#### Router Configuration

```python
# Initialize with specific Kamiwaza URL
router = KamiwazaRouter(
    kamiwaza_api_url="http://my-kamiwaza-server.com",
    cache_ttl_seconds=600,  # Cache model list for 10 minutes
    model_pattern="72b",    # Only use models with "72b" in their name
)

# Initialize with static model list alongside Kamiwaza models
router = KamiwazaRouter(
    model_list=[
        {
            "model_name": "my-static-model",
            "litellm_params": {
                "model": "openai/gpt-4",
                "api_key": "sk-your-api-key"
            }
        }
    ]
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
