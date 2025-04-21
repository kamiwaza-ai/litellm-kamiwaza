"""
Static model configuration for Kamiwaza tests.

This file defines static LLM model configurations that can be used
in combination with or instead of models discovered from Kamiwaza.
"""
from typing import List, Dict, Any, Optional


def get_static_model_configs() -> Optional[List[Dict[str, Any]]]:
    """
    Returns a list of statically defined model configurations.
    
    These will be merged with models discovered from Kamiwaza instances,
    or used alone if no Kamiwaza instances are available.
    
    Returns:
        List of model configuration dictionaries compatible with litellm Router,
        or None if no static models are defined.
    """
    return [
        {
            "model_name": "static-eschaton-model", 
            "litellm_params": {
                "model": "openai/model",  # OpenAI-compatible model type
                "api_key": "no_key",      # No API key required
                "api_base": "http://eschaton.local:51107/v1"  # Base URL for the model
            },
            "model_info": {
                "id": "static-eschaton-model",
                "provider": "static",
                "description": "Static model configuration for testing"
            }
        }
    ] 