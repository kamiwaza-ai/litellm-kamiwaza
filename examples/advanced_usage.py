"""
Advanced usage example for litellm-kamiwaza package.

This example demonstrates:
1. Using model pattern filtering
2. Setting up static models alongside Kamiwaza models
3. Using fallbacks for reliability
4. Batch processing with multiple models
"""

import os
import logging
from litellm_kamiwaza import KamiwazaRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Define static models (can be used alongside Kamiwaza models)
    static_models = [
        {
            "model_name": "openai-model",
            "litellm_params": {
                "model": "gpt-3.5-turbo",
                "api_key": "${OPENAI_API_KEY:-dummy_key}"  # Uses env var with fallback
            },
            "model_info": {
                "id": "openai-model-id"
            }
        }
    ]
    
    # Initialize router with advanced options
    logger.info("Initializing KamiwazaRouter with advanced options...")
    
    try:
        router = KamiwazaRouter(
            model_list=static_models,
            # Filter to only use models with "72b" in their name
            model_pattern="72b",
            # Cache model list for 5 minutes
            cache_ttl_seconds=300,
            # Use fallbacks when models fail
            fallbacks=[
                # If any 72b model fails, try openai-model
                {"*": ["openai-model"]}
            ],
            # Default parameters for all requests
            default_litellm_params={
                "timeout": 30,
                "max_retries": 3,
                "metadata": {"source": "litellm-kamiwaza-example"}
            }
        )
        
        # Get available models
        logger.info("Fetching filtered model list...")
        models = router.get_model_list()
        
        if not models:
            logger.warning("No models available after filtering")
            return
        
        # Display available models
        logger.info(f"Found {len(models)} available models after filtering:")
        for i, model in enumerate(models):
            logger.info(f"  {i+1}. {model['model_name']}")
        
        # Use model pattern
        model_72b = None
        for model in models:
            if "72b" in model["model_name"].lower():
                model_72b = model["model_name"]
                break
        
        if model_72b:
            logger.info(f"Found 72b model: {model_72b}")
            
            # Make a completion request with the 72b model
            logger.info("Making completion request with 72b model...")
            response = router.completion(
                model=model_72b,
                messages=[{"role": "user", "content": "Explain the concept of recursive functions in 2 sentences"}]
            )
            logger.info(f"72b model response: {response.choices[0].message.content}")
        else:
            logger.info("No 72b model found, using first available model")
            model_72b = models[0]["model_name"]
        
        # Batch processing with multiple models
        logger.info("Performing batch processing across all available models...")
        batch_response = router.abatch_completion(
            models=[m["model_name"] for m in models],
            messages=[{"role": "user", "content": "What's your name?"}]
        )
        
        for i, resp in enumerate(batch_response):
            if isinstance(resp, Exception):
                logger.error(f"Model #{i+1} failed with: {resp}")
            else:
                logger.info(f"Model #{i+1} response: {resp.choices[0].message.content}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
