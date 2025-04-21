"""
Example demonstrating basic usage of KamiwazaRouter.

This example demonstrates how to initialize the KamiwazaRouter
and make a simple completion request.

Environment Variables:
    KAMIWAZA_API_URL: URL for the Kamiwaza API
    or
    KAMIWAZA_URL_LIST: Comma-separated list of Kamiwaza URLs
"""

import os
import logging
import pathlib
from dotenv import load_dotenv
from litellm_kamiwaza import KamiwazaRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    env_path = pathlib.Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)
    
    # Get Kamiwaza URL from environment
    kamiwaza_url = os.environ.get("KAMIWAZA_API_URL", "https://localhost/api")
    
    # Initialize the router
    logger.info(f"Initializing KamiwazaRouter with API URL: {kamiwaza_url}")
    router = KamiwazaRouter(
        kamiwaza_api_url=kamiwaza_url,
        cache_ttl_seconds=300  # Cache model list for 5 minutes
    )
    
    # List available models
    model_list = router.get_model_list()
    logger.info(f"Found {len(model_list)} available models")
    
    for idx, model in enumerate(model_list):
        model_name = model.get('model_name', 'unknown')
        api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
        provider = model.get('model_info', {}).get('provider', 'unknown')
        logger.info(f"Model #{idx+1}: {model_name} → {api_base} (Provider: {provider})")
    
    # If models are available, try a completion
    if model_list:
        try:
            # Get the first model name
            model_name = model_list[0]["model_name"]
            logger.info(f"Making a completion request to model: {model_name}")
            
            # Make a completion request
            response = router.completion(
                model=model_name,
                messages=[{"role": "user", "content": "What is artificial intelligence?"}],
                max_tokens=100
            )
            
            # Print the response
            content = response['choices'][0]['message']['content']
            logger.info(f"Response from {model_name} (first 150 chars):\n{content[:150]}...")
            
        except Exception as e:
            logger.error(f"Error making completion request: {e}")
    else:
        logger.warning("No models available for testing")

if __name__ == "__main__":
    main()
