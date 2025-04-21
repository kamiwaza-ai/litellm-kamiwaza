"""
Example demonstrating pattern matching with KamiwazaRouter.
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
    
    # Get model pattern from environment
    model_pattern = os.environ.get("KAMIWAZA_MODEL_PATTERN", "qwen")
    
    # Get Kamiwaza URLs from environment
    test_url_list = os.environ.get("KAMIWAZA_TEST_URL_LIST", "")
    api_urls = [url.strip() for url in test_url_list.split(",") if url.strip()]
    
    # If no URLs specified, use the single API URL as fallback
    if not api_urls:
        api_url = os.environ.get("KAMIWAZA_API_URL", "https://localhost/api")
        api_urls = [api_url]
    
    # Create a comma-separated string of URLs
    uri_list = ",".join(api_urls)
    
    # Initialize the router with a model pattern filter
    logger.info(f"Initializing KamiwazaRouter with URI list: {uri_list} and pattern '{model_pattern}'")
    router = KamiwazaRouter(
        kamiwaza_uri_list=uri_list,
        model_pattern=model_pattern
    )
    
    # List available models
    model_list = router.get_model_list()
    logger.info(f"Found {len(model_list)} models matching the '{model_pattern}' pattern")
    
    for idx, model in enumerate(model_list):
        model_name = model.get('model_name', 'unknown')
        api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
        logger.info(f"Model #{idx+1}: {model_name} → {api_base}")
    
    # If models are available, try a completion
    if model_list:
        try:
            # Get the first model name
            model_name = model_list[0]["model_name"]
            logger.info(f"Making a completion request to {model_pattern} model: {model_name}")
            
            # Make a completion request
            response = router.completion(
                model=model_name,
                messages=[{"role": "user", "content": "Write a haiku about artificial intelligence."}],
                max_tokens=30
            )
            
            # Print the response
            content = response['choices'][0]['message']['content']
            logger.info(f"Response from {model_name}:\n{content}")
            
        except Exception as e:
            logger.error(f"Error making completion request: {e}")
    else:
        logger.warning(f"No models matching '{model_pattern}' available for testing")

if __name__ == "__main__":
    main()
