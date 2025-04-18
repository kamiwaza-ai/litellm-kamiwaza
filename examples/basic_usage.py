"""
Basic usage example for litellm-kamiwaza package.

This example demonstrates how to initialize the KamiwazaRouter
and make a simple completion request.

Environment Variables:
    KAMIWAZA_API_URL: URL for the Kamiwaza API
    or
    KAMIWAZA_URL_LIST: Comma-separated list of Kamiwaza URLs
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
    # Initialize the router
    logger.info("Initializing KamiwazaRouter...")
    
    try:
        # Create router with default settings (uses environment variables)
        router = KamiwazaRouter()
        
        # Get available models
        logger.info("Fetching model list...")
        models = router.get_model_list()
        
        if not models:
            logger.warning("No models available")
            return
        
        # Display available models
        logger.info(f"Found {len(models)} available models:")
        for i, model in enumerate(models):
            logger.info(f"  {i+1}. {model['model_name']}")
        
        # Get the first model
        first_model = models[0]['model_name']
        logger.info(f"Using model: {first_model}")
        
        # Make a completion request
        logger.info("Making completion request...")
        response = router.completion(
            model=first_model,
            messages=[{"role": "user", "content": "Tell me a haiku about AI"}]
        )
        
        # Display response
        logger.info(f"Response: {response.choices[0].message.content}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
