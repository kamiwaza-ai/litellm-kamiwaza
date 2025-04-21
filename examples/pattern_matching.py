"""
Example demonstrating pattern matching with KamiwazaRouter.
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
    # Get Kamiwaza URL from environment or use default
    kamiwaza_url = os.getenv("KAMIWAZA_API_URL", "https://localhost/api")
    
    # Initialize the router with a model pattern filter
    logger.info(f"Initializing KamiwazaRouter with API URL: {kamiwaza_url} and pattern 'qwen'")
    router = KamiwazaRouter(
        kamiwaza_api_url=kamiwaza_url,
        model_pattern="qwen"  # Only match models containing "qwen" in their name
    )
    
    # List available models
    model_list = router.get_model_list()
    logger.info(f"Found {len(model_list)} models matching the 'qwen' pattern")
    
    for idx, model in enumerate(model_list):
        model_name = model.get('model_name', 'unknown')
        api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
        logger.info(f"Model #{idx+1}: {model_name} → {api_base}")
    
    # If models are available, try a completion
    if model_list:
        try:
            # Get the first model name
            model_name = model_list[0]["model_name"]
            logger.info(f"Making a completion request to qwen model: {model_name}")
            
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
        logger.warning("No qwen models available for testing")

if __name__ == "__main__":
    main()
