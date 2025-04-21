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
    
    # Get Kamiwaza URL list from environment
    test_url_list = os.environ.get("KAMIWAZA_TEST_URL_LIST", "")
    api_urls = [url.strip() for url in test_url_list.split(",") if url.strip()]
    
    # If no URLs specified, use the single API URL as fallback
    if not api_urls:
        api_url = os.environ.get("KAMIWAZA_API_URL", "https://localhost/api")
        api_urls = [api_url]
    
    # Create a comma-separated string of URLs
    uri_list = ",".join(api_urls)
    
    # Initialize the router with multiple instances
    logger.info(f"Initializing KamiwazaRouter with {len(api_urls)} Kamiwaza instances")
    for i, url in enumerate(api_urls):
        logger.info(f"  Instance {i+1}: {url}")
    
    router = KamiwazaRouter(
        kamiwaza_uri_list=uri_list,
        cache_ttl_seconds=0  # Disable caching for this example
    )
    
    # Get all available models
    model_list = router.get_model_list()
    logger.info(f"Found {len(model_list)} total available models")
    
    # Group models by instance
    models_by_instance = {}
    static_models = []
    
    for model in model_list:
        model_name = model.get('model_name', 'unknown')
        provider = model.get('model_info', {}).get('provider', 'unknown')
        
        if provider == 'static':
            static_models.append(model)
        else:
            api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
            if api_base not in models_by_instance:
                models_by_instance[api_base] = []
            models_by_instance[api_base].append(model)
    
    # Print model details
    logger.info(f"Models by source:")
    logger.info(f"  - Static models: {len(static_models)}")
    if static_models:
        for i, model in enumerate(static_models):
            model_name = model.get('model_name', 'unknown')
            api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
            logger.info(f"    {i+1}. {model_name} → {api_base}")
    
    logger.info(f"  - Kamiwaza models: {sum(len(models) for models in models_by_instance.values())} across {len(models_by_instance)} instances")
    for instance_url, instance_models in models_by_instance.items():
        logger.info(f"    • {instance_url}: {len(instance_models)} models")
        for i, model in enumerate(instance_models):
            model_name = model.get('model_name', 'unknown')
            logger.info(f"      {i+1}. {model_name}")
    
    # Test one model from each source if available
    models_to_test = []
    
    # Add one static model if available
    if static_models:
        models_to_test.append(static_models[0])
    
    # Add one model from each Kamiwaza instance
    for instance_url, instance_models in models_by_instance.items():
        if instance_models:
            models_to_test.append(instance_models[0])
    
    # Test each selected model
    if models_to_test:
        logger.info(f"Testing {len(models_to_test)} models (one from each source)")
        
        for model in models_to_test:
            model_name = model.get('model_name', 'unknown')
            api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
            provider = model.get('model_info', {}).get('provider', 'unknown')
            source = "static" if provider == "static" else "Kamiwaza"
            
            logger.info(f"\nTesting {source} model: {model_name} → {api_base}")
            
            try:
                # Make a completion request
                response = router.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": "Explain how routers work in LiteLLM in one sentence."}],
                    max_tokens=50
                )
                
                # Print the response
                content = response['choices'][0]['message']['content']
                logger.info(f"Response from {model_name}:\n{content}")
                logger.info(f"✅ Success with {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Error with {model_name}: {e}")
    else:
        logger.warning("No models available for testing")

if __name__ == "__main__":
    main()
