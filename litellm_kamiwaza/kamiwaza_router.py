from asyncio.log import logger
import logging
import litellm
from litellm import Router, completion
from kamiwaza_client import KamiwazaClient
import os
import time
from typing import List, Dict, Any, Optional

# Assuming static_models_conf.py is in the parent directory (project root)
# Adjust the import path if necessary based on your project structure
import sys
# Get the directory containing the current file (llms/litellm)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (llms)
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (project root)
project_root = os.path.dirname(parent_dir)
# Add project root to sys.path to allow importing static_models_conf
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from static_models_conf import get_static_model_configs
except ImportError as e:
    logger.error(f"Could not import 'get_static_model_configs' from 'static_models_conf.py'. Ensure the file exists and is in the correct path relative to the project root. Error: {e}")
    # Define a dummy function or raise the error depending on desired behavior
    def get_static_model_configs() -> Optional[List[Dict[str, Any]]]:
        return None # Or raise ImportError("Static models config not found")


class KamiwazaRouter(Router):
    def __init__(
        self,
        model_list: Optional[List[Dict[str, Any]]] = None,
        kamiwaza_api_url: Optional[str] = None,
        kamiwaza_uri_list: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        model_pattern: Optional[str] = None,
        fallbacks: List = [], # this is also used by super() and we need it for init
        **kwargs  # This captures all other parameters
    ):
        """
        Initialize KamiwazaRouter with Kamiwaza-specific parameters plus all Router parameters.
        
        Args:
            model_list: Optional starting model list (will be merged with Kamiwaza models)
            kamiwaza_api_url: Optional Kamiwaza API URL
            kamiwaza_uri_list: Optional comma-separated list of Kamiwaza URIs
            cache_ttl_seconds: TTL for model cache in seconds
            **kwargs: All other parameters are passed to the Router parent class
        """
        # Initialize the fallbacks attribute first to avoid 'AttributeError: 'KamiwazaRouter' object has no attribute 'fallbacks''.
        # This is called by get_model_list() before parent class initialization completes
        self.fallbacks = fallbacks or []
        # Set up Kamiwaza-specific attributes
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self._cached_model_list: Optional[List[Dict[str, Any]]] = None
        self._cache_timestamp: float = 0.0
        self.cache_ttl_seconds: int = cache_ttl_seconds
        self.model_pattern: Optional[str] = model_pattern
        
        # Initialize Kamiwaza clients
        if not kamiwaza_api_url:
            kamiwaza_api_url = os.getenv("KAMIWAZA_API_URL")
        if not kamiwaza_uri_list:
            kamiwaza_uri_list = os.getenv("KAMIWAZA_URL_LIST")

        # Determine if Kamiwaza source is configured
        self.has_kamiwaza_source = bool(kamiwaza_api_url or kamiwaza_uri_list)

        self.kamiwaza_clients: List[KamiwazaClient] = []
        self.kamiwaza_client: Optional[KamiwazaClient] = None

        # Configure Kamiwaza clients
        if not self.has_kamiwaza_source:
            # Log a warning if no Kamiwaza source is defined, but allow proceeding
            # if static models might be available.
            logger.warning(
                "Neither KAMIWAZA_API_URL nor KAMIWAZA_URL_LIST is set. "
                "Router will rely solely on static models if configured."
            )
        elif kamiwaza_uri_list:
            # Split URI list at commas and create a client for each
            uris = kamiwaza_uri_list.split(",")
            self.kamiwaza_clients = [KamiwazaClient(uri.strip()) for uri in uris if uri.strip()]
            for client in self.kamiwaza_clients:
                if not os.getenv("KAMIWAZA_VERIFY_SSL", "False").lower() == "true": # Check env var safely
                    logger.warning(f"Disabling SSL verification for Kamiwaza client: {client.base_url}")
                    client.session.verify = False
        elif kamiwaza_api_url:
            # Use the API URL if provided
            self.kamiwaza_client = KamiwazaClient(kamiwaza_api_url)
            if not os.getenv("KAMIWAZA_VERIFY_SSL", "False").lower() == "true": # Check env var safely
                logger.warning(f"Disabling SSL verification for Kamiwaza client: {self.kamiwaza_client.base_url}")
                self.kamiwaza_client.session.verify = False

        # Get initial model list from Kamiwaza and static configs
        # When creating a new router with pattern, we should not use cache to ensure proper filtering
        kamiwaza_models = self.get_kamiwaza_model_list(use_cache=False)
        self.logger.info(f"Fetched initial {len(kamiwaza_models or [])} models before pattern filtering")
        
        # Check if any models were actually loaded from Kamiwaza or static configs
        if not kamiwaza_models and not model_list:
            # Raise error only if NO models (neither Kamiwaza nor static) could be loaded and no model_list provided
            raise ValueError(
                "Failed to load any models. No Kamiwaza sources were reachable "
                "and/or no valid static models were configured."
            )
            

        # Merge provided model_list with Kamiwaza models if both exist
        final_model_list = []
        if model_list:
            final_model_list.extend(model_list)
        if kamiwaza_models:
            # Add Kamiwaza models, avoiding duplicates by checking model_name
            existing_model_names = {m.get('model_name') for m in final_model_list if 'model_name' in m}
            for model in kamiwaza_models:
                if model.get('model_name') not in existing_model_names:
                    final_model_list.append(model)
                    
        if model_pattern:
            # Properly filter models by pattern, ensuring it's a substring match with proper case handling
            filtered_models = []
            for m in final_model_list:
                model_name = m.get('model_name', '')
                # Ensure exact pattern match (case insensitive)
                if model_pattern.lower() in model_name.lower():
                    self.logger.debug(f"Model {model_name} matches pattern {model_pattern}")
                    filtered_models.append(m)
                else:
                    self.logger.debug(f"Model {model_name} does NOT match pattern {model_pattern}")
            final_model_list = filtered_models
            self.logger.info(f"Filtered to {len(final_model_list)} models matching pattern '{model_pattern}'")
            if len(final_model_list) == 0:
                self.logger.warning(f"No models match the pattern '{model_pattern}'!")

        # Store in our cache to avoid reloading
        self._cached_model_list = final_model_list.copy()
        self._cache_timestamp = time.time()

        # Call the parent class's __init__ with the merged model list and all other params
        super().__init__(model_list=final_model_list, **kwargs)
        logger.info(f"KamiwazaRouter initialized with {len(final_model_list)} models.")


    def get_models_from_kamiwaza(self, kamiwaza_client: KamiwazaClient) -> List[Dict[str, Any]]:
        """Fetches and formats deployed models from a specific Kamiwaza instance. Returns empty list on failure."""
        try:
            logger.debug(f"Attempting to fetch models from Kamiwaza: {kamiwaza_client.base_url}")
            # Set a timeout of 1 second for deployments API call
            # This is a quick operation if the service is up
            deployments = kamiwaza_client.serving.list_deployments()
            logger.debug(f"Received {len(deployments)} deployments from {kamiwaza_client.base_url}")

            # Filter for deployments that are deployed and have at least one deployed instance
            up_deployments = [
                d for d in deployments
                if hasattr(d, 'status') and d.status == 'DEPLOYED' and # Check attribute existence
                   any(hasattr(i, 'status') and i.status == 'DEPLOYED' for i in getattr(d, 'instances', []))
            ]
            logger.debug(f"Found {len(up_deployments)} 'DEPLOYED' deployments with deployed instances.")

            models_list = []
            for d in up_deployments:
                # Safely get deployment name
                deployment_name = getattr(d, 'name', 'model')

                # Get the first DEPLOYED instance
                deployed_instances = [i for i in getattr(d, 'instances', []) if hasattr(i, 'status') and i.status == 'DEPLOYED']
                instance = deployed_instances[0] if deployed_instances else None

                # Determine the host to use
                host = None
                if instance and hasattr(instance, 'host_name') and instance.host_name:
                    host = instance.host_name
                else:
                    # Fallback: Extract host from client.base_url if instance host_name is missing
                    base_url = kamiwaza_client.base_url
                    try:
                        if '://' in base_url:
                            host_part = base_url.split('://')[1].split('/')[0]
                        else:
                            host_part = base_url.split('/')[0]
                        host = host_part.split(':')[0] if ':' in host_part else host_part
                        # Use m_name for logging context if available
                        log_name = getattr(d, 'm_name', getattr(d, 'name', 'model'))
                        logger.debug(f"Using host '{host}' derived from base_url for deployment '{log_name}'")
                    except IndexError:
                         # Use m_name for logging context if available
                         log_name = getattr(d, 'm_name', getattr(d, 'name', 'model'))
                         logger.warning(f"Could not parse host from base_url: {base_url} for deployment '{log_name}'. Skipping.")
                         continue # Skip this deployment if host cannot be determined

                lb_port = getattr(d, 'lb_port', None)
                # Use the actual model name (m_name) for the outer key
                outer_model_name = getattr(d, 'm_name', None)

                if host and lb_port and outer_model_name:
                    # Use deployment name (d.name) for the inner 'model' parameter targeting the specific endpoint
                    
                    #inner_model_param = f"openai/{getattr(d, 'name', outer_model_name)}" # Fallback to outer name if d.name missing
                    inner_model_param = "openai/model"

                    model_config = {
                        "model_name": outer_model_name, # Use actual model name (m_name) as the identifier
                        "litellm_params": {
                            "model": inner_model_param, # Target the specific deployment endpoint via d.name
                            "api_key": "no_key", # API key is often handled by the proxy/gateway
                            "api_base": f"http://{host}:{lb_port}/v1" # Assuming HTTP endpoint for the load balancer
                        }
                        # Add other params like rpm, tpm if available from deployment object `d`
                        # "rpm": getattr(d, 'rpm_limit', None),
                        # "tpm": getattr(d, 'tpm_limit', None),
                    }
                    models_list.append(model_config)
                    logger.debug(f"Successfully processed deployment '{getattr(d, 'name', outer_model_name)}' (model: {outer_model_name}) from {kamiwaza_client.base_url}")
                else:
                    log_name = getattr(d, 'm_name', getattr(d, 'name', 'model'))
                    missing_parts = []
                    if not host: missing_parts.append(f"host ('{host}')")
                    if not lb_port: missing_parts.append(f"lb_port ('{lb_port}')")
                    if not outer_model_name: missing_parts.append(f"m_name ('{outer_model_name}')")
                    logger.warning(f"Skipping deployment '{log_name}' from {kamiwaza_client.base_url} due to missing: {', '.join(missing_parts)}.")

            logger.info(f"Successfully fetched and processed {len(models_list)} models from {kamiwaza_client.base_url}")
            return models_list
        except Exception as e:
            # Log as warning instead of error, return empty list
            logger.warning(f"Could not fetch or process models from Kamiwaza {kamiwaza_client.base_url}: {e}", exc_info=True)  # Include full traceback for debugging
            return []  # Return empty list to allow other sources to potentially provide models


    def _get_static_models(self) -> List[Dict[str, Any]]:
        """Fetches static model configurations from static_models_conf.py."""
        logger.debug("Attempting to load static model configurations.")
        try:
            static_models = get_static_model_configs() # Call the imported function
            if static_models:
                # Log each model that was loaded
                logger.info(f"Loaded {len(static_models)} static model configurations.")
                for i, model in enumerate(static_models):
                    model_name = model.get('model_name', f'unnamed_model_{i}')
                    # Redact API keys for logging
                    if 'litellm_params' in model and 'api_key' in model['litellm_params']:
                        if model['litellm_params']['api_key'] != 'no_key' and model['litellm_params']['api_key'] != 'sk-no-key-required':
                            api_key_info = '******'
                        else:
                            api_key_info = model['litellm_params']['api_key']
                    else:
                        api_key_info = 'not set'
                    
                    logger.info(f"  Static model {i+1}: {model_name} (API key: {api_key_info})")
                return static_models
            else:
                logger.info("No static model configurations returned or defined.")
                return []
        except Exception as e:
            logger.error(f"Error loading static model configurations from static_models_conf.py: {e}", exc_info=True)
            return [] # Return empty list on error


    def get_kamiwaza_model_list(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves the list of models from Kamiwaza instances and static config,
        using a cache. Refreshes cache if older than cache_ttl_seconds.

        Args:
            use_cache: If True, attempts to use the cached model list if available
                       and not expired. Defaults to True.

        Returns:
            A list of model dictionaries compatible with litellm.Router.
        """
        current_time = time.time()

        # Check cache validity
        if use_cache and self._cached_model_list is not None and \
           (current_time - self._cache_timestamp) < self.cache_ttl_seconds:
            logger.debug("Returning cached model list.")
            return self._cached_model_list

        logger.info(f"Cache expired or not used. Fetching fresh model list (TTL: {self.cache_ttl_seconds}s).")
        all_models: List[Dict[str, Any]] = []

        # Fetch from Kamiwaza instances if configured
        if self.has_kamiwaza_source:
            if self.kamiwaza_client:
                kamiwaza_models = self.get_models_from_kamiwaza(self.kamiwaza_client)
                all_models.extend(kamiwaza_models)
            elif self.kamiwaza_clients:
                for client in self.kamiwaza_clients:
                    client_models = self.get_models_from_kamiwaza(client)
                    all_models.extend(client_models)

        # Fetch static models
        static_models = self._get_static_models()
        all_models.extend(static_models)

        # Filter out duplicate model_name entries, prioritizing non-static models (Kamiwaza)
        # If a static model has the same name as a Kamiwaza model, the Kamiwaza one is kept.
        seen_model_names = set()
        unique_models = []
        # Prioritize Kamiwaza models by adding them first if they exist
        kamiwaza_source_count = 0
        if self.has_kamiwaza_source:
            kamiwaza_models_from_fetch = [m for m in all_models if m not in static_models] # Simple way to separate, might need refinement
            for model in kamiwaza_models_from_fetch:
                model_name = model.get("model_name")
                if model_name and model_name not in seen_model_names:
                    # Ensure model has model_info with at least an id
                    if 'model_info' not in model:
                        model['model_info'] = {'id': model_name}
                    elif 'id' not in model['model_info']:
                        model['model_info']['id'] = model_name
                        
                    unique_models.append(model)
                    seen_model_names.add(model_name)
                    kamiwaza_source_count += 1
                    logger.debug(f"Added Kamiwaza model '{model_name}' from API")
                elif model_name in seen_model_names:
                    logger.warning(f"Duplicate Kamiwaza model '{model_name}' found, skipping.")
                elif not model_name:
                    logger.warning(f"Found Kamiwaza model entry without 'model_name', skipping: {model}")

        # Add static models only if the name hasn't been seen
        static_source_count = 0
        for model in static_models:
            model_name = model.get("model_name")
            if model_name and model_name not in seen_model_names:
                # Ensure model has model_info with at least an id
                if 'model_info' not in model:
                    model['model_info'] = {'id': model_name}
                elif 'id' not in model['model_info']:
                    model['model_info']['id'] = model_name
                    
                unique_models.append(model)
                seen_model_names.add(model_name)
                static_source_count += 1
                logger.info(f"Added static model '{model_name}'")
            elif model_name in seen_model_names:
                # Instead of prioritizing Kamiwaza models, we now prioritize static models
                # Find which model in unique_models has this name
                for i, existing_model in enumerate(unique_models):
                    if existing_model.get("model_name") == model_name:
                        # Replace the existing model with our static model
                        logger.info(f"Static model '{model_name}' conflicts with a Kamiwaza model, prioritizing static model.")
                        unique_models[i] = model
                        break
            elif not model_name:
                 logger.warning(f"Found static model entry without 'model_name', skipping: {model}")

        # Apply model pattern filtering if set
        if hasattr(self, 'model_pattern') and self.model_pattern:
            pattern = self.model_pattern.lower()
            pattern_filtered_models = []
            for m in unique_models:
                model_name = m.get('model_name', '')
                if pattern in model_name.lower():
                    logger.debug(f"Model {model_name} matches pattern {pattern}")
                    pattern_filtered_models.append(m)
                else:
                    logger.debug(f"Model {model_name} does NOT match pattern {pattern}")
            
            logger.info(f"Filtered from {len(unique_models)} to {len(pattern_filtered_models)} models matching pattern '{pattern}'")
            unique_models = pattern_filtered_models
        
        # Update cache
        self._cached_model_list = unique_models
        self._cache_timestamp = current_time
        logger.info(f"Updated model cache. Total unique models: {len(unique_models)} ({kamiwaza_source_count} from Kamiwaza, {static_source_count} static).")

        return unique_models

    # Override get_model_list to be compatible with parent class
    def get_model_list(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Gets the list of available models, compatible with parent Router class.
        
        Args:
            model_name: Optional filter for specific model name.
                      
        Returns:
            List of model dictionaries.
        """
        # First, ensure model_list is up-to-date with our cached data
        current_models = self.get_kamiwaza_model_list(use_cache=True)
        
        # Set the model_list attribute directly
        self.__dict__['model_list'] = current_models
        
        # Setup fallbacks for all models to enable load balancing across different models
        # This allows any model to fallback to any other model in the list
        all_model_names = list(set(m.get('model_name') for m in current_models if m.get('model_name')))
        
        # Create fallback configurations where each model can fallback to all others
        all_fallbacks = {}
        for model_name in all_model_names:
            # For each model, set its fallbacks to all other models
            others = [m for m in all_model_names if m != model_name]
            if others:  # Only add if there are other models
                all_fallbacks[model_name] = others
        
        # Set fallbacks for the router if we have multiple models
        if len(all_fallbacks) > 1:
            # Check if fallbacks attribute exists already
            if hasattr(self, 'fallbacks'):
                self.fallbacks = [all_fallbacks]
                logger.info(f"Setup cross-model fallbacks for {len(all_model_names)} models")
                # Also add a wildcard fallback that matches any model pattern
                self.fallbacks.append({"*": all_model_names})
            else:
                # During initialization, we can't modify self.fallbacks yet
                # We'll just prepare the data for the parent class initialization
                logger.info(f"Preparing fallbacks for {len(all_model_names)} models during initialization")
                return current_models
            logger.info(f"Added wildcard fallback to all {len(all_model_names)} models")
        
        # Use Router's implementation for filtering to maintain compatibility
        try:
            return Router.get_model_list(self, model_name=model_name)
        except RecursionError:
            # If we hit a recursion error, fall back to simple filtering
            if model_name is not None:
                return [m for m in current_models if m.get('model_name') == model_name]
            return current_models

    # Override set_model_list to clear cache if models are set externally
    def set_model_list(self, model_list: list):
        """Sets the model list directly and clears the cache."""
        super().set_model_list(model_list=model_list)
        # Clear cache when model list is set manually
        self._cached_model_list = None
        self._cache_timestamp = 0.0
        logger.info("Model list set externally via set_model_list, cache cleared.")
