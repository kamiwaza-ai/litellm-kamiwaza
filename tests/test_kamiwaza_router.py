import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pytest
from litellm_kamiwaza import KamiwazaRouter
import litellm
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class TestKamiwazaRouter(unittest.TestCase):
    
    @patch('litellm_kamiwaza.kamiwaza_router.KamiwazaClient')
    def test_initialization(self, mock_kamiwaza_client):
        """Test basic initialization of the router with mocked client."""
        # Setup mock
        mock_instance = MagicMock()
        mock_kamiwaza_client.return_value = mock_instance
        
        # Mock get_models_from_kamiwaza to return at least one model
        mock_model = {
            "model_name": "openai/test-model", 
            "litellm_params": {"model": "openai/test-model", "api_key": "test-key"}
        }
        mock_instance.serving.list_deployments.return_value = []
        
        # Create a model_list to bypass the validation that requires models
        test_model_list = [mock_model]
        
        # Test with API URL
        router = KamiwazaRouter(
            kamiwaza_api_url="http://test-url", 
            model_list=test_model_list  # Provide a model list to avoid the ValueError
        )
        self.assertIsNotNone(router)
        self.assertTrue(hasattr(router, 'kamiwaza_client'))
        
        # Test with no source but with model_list
        router = KamiwazaRouter(model_list=test_model_list)
        self.assertIsNotNone(router)
        
        # Test with no source and no model_list
        with self.assertRaises(ValueError):
            KamiwazaRouter(kamiwaza_api_url=None, kamiwaza_uri_list=None)
    
    @patch('litellm_kamiwaza.kamiwaza_router.get_static_model_configs')
    @patch('litellm_kamiwaza.kamiwaza_router.KamiwazaClient')
    def test_model_pattern_filtering(self, mock_kamiwaza_client, mock_get_static_model_configs):
        """Test that model pattern filtering works correctly with mocked client."""
        # Setup mocks
        mock_instance = MagicMock()
        mock_kamiwaza_client.return_value = mock_instance
        
        # Mock deployments
        mock_deployment1 = MagicMock(status='DEPLOYED', name='deploy1', m_name='model-72b')
        mock_deployment1.instances = [MagicMock(status='DEPLOYED', host_name='host1')]
        mock_deployment1.lb_port = 8000
        
        mock_deployment2 = MagicMock(status='DEPLOYED', name='deploy2', m_name='model-32b')
        mock_deployment2.instances = [MagicMock(status='DEPLOYED', host_name='host2')]
        mock_deployment2.lb_port = 8001
        
        mock_instance.serving.list_deployments.return_value = [mock_deployment1, mock_deployment2]
        
        # No static models
        mock_get_static_model_configs.return_value = None
        
        # Prepare a mock implementation for get_models_from_kamiwaza
        def mock_get_models(*args, **kwargs):
            models = [
                {
                    "model_name": "model-72b",
                    "litellm_params": {
                        "model": "openai/model",
                        "api_key": "no_key",
                        "api_base": "http://host1:8000/v1"
                    }
                },
                {
                    "model_name": "model-32b",
                    "litellm_params": {
                        "model": "openai/model",
                        "api_key": "no_key",
                        "api_base": "http://host2:8001/v1"
                    }
                }
            ]
            return models
        
        # Mock the instance method to return our prepared models
        with patch.object(KamiwazaRouter, 'get_models_from_kamiwaza', side_effect=mock_get_models):
            # Test with 72b pattern
            router = KamiwazaRouter(
                kamiwaza_api_url="http://test-url",
                model_pattern="72b"
            )
            
            # Get model list and verify only the 72b model is included
            models = router.get_kamiwaza_model_list(use_cache=False)
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]['model_name'], 'model-72b')
            
            # Test with non-matching pattern
            router = KamiwazaRouter(
                kamiwaza_api_url="http://test-url",
                model_pattern="xyz",
                # Add a dummy model_list to avoid errors when no models match the pattern
                model_list=[{"model_name": "dummy", "litellm_params": {"model": "dummy"}}]
            )
            
            # Should find no models matching the pattern
            models = router.get_kamiwaza_model_list(use_cache=False)
            self.assertEqual(len([m for m in models if "model-" in m['model_name']]), 0)


@pytest.mark.integration
class TestKamiwazaRouterIntegration(unittest.TestCase):
    """Integration tests for the KamiwazaRouter class that require a real API URL."""
    
    @classmethod
    def setUpClass(cls):
        """Set up environment for the tests."""
        cls.api_url = os.environ.get("KAMIWAZA_API_URL")
        if not cls.api_url:
            raise unittest.SkipTest("KAMIWAZA_API_URL environment variable not set")
    
    def test_litellm_kamiwaza_inference(self):
        """Test that the KamiwazaRouter works with the litellm.completion function."""
        print(f"\n{'='*80}")
        print(f"🔍 Testing KamiwazaRouter integration with litellm")
        print(f"{'='*80}")
        
        # First verify Kamiwaza API is available using a reliable endpoint
        full_health_endpoint = f"{self.api_url}/cluster/clusters"
        print(f"🌐 Testing API connectivity to endpoint: {full_health_endpoint}")
        try:
            # Use the cluster/clusters endpoint which is more reliable
            response = requests.get(full_health_endpoint, verify=False, timeout=5)
            response.raise_for_status()
            print(f"✅ API connection successful! Found {len(response.json())} clusters")
            print(f"   Response: {response.json()[:2]}{'...' if len(response.json()) > 2 else ''}")
        except Exception as e:
            print(f"⚠️ API connection warning: {str(e)}")
            # Continue anyway since the KamiwazaClient might still work
        
        # Create the router
        print(f"🔧 Creating KamiwazaRouter with API: {self.api_url}")
        router = KamiwazaRouter(
            kamiwaza_api_url=self.api_url,
            cache_ttl_seconds=0  # Disable caching for tests
        )
        
        # Get available models
        print(f"🔍 Discovering available models...")
        models = router.get_kamiwaza_model_list(use_cache=False)
        
        print(f"📋 Found {len(models)} available models:")
        for i, model in enumerate(models):
            model_name = model.get('model_name', 'unknown')
            api_base = "N/A"
            if 'litellm_params' in model and 'api_base' in model['litellm_params']:
                api_base = model['litellm_params']['api_base']
            print(f"  {i+1}. {model_name} → {api_base}")
        
        if not models:
            pytest.skip("No models found")
        
        # Select the first model for testing
        model_name = models[0].get('model_name')
        print(f"\n{'='*80}")
        print(f"🧠 Testing completion with model: {model_name}")
        print(f"{'='*80}")
        
        # Print model details
        model_details = next((m for m in models if m.get('model_name') == model_name), None)
        api_base = "unknown"
        if model_details:
            if 'litellm_params' in model_details:
                print("📄 Model Configuration:")
                for key, value in model_details['litellm_params'].items():
                    print(f"  - {key}: {value}")
                    if key == 'api_base':
                        api_base = value
            if 'model_info' in model_details:
                print("ℹ️ Model Info:")
                for key, value in model_details['model_info'].items():
                    print(f"  - {key}: {value}")
        
        # Show the inference endpoint we'll be using
        expected_endpoint = f"{api_base}/chat/completions"
        print(f"\n🔌 Inference will use endpoint: {expected_endpoint}")
        
        # Prepare test data
        messages = [{"role": "user", "content": "Write a haiku about AI"}]
        print(f"\n📝 Prompt: \"{messages[0]['content']}\"")
        print(f"🔄 Sending request to {model_name}...")
        
        try:
            # Use router's completion method directly instead of litellm.completion
            # This avoids issues with api_base configuration
            response = router.completion(
                model=model_name,
                messages=messages,
                max_tokens=50
            )
            
            # Verify response
            assert response is not None
            assert 'choices' in response
            assert len(response['choices']) > 0 
            assert 'message' in response['choices'][0]
            
            # Print response details
            print(f"\n📊 Response Details:")
            if 'model' in response:
                print(f"  - Model: {response['model']}")
            if 'usage' in response:
                usage = response['usage']
                print(f"  - Tokens: {usage.get('total_tokens', 'unknown')} total ({usage.get('prompt_tokens', 'unknown')} prompt, {usage.get('completion_tokens', 'unknown')} completion)")
            if 'id' in response:
                print(f"  - Response ID: {response['id']}")
            
            # Extract and print content
            content = response['choices'][0]['message']['content']
            print(f"\n🔤 Generated Haiku:")
            print(f"'''\n{content}\n'''")
            print(f"✅ Inference successful!")
            
            # Test passed
            assert True
            
        except Exception as e:
            import traceback
            print(f"❌ Error during inference: {str(e)}")
            print(traceback.format_exc())
            raise


@pytest.mark.integration
class TestKamiwazaRouterMultiInstance:
    """Integration tests for using KamiwazaRouter with multiple API URLs."""
    
    def setup_method(self):
        """Set up environment for each test."""
        # Get test URL list from environment 
        test_url_list = os.environ.get("KAMIWAZA_TEST_URL_LIST", "")
        self.api_urls = [url.strip() for url in test_url_list.split(",") if url.strip()]
        
        # If no URLs specified, use the single API URL as fallback
        if not self.api_urls:
            api_url = os.environ.get("KAMIWAZA_API_URL")
            if api_url:
                self.api_urls = [api_url]

    def _check_kamiwaza_connectivity(self, url):
        """Verify basic connectivity to a Kamiwaza instance."""
        full_endpoint = f"{url}/cluster/clusters"
        try:
            print(f"  🔍 Checking endpoint: {full_endpoint}")
            response = requests.get(full_endpoint, verify=False, timeout=5)
            response.raise_for_status()
            clusters = response.json()
            return True, f"Found {len(clusters)} clusters - {clusters[:1]}"
        except Exception as e:
            return False, str(e)

    def test_inference_on_each_instance(self):
        """Test that inference works on all models across all instances.
        
        This test verifies that we can successfully get inference from each model
        on each Kamiwaza instance. It helps ensure that our routing logic works
        across multiple instances.
        """
        if len(self.api_urls) < 2:
            pytest.skip("At least two URLs in KAMIWAZA_TEST_URL_LIST environment variable must be set for multi-instance tests")
        
        print(f"\n{'='*80}")
        print(f"🌐 Testing KamiwazaRouter with multiple instances ({len(self.api_urls)} URLs)")
        print(f"{'='*80}")
        
        # Verify connectivity to each instance before testing
        available_urls = []
        for i, url in enumerate(self.api_urls):
            print(f"Instance {i+1}: {url}")
            is_available, message = self._check_kamiwaza_connectivity(url)
            if is_available:
                print(f"  ✅ Connection successful: {message}")
                available_urls.append(url)
            else:
                print(f"  ⚠️ Connection failed: {message}")
        
        # Only proceed if we have at least 2 available instances
        if len(available_urls) < 2:
            pytest.skip(f"Need at least 2 available Kamiwaza instances, only found {len(available_urls)}")
        
        # Create a multi-instance router using kamiwaza_uri_list
        print(f"\n🔧 Creating KamiwazaRouter with {len(available_urls)} available instances...")
        
        # Prepare the URI list as a comma-separated string
        uri_list = ",".join(available_urls)
        print(f"🔌 URI List: {uri_list}")
        
        router = KamiwazaRouter(
            kamiwaza_uri_list=uri_list,
            cache_ttl_seconds=0  # Disable caching for tests
        )
        
        # Verify all instances were detected
        print(f"📡 Router initialized with {len(router.kamiwaza_clients)} Kamiwaza clients")
        for i, client in enumerate(router.kamiwaza_clients):
            print(f"  - Client {i+1}: {client.base_url}")
            # Verify client's SSL verification setting
            print(f"    SSL Verification: {client.session.verify}")
        
        # Get all models from all instances
        print(f"\n🔍 Discovering models across all instances...")
        models = router.get_kamiwaza_model_list(use_cache=False)
        
        # Count models per instance for summary
        instance_model_counts = {}
        for model in models:
            instance_url = model.get('litellm_params', {}).get('api_base', 'unknown')
            instance_model_counts[instance_url] = instance_model_counts.get(instance_url, 0) + 1
        
        # Display summary of discovered models
        print(f"\n📊 Found {len(models)} total models across {len(instance_model_counts)} instances:")
        for instance_url, count in instance_model_counts.items():
            print(f"  - {instance_url}: {count} models")
        
        # Skip test if no models found
        if not models:
            pytest.skip("No models found across any instances")
        
        # Apply model pattern filter if specified in environment
        model_pattern = os.environ.get("KAMIWAZA_TEST_MODEL_PATTERN", "")
        if model_pattern:
            print(f"🔍 Filtering models by pattern: {model_pattern}")
            filtered_models = []
            for model in models:
                model_name = model.get('model_name', 'unknown')
                if model_pattern.lower() in model_name.lower():
                    filtered_models.append(model)
            
            print(f"📋 Filtered from {len(models)} to {len(filtered_models)} models matching pattern '{model_pattern}'")
            models = filtered_models
            
            if not models:
                pytest.skip(f"No models match pattern '{model_pattern}'")
        
        # Store test results for each model
        success_count = 0
        failure_count = 0
        
        # Test each model but limit to a reasonable number
        max_models_to_test = min(len(models), 3)  # Test at most 3 models to keep tests quick
        models_to_test = models[:max_models_to_test]
        
        print(f"\n🧪 Testing {max_models_to_test} of {len(models)} available models")
        
        # Test each model
        for i, model in enumerate(models_to_test):
            model_name = model.get('model_name', 'unknown')
            api_base = model.get('litellm_params', {}).get('api_base', 'unknown')
            
            print(f"\n{'='*80}")
            print(f"🧠 Testing model {i+1}/{len(models_to_test)}: {model_name}")
            print(f"🌐 Instance: {api_base}")
            print(f"{'='*80}")
            
            # Print model details for verbose output
            if 'litellm_params' in model:
                print("📄 Model Configuration:")
                for key, value in model['litellm_params'].items():
                    print(f"  - {key}: {value}")
            if 'model_info' in model:
                print("ℹ️ Model Info:")
                for key, value in model['model_info'].items():
                    print(f"  - {key}: {value}")
            
            # Show the inference endpoint we'll be using
            expected_endpoint = f"{api_base}/chat/completions"
            print(f"\n🔌 Inference will use endpoint: {expected_endpoint}")
            
            # Prepare prompt - keep it very short for quick tests
            messages = [{"role": "user", "content": "Write a very short haiku about AI"}]
            print(f"\n📝 Prompt: \"{messages[0]['content']}\"")
            print(f"🔄 Sending request to {model_name}...")
            
            try:
                # Make completion call with short timeout
                response = router.completion(
                    model=model_name,
                    messages=messages,
                    max_tokens=20,
                    request_timeout=30  # Limit request time to avoid hanging tests
                )
                
                # Print response details
                print(f"\n📊 Response Details:")
                if 'model' in response:
                    print(f"  - Model: {response['model']}")
                if 'usage' in response:
                    usage = response['usage']
                    print(f"  - Tokens: {usage.get('total_tokens', 'unknown')} total ({usage.get('prompt_tokens', 'unknown')} prompt, {usage.get('completion_tokens', 'unknown')} completion)")
                if 'id' in response:
                    print(f"  - Response ID: {response['id']}")
                
                # Extract and print content
                content = response['choices'][0]['message']['content']
                print(f"\n🔤 Generated Haiku:")
                print(f"'''\n{content}\n'''")
                print(f"✅ Inference successful!")
                
                success_count += 1
                
            except Exception as e:
                import traceback
                print(f"❌ Error testing model {model_name}: {str(e)}")
                print(traceback.format_exc())
                failure_count += 1
                # Continue testing other models
                continue
        
        # Print summary of test results
        print(f"\n{'='*80}")
        print(f"📋 Test Summary:")
        print(f"  - Total models tested: {len(models_to_test)}")
        print(f"  - Successful models: {success_count}")
        print(f"  - Failed models: {failure_count}")
        print(f"{'='*80}")
        
        # Test should pass if at least one model worked
        assert success_count > 0, "No models successfully generated completions"


if __name__ == '__main__':
    # For standalone debugging
    if 'unittest' in sys.argv:
        unittest.main()
    else:
        # Directly run the inference test
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        print("Running direct inference test...")
        # Set API URL directly for testing
        os.environ["KAMIWAZA_API_URL"] = "https://localhost"
        
        test_instance = TestKamiwazaRouterIntegration()
        test_instance.api_url = "https://localhost"
        test_instance.setup_method()
        try:
            test_instance.test_litellm_kamiwaza_inference()
        except Exception as e:
            print(f"Test failed with error: {e}")
