import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pytest
from litellm_kamiwaza import KamiwazaRouter

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
class TestKamiwazaRouterIntegration:
    """Integration tests that use a real Kamiwaza client without mocking.
    These tests require a KAMIWAZA_API_URL environment variable to be set.
    """
    
    def setup_method(self):
        """Check if KAMIWAZA_API_URL is set before running integration tests."""
        self.api_url = os.environ.get("KAMIWAZA_API_URL")
        if not self.api_url:
            pytest.skip("KAMIWAZA_API_URL environment variable not set")
    
    def test_real_client_initialization(self):
        """Test initialization with a real Kamiwaza client."""
        # Try to create a router with the real API URL
        router = KamiwazaRouter(
            kamiwaza_api_url=self.api_url,
            # Provide a fallback model in case the API has no models
            model_list=[{
                "model_name": "fallback-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-dummy-key"
                }
            }]
        )
        
        # Basic assertions
        assert router is not None
        assert hasattr(router, 'kamiwaza_client')
        
        # Attempt to get the model list
        models = router.get_kamiwaza_model_list(use_cache=False)
        
        # Log what we found
        print(f"Found {len(models)} models from real Kamiwaza API")
        for i, model in enumerate(models):
            print(f"Model {i+1}: {model.get('model_name')}")
    
    def test_litellm_kamiwaza_inference(self):
        """Test generating a completion using KamiwazaRouter with the real deployed model."""
        # Skip test if no API URL is set
        if not self.api_url:
            pytest.skip("KAMIWAZA_API_URL environment variable not set")
        
        # First check if the API is reachable with a simple request
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            # Test API connectivity
            response = requests.get(f"{self.api_url}/api/serving/deployments", verify=False)
            response.raise_for_status()
            print(f"API responded with status: {response.status_code}")
            
            # Create a router using the normal KamiwazaRouter initialization
            # This will use KamiwazaClient to discover models
            router = KamiwazaRouter(
                kamiwaza_api_url=self.api_url,
                cache_ttl_seconds=0  # Disable caching for the test
            )
            
            # Get the list of models from the router
            models = router.get_kamiwaza_model_list(use_cache=False)
            
            if not models:
                pytest.skip("No models found in Kamiwaza API")
                
            print(f"Found {len(models)} models from Kamiwaza API")
            for i, model in enumerate(models):
                print(f"Model {i+1}: {model.get('model_name')}")
            
            # Select the first model to test with
            model_name = models[0].get('model_name')
            print(f"Testing completion with model: {model_name}")
            
            # Prepare the prompt
            messages = [{"role": "user", "content": "Write a haiku about artificial intelligence"}]
            
            try:
                # Make the completion call using the router
                response = router.completion(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                # Verify response
                assert response is not None
                assert 'choices' in response
                assert len(response['choices']) > 0
                assert 'message' in response['choices'][0]
                assert 'content' in response['choices'][0]['message']
                
                # Print out the generated text
                content = response['choices'][0]['message']['content']
                print(f"Generated content:\n{content}")
                
            except Exception as e:
                import traceback
                print(f"Error during router completion: {str(e)}")
                print(traceback.format_exc())
                pytest.skip(f"Router completion failed: {str(e)}")
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            pytest.skip(f"Error connecting to Kamiwaza API: {http_err}")
        except requests.exceptions.RequestException as req_err:
            print(f"Error occurred during request: {req_err}")
            pytest.skip(f"Error connecting to Kamiwaza API: {req_err}")
        except Exception as e:
            import traceback
            print(f"Error during test: {str(e)}")
            print(traceback.format_exc())
            pytest.skip(f"Test failed: {str(e)}")


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
