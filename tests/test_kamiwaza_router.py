import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from litellm_kamiwaza import KamiwazaRouter

class TestKamiwazaRouter(unittest.TestCase):
    
    @patch('litellm_kamiwaza.kamiwaza_router.KamiwazaClient')
    def test_initialization(self, mock_kamiwaza_client):
        """Test basic initialization of the router."""
        # Setup mock
        mock_instance = MagicMock()
        mock_kamiwaza_client.return_value = mock_instance
        mock_instance.serving.list_deployments.return_value = []
        
        # Test with API URL
        router = KamiwazaRouter(kamiwaza_api_url="http://test-url")
        self.assertIsNotNone(router)
        self.assertTrue(hasattr(router, 'kamiwaza_client'))
        
        # Test with no source
        with self.assertRaises(ValueError):
            KamiwazaRouter(kamiwaza_api_url=None, kamiwaza_uri_list=None)
    
    @patch('litellm_kamiwaza.kamiwaza_router.get_static_model_configs')
    @patch('litellm_kamiwaza.kamiwaza_router.KamiwazaClient')
    def test_model_pattern_filtering(self, mock_kamiwaza_client, mock_get_static_model_configs):
        """Test that model pattern filtering works correctly."""
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
            model_pattern="xyz"
        )
        
        # Should find no models
        models = router.get_kamiwaza_model_list(use_cache=False)
        self.assertEqual(len(models), 0)

if __name__ == '__main__':
    unittest.main()
