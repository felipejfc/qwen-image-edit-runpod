#!/usr/bin/env python3
"""Unit tests for storage.py S3 operations."""
import io
import unittest
from unittest.mock import MagicMock, patch

# Mock config before importing storage
with patch.dict('os.environ', {
    'AWS_ACCESS_KEY_ID': 'test-key',
    'AWS_SECRET_ACCESS_KEY': 'test-secret',
    'AWS_REGION': 'us-east-1',
    'BUCKET_NAME': 'test-bucket',
    'S3_ENDPOINT_URL': '',
    'S3_PUBLIC_URL': '',
}):
    # Need to reload config to pick up test values
    import config
    config.AWS_ACCESS_KEY_ID = 'test-key'
    config.AWS_SECRET_ACCESS_KEY = 'test-secret'
    config.AWS_REGION = 'us-east-1'
    config.S3_BUCKET_NAME = 'test-bucket'
    config.S3_ENDPOINT_URL = None
    config.S3_PUBLIC_URL = ''
    
    import storage


class TestStorage(unittest.TestCase):
    """Test cases for S3 storage operations."""

    def setUp(self):
        """Reset S3 client singleton before each test."""
        storage._s3_client = None

    @patch('storage.boto3.client')
    def test_get_s3_client_creates_singleton(self, mock_boto_client):
        """Test that get_s3_client creates a singleton client."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # First call should create client
        client1 = storage.get_s3_client()
        self.assertEqual(client1, mock_client)
        
        # Second call should return same client
        client2 = storage.get_s3_client()
        self.assertEqual(client1, client2)
        
        # boto3.client should only be called once
        mock_boto_client.assert_called_once()

    @patch('storage.boto3.client')
    def test_download_image_success(self, mock_boto_client):
        """Test successful image download from S3."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # Mock S3 response
        test_image_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\x00'
        mock_body = MagicMock()
        mock_body.read.return_value = test_image_bytes
        mock_client.get_object.return_value = {
            'Body': mock_body,
            'ContentType': 'image/png',
        }
        
        # Call download_image
        image_bytes, content_type = storage.download_image('gen/user-123/source/image.png')
        
        # Verify
        self.assertEqual(image_bytes, test_image_bytes)
        self.assertEqual(content_type, 'image/png')
        mock_client.get_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='gen/user-123/source/image.png'
        )

    @patch('storage.boto3.client')
    def test_download_image_default_content_type(self, mock_boto_client):
        """Test that download_image defaults to image/jpeg when ContentType is missing."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        mock_body = MagicMock()
        mock_body.read.return_value = b'test-bytes'
        mock_client.get_object.return_value = {
            'Body': mock_body,
            # No ContentType
        }
        
        _, content_type = storage.download_image('test-key')
        
        self.assertEqual(content_type, 'image/jpeg')

    @patch('storage.boto3.client')
    def test_upload_image_success(self, mock_boto_client):
        """Test successful image upload to S3."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        test_image_bytes = b'\xff\xd8\xff\xe0\x00\x10JFIF'
        output_key = 'gen/user-123/output/job-456.jpg'
        
        # Call upload_image
        url = storage.upload_image(output_key, test_image_bytes, 'image/jpeg')
        
        # Verify S3 put was called
        call_args = mock_client.put_object.call_args
        self.assertEqual(call_args.kwargs['Bucket'], 'test-bucket')
        self.assertEqual(call_args.kwargs['Key'], output_key)
        self.assertEqual(call_args.kwargs['ContentType'], 'image/jpeg')
        
        # Verify URL was returned
        self.assertIn(output_key, url)

    def test_build_public_url_default(self):
        """Test building public URL with default S3 URL structure."""
        # Reset public URL config
        storage.S3_PUBLIC_URL = ''
        storage.S3_ENDPOINT_URL = None
        storage.S3_BUCKET_NAME = 'test-bucket'
        storage.AWS_REGION = 'us-east-1'
        
        url = storage.build_public_url('gen/user-123/output/job.jpg')
        
        expected = 'https://test-bucket.s3.us-east-1.amazonaws.com/gen/user-123/output/job.jpg'
        self.assertEqual(url, expected)

    def test_build_public_url_with_custom_url(self):
        """Test building public URL with custom S3_PUBLIC_URL."""
        storage.S3_PUBLIC_URL = 'https://cdn.example.com'
        
        url = storage.build_public_url('gen/user-123/output/job.jpg')
        
        self.assertEqual(url, 'https://cdn.example.com/gen/user-123/output/job.jpg')
        
        # Reset
        storage.S3_PUBLIC_URL = ''

    def test_build_public_url_with_endpoint_url(self):
        """Test building public URL with custom endpoint (e.g., R2/MinIO)."""
        storage.S3_PUBLIC_URL = ''
        storage.S3_ENDPOINT_URL = 'https://account.r2.cloudflarestorage.com'
        storage.S3_BUCKET_NAME = 'test-bucket'
        
        url = storage.build_public_url('gen/user-123/output/job.jpg')
        
        expected = 'https://account.r2.cloudflarestorage.com/test-bucket/gen/user-123/output/job.jpg'
        self.assertEqual(url, expected)
        
        # Reset
        storage.S3_ENDPOINT_URL = None

    @patch('storage.boto3.client')
    def test_object_exists_true(self, mock_boto_client):
        """Test object_exists returns True when object exists."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.head_object.return_value = {}
        
        result = storage.object_exists('existing-key')
        
        self.assertTrue(result)
        mock_client.head_object.assert_called_once_with(
            Bucket='test-bucket',
            Key='existing-key'
        )

    @patch('storage.boto3.client')
    def test_object_exists_false(self, mock_boto_client):
        """Test object_exists returns False when object doesn't exist."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # Create a mock ClientError
        mock_client.exceptions.ClientError = Exception
        mock_client.head_object.side_effect = Exception('Not Found')
        
        result = storage.object_exists('non-existing-key')
        
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()

