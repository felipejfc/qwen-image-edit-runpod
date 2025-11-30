#!/usr/bin/env python3
"""Unit tests for job_store.py DynamoDB operations."""
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

# Mock config before importing job_store
with patch.dict('os.environ', {
    'AWS_ACCESS_KEY_ID': 'test-key',
    'AWS_SECRET_ACCESS_KEY': 'test-secret',
    'AWS_REGION': 'us-east-1',
    'DYNAMODB_JOBS_TABLE': 'test-jobs',
    'DYNAMODB_ENDPOINT_URL': '',
}):
    import config
    config.AWS_ACCESS_KEY_ID = 'test-key'
    config.AWS_SECRET_ACCESS_KEY = 'test-secret'
    config.AWS_REGION = 'us-east-1'
    config.DYNAMODB_JOBS_TABLE = 'test-jobs'
    config.DYNAMODB_ENDPOINT_URL = None
    
    import job_store
    from job_store import JobStatus


class TestJobStore(unittest.TestCase):
    """Test cases for DynamoDB job store operations."""

    def setUp(self):
        """Reset DynamoDB client singleton before each test."""
        job_store._dynamodb_client = None

    @patch('job_store.boto3.client')
    def test_get_dynamodb_client_creates_singleton(self, mock_boto_client):
        """Test that get_dynamodb_client creates a singleton client."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # First call should create client
        client1 = job_store.get_dynamodb_client()
        self.assertEqual(client1, mock_client)
        
        # Second call should return same client
        client2 = job_store.get_dynamodb_client()
        self.assertEqual(client1, client2)
        
        # boto3.client should only be called once
        mock_boto_client.assert_called_once()

    def test_parse_dynamodb_item_empty(self):
        """Test parsing empty DynamoDB item returns None."""
        result = job_store._parse_dynamodb_item(None)
        self.assertIsNone(result)
        
        result = job_store._parse_dynamodb_item({})
        self.assertIsNone(result)

    def test_parse_dynamodb_item_full(self):
        """Test parsing complete DynamoDB item."""
        created_ms = int(datetime(2024, 1, 15, 10, 30, 0).timestamp() * 1000)
        
        item = {
            'id': {'S': 'job-123'},
            'uid': {'S': 'user-456'},
            'type': {'S': 'image'},
            'status': {'S': 'queued'},
            'prompt': {'S': 'Make it blue'},
            'source_key': {'S': 'gen/user-456/source/image.jpg'},
            'model': {'S': 'runpod'},
            'style_code': {'S': 'STYLE123'},
            'style_name': {'S': 'Cool Style'},
            'nickname': {'S': 'john'},
            'output_key': {'S': 'gen/user-456/output/job-123.jpg'},
            'output_url': {'S': 'https://example.com/output.jpg'},
            'post_id': {'S': 'post-789'},
            'error_code': {'S': ''},
            'error_msg': {'S': ''},
            'attempts': {'N': '1'},
            'priority': {'N': '5'},
            'created_at': {'N': str(created_ms)},
        }
        
        result = job_store._parse_dynamodb_item(item)
        
        self.assertIsInstance(result, JobStatus)
        self.assertEqual(result.id, 'job-123')
        self.assertEqual(result.user_id, 'user-456')
        self.assertEqual(result.type, 'image')
        self.assertEqual(result.status, 'queued')
        self.assertEqual(result.prompt, 'Make it blue')
        self.assertEqual(result.source_key, 'gen/user-456/source/image.jpg')
        self.assertEqual(result.model, 'runpod')
        self.assertEqual(result.style_code, 'STYLE123')
        self.assertEqual(result.style_name, 'Cool Style')
        self.assertEqual(result.nickname, 'john')
        self.assertEqual(result.output_key, 'gen/user-456/output/job-123.jpg')
        self.assertEqual(result.output_url, 'https://example.com/output.jpg')
        self.assertEqual(result.post_id, 'post-789')
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.priority, 5)
        self.assertIsNotNone(result.created_at)

    def test_parse_dynamodb_item_defaults_model_to_gemini(self):
        """Test that missing model field defaults to 'gemini'."""
        item = {
            'id': {'S': 'job-123'},
            'uid': {'S': 'user-456'},
            'type': {'S': 'image'},
            'status': {'S': 'queued'},
            'prompt': {'S': 'Test'},
            'source_key': {'S': 'test-key'},
            # No 'model' field
        }
        
        result = job_store._parse_dynamodb_item(item)
        
        self.assertEqual(result.model, 'gemini')

    def test_parse_dynamodb_item_optional_fields_none(self):
        """Test that empty optional fields are set to None."""
        item = {
            'id': {'S': 'job-123'},
            'uid': {'S': 'user-456'},
            'type': {'S': 'image'},
            'status': {'S': 'queued'},
            'prompt': {'S': 'Test'},
            'source_key': {'S': 'test-key'},
            'model': {'S': 'runpod'},
            'style_code': {'S': ''},  # Empty string
            'error_code': {'S': ''},
        }
        
        result = job_store._parse_dynamodb_item(item)
        
        self.assertIsNone(result.style_code)
        self.assertIsNone(result.error_code)

    @patch('job_store.boto3.client')
    def test_get_job_success(self, mock_boto_client):
        """Test getting a job from DynamoDB."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        mock_client.get_item.return_value = {
            'Item': {
                'id': {'S': 'job-123'},
                'uid': {'S': 'user-456'},
                'type': {'S': 'image'},
                'status': {'S': 'queued'},
                'prompt': {'S': 'Make it blue'},
                'source_key': {'S': 'gen/user-456/source/image.jpg'},
                'model': {'S': 'runpod'},
            }
        }
        
        result = job_store.get_job('job-123')
        
        self.assertIsNotNone(result)
        self.assertEqual(result.id, 'job-123')
        self.assertEqual(result.status, 'queued')
        
        mock_client.get_item.assert_called_once_with(
            TableName='test-jobs',
            Key={
                'pk': {'S': 'J#job-123'},
                'sk': {'S': 'JOB'},
            }
        )

    @patch('job_store.boto3.client')
    def test_get_job_not_found(self, mock_boto_client):
        """Test getting a non-existent job returns None."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        mock_client.get_item.return_value = {}  # No 'Item' key
        
        result = job_store.get_job('non-existent')
        
        self.assertIsNone(result)

    @patch('job_store.get_job')
    @patch('job_store.boto3.client')
    def test_update_job_success(self, mock_boto_client, mock_get_job):
        """Test updating a job in DynamoDB."""
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        
        # Mock existing job
        existing_job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test prompt',
            source_key='test-key',
            model='runpod',
            created_at=datetime.now(timezone.utc),
        )
        mock_get_job.return_value = existing_job
        
        # Update the job
        result = job_store.update_job(
            'job-123',
            status='running',
            extend_ttl_minutes=30
        )
        
        self.assertTrue(result)
        mock_client.put_item.assert_called_once()
        
        # Verify the item being put
        put_call_args = mock_client.put_item.call_args
        item = put_call_args.kwargs['Item']
        self.assertEqual(item['status']['S'], 'running')

    @patch('job_store.get_job')
    def test_update_job_not_found(self, mock_get_job):
        """Test updating a non-existent job returns False."""
        mock_get_job.return_value = None
        
        result = job_store.update_job('non-existent', status='running')
        
        self.assertFalse(result)

    @patch('job_store.update_job')
    def test_update_job_running(self, mock_update):
        """Test update_job_running helper."""
        mock_update.return_value = True
        
        result = job_store.update_job_running('job-123')
        
        self.assertTrue(result)
        mock_update.assert_called_once_with(
            'job-123',
            status='running',
            extend_ttl_minutes=30
        )

    @patch('job_store.update_job')
    def test_update_job_succeeded(self, mock_update):
        """Test update_job_succeeded helper."""
        mock_update.return_value = True
        
        result = job_store.update_job_succeeded(
            'job-123',
            'gen/user/output/job.jpg',
            'https://example.com/output.jpg'
        )
        
        self.assertTrue(result)
        mock_update.assert_called_once_with(
            'job-123',
            status='succeeded',
            output_key='gen/user/output/job.jpg',
            output_url='https://example.com/output.jpg',
            extend_ttl_minutes=60
        )

    @patch('job_store.update_job')
    def test_update_job_failed(self, mock_update):
        """Test update_job_failed helper."""
        mock_update.return_value = True
        
        result = job_store.update_job_failed(
            'job-123',
            'generation_failed',
            'Model inference error'
        )
        
        self.assertTrue(result)
        mock_update.assert_called_once_with(
            'job-123',
            status='failed',
            error_code='generation_failed',
            error_msg='Model inference error',
            extend_ttl_minutes=60
        )

    @patch('job_store.update_job')
    def test_update_job_canceled(self, mock_update):
        """Test update_job_canceled helper."""
        mock_update.return_value = True
        
        result = job_store.update_job_canceled(
            'job-123',
            'stale_job',
            'Job older than 60s'
        )
        
        self.assertTrue(result)
        mock_update.assert_called_once_with(
            'job-123',
            status='canceled',
            error_code='stale_job',
            error_msg='Job older than 60s',
            extend_ttl_minutes=10
        )


if __name__ == '__main__':
    unittest.main()

