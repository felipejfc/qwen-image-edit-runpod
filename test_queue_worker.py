#!/usr/bin/env python3
"""Unit tests for queue_worker.py pipeline architecture."""
import base64
import json
import queue
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

# Mock config before importing queue_worker
with patch.dict('os.environ', {
    'REDIS_URL': 'redis://localhost:6379',
    'REDIS_QUEUE_NAME': 'queue:runpod_generation',
    'REDIS_BRPOP_TIMEOUT': '5',
    'WORKER_CONCURRENCY': '1',
    'STALE_JOB_TIMEOUT': '60',
    'PREFETCH_DEPTH': '2',
    'READY_QUEUE_SIZE': '3',
    'RESULT_QUEUE_SIZE': '5',
    'INTERNAL_QUEUE_TIMEOUT': '0.1',
    'AWS_ACCESS_KEY_ID': 'test',
    'AWS_SECRET_ACCESS_KEY': 'test',
    'AWS_REGION': 'us-east-1',
    'DYNAMODB_JOBS_TABLE': 'test-jobs',
    'BUCKET_NAME': 'test-bucket',
}):
    import config
    config.REDIS_URL = 'redis://localhost:6379'
    config.REDIS_QUEUE_NAME = 'queue:runpod_generation'
    config.REDIS_BRPOP_TIMEOUT = 5
    config.STALE_JOB_TIMEOUT_SECONDS = 60
    config.PREFETCH_DEPTH = 2
    config.READY_QUEUE_SIZE = 3
    config.RESULT_QUEUE_SIZE = 5
    config.INTERNAL_QUEUE_TIMEOUT = 0.1
    
    import queue_worker
    from queue_worker import (
        PreparedJob,
        CompletedJob,
        is_job_stale,
        format_prompt,
        publish_job_status,
        process_job,
    )
    from job_store import JobStatus


class TestDataClasses(unittest.TestCase):
    """Test data classes used in the pipeline."""

    def test_prepared_job_creation(self):
        """Test creating a PreparedJob instance."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test prompt',
            source_key='test-key',
            model='runpod'
        )
        prepared = PreparedJob(
            job=job,
            image_bytes=b'fake-image-data',
            content_type='image/jpeg'
        )
        
        self.assertEqual(prepared.job.id, 'job-123')
        self.assertEqual(prepared.image_bytes, b'fake-image-data')
        self.assertEqual(prepared.content_type, 'image/jpeg')

    def test_completed_job_success(self):
        """Test creating a successful CompletedJob."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test',
            source_key='test-key',
            model='runpod'
        )
        completed = CompletedJob(
            job=job,
            result_bytes=b'result-data',
            success=True
        )
        
        self.assertTrue(completed.success)
        self.assertEqual(completed.result_bytes, b'result-data')
        self.assertIsNone(completed.error_code)

    def test_completed_job_failure(self):
        """Test creating a failed CompletedJob."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test',
            source_key='test-key',
            model='runpod'
        )
        completed = CompletedJob(
            job=job,
            result_bytes=b'',
            success=False,
            error_code='generation_failed',
            error_msg='GPU error'
        )
        
        self.assertFalse(completed.success)
        self.assertEqual(completed.error_code, 'generation_failed')
        self.assertEqual(completed.error_msg, 'GPU error')


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions."""

    def test_is_job_stale_no_created_at(self):
        """Test is_job_stale returns False when created_at is None."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=None
        )
        
        self.assertFalse(is_job_stale(job))

    def test_is_job_stale_fresh_job(self):
        """Test is_job_stale returns False for fresh jobs."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=datetime.now(timezone.utc)
        )
        
        self.assertFalse(is_job_stale(job))

    def test_is_job_stale_old_job(self):
        """Test is_job_stale returns True for old jobs."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=old_time
        )
        
        self.assertTrue(is_job_stale(job))

    def test_format_prompt_no_style(self):
        """Test format_prompt without style code."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Make it blue',
            source_key='test-key',
            model='runpod',
            style_code=None
        )
        
        result = format_prompt(job)
        self.assertEqual(result, 'Make it blue')

    def test_format_prompt_with_style(self):
        """Test format_prompt with style code."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Casual summer',
            source_key='test-key',
            model='runpod',
            style_code='SUMMER123'
        )
        
        result = format_prompt(job)
        self.assertIn('Change the clothing style to:', result)
        self.assertIn('Casual summer', result)


class TestRedisOperations(unittest.TestCase):
    """Test Redis client and pub/sub operations."""

    def setUp(self):
        """Reset Redis client singletons before each test."""
        queue_worker._redis_client = None
        queue_worker._redis_pubsub_client = None
        queue_worker._shutdown_event.clear()

    @patch('queue_worker.redis.from_url')
    def test_get_redis_client_creates_singleton(self, mock_from_url):
        """Test that get_redis_client creates a singleton client."""
        mock_client = MagicMock()
        mock_from_url.return_value = mock_client
        
        client1 = queue_worker.get_redis_client()
        client2 = queue_worker.get_redis_client()
        
        self.assertEqual(client1, client2)
        mock_from_url.assert_called_once()

    @patch('queue_worker.get_redis_pubsub_client')
    def test_publish_job_status_running(self, mock_get_client):
        """Test publishing running status update."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        publish_job_status('job-123', 'running')
        
        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        channel = call_args[0][0]
        payload = json.loads(call_args[0][1])
        
        self.assertEqual(channel, 'job:job-123')
        self.assertEqual(payload['status'], 'running')

    @patch('queue_worker.get_redis_pubsub_client')
    def test_publish_job_status_succeeded(self, mock_get_client):
        """Test publishing succeeded status with output URL."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        publish_job_status('job-123', 'succeeded', output_url='https://example.com/output.jpg')
        
        call_args = mock_client.publish.call_args
        payload = json.loads(call_args[0][1])
        
        self.assertEqual(payload['status'], 'succeeded')
        self.assertEqual(payload['output_url'], 'https://example.com/output.jpg')

    @patch('queue_worker.get_redis_pubsub_client')
    def test_publish_job_status_handles_error(self, mock_get_client):
        """Test that publish_job_status handles errors gracefully."""
        mock_client = MagicMock()
        mock_client.publish.side_effect = Exception('Redis error')
        mock_get_client.return_value = mock_client
        
        # Should not raise
        publish_job_status('job-123', 'running')


class TestLegacyProcessJob(unittest.TestCase):
    """Test legacy single-threaded process_job function."""

    def setUp(self):
        queue_worker._shutdown_event.clear()

    @patch('queue_worker.upload_image')
    @patch('queue_worker.download_image')
    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_succeeded')
    @patch('queue_worker.update_job_running')
    @patch('queue_worker.get_job')
    def test_process_job_success(
        self,
        mock_get_job,
        mock_update_running,
        mock_update_succeeded,
        mock_publish,
        mock_download,
        mock_upload
    ):
        """Test successful job processing."""
        mock_job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Make it blue',
            source_key='gen/user-456/source/image.jpg',
            model='runpod',
            created_at=datetime.now(timezone.utc),
        )
        mock_get_job.return_value = mock_job
        mock_download.return_value = (b'fake-image-bytes', 'image/jpeg')
        mock_upload.return_value = 'https://example.com/output.jpg'
        
        result_image = base64.b64encode(b'result-image').decode('utf-8')
        mock_handler = MagicMock(return_value={'image': result_image})
        
        success = process_job(mock_handler, 'job-123')
        
        self.assertTrue(success)
        mock_get_job.assert_called_once_with('job-123')
        mock_update_running.assert_called_once()
        mock_update_succeeded.assert_called_once()

    @patch('queue_worker.get_job')
    def test_process_job_not_found(self, mock_get_job):
        """Test processing a job that doesn't exist."""
        mock_get_job.return_value = None
        mock_handler = MagicMock()
        
        success = process_job(mock_handler, 'non-existent')
        
        self.assertFalse(success)
        mock_handler.assert_not_called()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_canceled')
    @patch('queue_worker.get_job')
    def test_process_job_stale(self, mock_get_job, mock_update_canceled, mock_publish):
        """Test processing a stale job."""
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        mock_job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=old_time,
        )
        mock_get_job.return_value = mock_job
        mock_handler = MagicMock()
        
        success = process_job(mock_handler, 'job-123')
        
        self.assertFalse(success)
        mock_update_canceled.assert_called_once()
        mock_handler.assert_not_called()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_failed')
    @patch('queue_worker.download_image')
    @patch('queue_worker.update_job_running')
    @patch('queue_worker.get_job')
    def test_process_job_download_failure(
        self,
        mock_get_job,
        mock_update_running,
        mock_download,
        mock_update_failed,
        mock_publish
    ):
        """Test handling download failure."""
        mock_job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=datetime.now(timezone.utc),
        )
        mock_get_job.return_value = mock_job
        mock_download.side_effect = Exception('S3 error')
        mock_handler = MagicMock()
        
        success = process_job(mock_handler, 'job-123')
        
        self.assertFalse(success)
        mock_update_failed.assert_called_once()
        call_args = mock_update_failed.call_args
        self.assertEqual(call_args[0][1], 'download_failed')


class TestPrefetchWorker(unittest.TestCase):
    """Test the prefetch worker thread."""

    def setUp(self):
        queue_worker._redis_client = None
        queue_worker._shutdown_event.clear()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_running')
    @patch('queue_worker.download_image')
    @patch('queue_worker.get_job')
    @patch('queue_worker.get_redis_client')
    def test_prefetch_worker_fetches_job(
        self,
        mock_get_redis,
        mock_get_job,
        mock_download,
        mock_update_running,
        mock_publish
    ):
        """Test that prefetch worker fetches and prepares jobs."""
        # Setup mocks
        mock_redis = MagicMock()
        mock_redis.brpop.side_effect = [
            (queue_worker.REDIS_QUEUE_NAME, 'job-123'),
            None,  # Second call returns None
        ]
        mock_get_redis.return_value = mock_redis
        
        mock_job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='queued',
            prompt='Test',
            source_key='test-key',
            model='runpod',
            created_at=datetime.now(timezone.utc),
        )
        mock_get_job.return_value = mock_job
        mock_download.return_value = (b'image-data', 'image/jpeg')
        
        ready_queue = queue.Queue(maxsize=3)
        
        # Run prefetch worker in a thread, stop after short time
        def run_prefetch():
            queue_worker.prefetch_worker(ready_queue, worker_id=0)
        
        # Set shutdown after a short delay
        def trigger_shutdown():
            time.sleep(0.3)
            queue_worker._shutdown_event.set()
        
        shutdown_thread = threading.Thread(target=trigger_shutdown)
        shutdown_thread.start()
        
        prefetch_thread = threading.Thread(target=run_prefetch)
        prefetch_thread.start()
        prefetch_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=1.0)
        
        # Verify job was prepared
        self.assertFalse(ready_queue.empty())
        prepared = ready_queue.get_nowait()
        self.assertEqual(prepared.job.id, 'job-123')
        self.assertEqual(prepared.image_bytes, b'image-data')


class TestInferenceWorker(unittest.TestCase):
    """Test the inference worker thread."""

    def setUp(self):
        queue_worker._shutdown_event.clear()

    def test_inference_worker_processes_job(self):
        """Test that inference worker processes prepared jobs."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test prompt',
            source_key='test-key',
            model='runpod',
        )
        prepared = PreparedJob(
            job=job,
            image_bytes=b'fake-image',
            content_type='image/jpeg'
        )
        
        ready_queue = queue.Queue()
        result_queue = queue.Queue()
        ready_queue.put(prepared)
        
        result_image = base64.b64encode(b'result-image').decode('utf-8')
        mock_handler = MagicMock(return_value={'image': result_image})
        
        # Run inference worker in a thread
        def run_inference():
            queue_worker.inference_worker(mock_handler, ready_queue, result_queue, worker_id=0)
        
        def trigger_shutdown():
            time.sleep(0.3)
            queue_worker._shutdown_event.set()
        
        shutdown_thread = threading.Thread(target=trigger_shutdown)
        shutdown_thread.start()
        
        inference_thread = threading.Thread(target=run_inference)
        inference_thread.start()
        inference_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=1.0)
        
        # Verify result was produced
        self.assertFalse(result_queue.empty())
        completed = result_queue.get_nowait()
        self.assertTrue(completed.success)
        self.assertEqual(completed.job.id, 'job-123')


class TestUploadWorker(unittest.TestCase):
    """Test the upload worker thread."""

    def setUp(self):
        queue_worker._shutdown_event.clear()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_succeeded')
    @patch('queue_worker.upload_image')
    def test_upload_worker_uploads_result(
        self,
        mock_upload,
        mock_update_succeeded,
        mock_publish
    ):
        """Test that upload worker uploads successful results."""
        mock_upload.return_value = 'https://example.com/output.jpg'
        
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test',
            source_key='test-key',
            model='runpod',
        )
        completed = CompletedJob(
            job=job,
            result_bytes=b'result-image',
            success=True
        )
        
        result_queue = queue.Queue()
        result_queue.put(completed)
        
        def run_upload():
            queue_worker.upload_worker(result_queue, worker_id=0)
        
        def trigger_shutdown():
            time.sleep(0.3)
            queue_worker._shutdown_event.set()
        
        shutdown_thread = threading.Thread(target=trigger_shutdown)
        shutdown_thread.start()
        
        upload_thread = threading.Thread(target=run_upload)
        upload_thread.start()
        upload_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=1.0)
        
        # Verify upload was called
        mock_upload.assert_called_once()
        mock_update_succeeded.assert_called_once()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_failed')
    def test_upload_worker_handles_failure(
        self,
        mock_update_failed,
        mock_publish
    ):
        """Test that upload worker handles failed jobs."""
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test',
            source_key='test-key',
            model='runpod',
        )
        completed = CompletedJob(
            job=job,
            result_bytes=b'',
            success=False,
            error_code='generation_failed',
            error_msg='GPU error'
        )
        
        result_queue = queue.Queue()
        result_queue.put(completed)
        
        def run_upload():
            queue_worker.upload_worker(result_queue, worker_id=0)
        
        def trigger_shutdown():
            time.sleep(0.3)
            queue_worker._shutdown_event.set()
        
        shutdown_thread = threading.Thread(target=trigger_shutdown)
        shutdown_thread.start()
        
        upload_thread = threading.Thread(target=run_upload)
        upload_thread.start()
        upload_thread.join(timeout=2.0)
        shutdown_thread.join(timeout=1.0)
        
        # Verify failure was recorded
        mock_update_failed.assert_called_once()


class TestDrainQueues(unittest.TestCase):
    """Test graceful shutdown queue draining."""

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_succeeded')
    @patch('queue_worker.upload_image')
    def test_drain_result_queue(
        self,
        mock_upload,
        mock_update_succeeded,
        mock_publish
    ):
        """Test draining the result queue during shutdown."""
        mock_upload.return_value = 'https://example.com/output.jpg'
        
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test',
            source_key='test-key',
            model='runpod',
        )
        completed = CompletedJob(
            job=job,
            result_bytes=b'result-image',
            success=True
        )
        
        ready_queue = queue.Queue()
        result_queue = queue.Queue()
        result_queue.put(completed)
        
        mock_handler = MagicMock()
        
        queue_worker.drain_queues(ready_queue, result_queue, mock_handler, timeout_seconds=5.0)
        
        # Verify result was uploaded
        mock_upload.assert_called_once()
        mock_update_succeeded.assert_called_once()

    @patch('queue_worker.publish_job_status')
    @patch('queue_worker.update_job_succeeded')
    @patch('queue_worker.upload_image')
    def test_drain_ready_queue(
        self,
        mock_upload,
        mock_update_succeeded,
        mock_publish
    ):
        """Test draining the ready queue during shutdown."""
        mock_upload.return_value = 'https://example.com/output.jpg'
        
        job = JobStatus(
            id='job-123',
            user_id='user-456',
            type='image',
            status='running',
            prompt='Test prompt',
            source_key='test-key',
            model='runpod',
        )
        prepared = PreparedJob(
            job=job,
            image_bytes=b'fake-image',
            content_type='image/jpeg'
        )
        
        ready_queue = queue.Queue()
        result_queue = queue.Queue()
        ready_queue.put(prepared)
        
        result_image = base64.b64encode(b'result-image').decode('utf-8')
        mock_handler = MagicMock(return_value={'image': result_image})
        
        queue_worker.drain_queues(ready_queue, result_queue, mock_handler, timeout_seconds=5.0)
        
        # Verify job was processed and uploaded
        mock_handler.assert_called_once()
        mock_upload.assert_called_once()


class TestQueueDepth(unittest.TestCase):
    """Test queue depth monitoring."""

    @patch('queue_worker.get_redis_client')
    def test_get_queue_depth(self, mock_get_client):
        """Test getting queue depth."""
        mock_client = MagicMock()
        mock_client.llen.return_value = 42
        mock_get_client.return_value = mock_client
        
        depth = queue_worker.get_queue_depth()
        
        self.assertEqual(depth, 42)

    @patch('queue_worker.get_redis_client')
    def test_get_queue_depth_error(self, mock_get_client):
        """Test get_queue_depth returns -1 on error."""
        mock_client = MagicMock()
        mock_client.llen.side_effect = Exception('Redis error')
        mock_get_client.return_value = mock_client
        
        depth = queue_worker.get_queue_depth()
        
        self.assertEqual(depth, -1)


if __name__ == '__main__':
    unittest.main()
