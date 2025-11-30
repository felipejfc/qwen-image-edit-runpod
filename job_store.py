"""DynamoDB job store operations for reading and updating job status."""
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config as BotoConfig

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    DYNAMODB_JOBS_TABLE,
    DYNAMODB_ENDPOINT_URL,
)

_dynamodb_client = None


def get_dynamodb_client():
    """Get or create the DynamoDB client singleton."""
    global _dynamodb_client
    if _dynamodb_client is None:
        config = BotoConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        )
        kwargs = {
            "service_name": "dynamodb",
            "region_name": AWS_REGION,
            "config": config,
        }
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if DYNAMODB_ENDPOINT_URL:
            kwargs["endpoint_url"] = DYNAMODB_ENDPOINT_URL
        _dynamodb_client = boto3.client(**kwargs)
    return _dynamodb_client


@dataclass
class JobStatus:
    """Job status data class matching the Go backend structure."""
    id: str
    user_id: str
    type: str
    status: str
    prompt: str
    source_key: str
    model: str
    style_code: Optional[str] = None
    style_name: Optional[str] = None
    nickname: Optional[str] = None
    output_key: Optional[str] = None
    output_url: Optional[str] = None
    post_id: Optional[str] = None
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    attempts: int = 0
    priority: int = 0
    created_at: Optional[datetime] = None


def _parse_dynamodb_item(item: Dict[str, Any]) -> Optional[JobStatus]:
    """Parse a DynamoDB item into a JobStatus object."""
    if not item:
        return None
    
    def get_s(key: str) -> str:
        val = item.get(key, {})
        if isinstance(val, dict):
            return val.get("S", "")
        return ""
    
    def get_n(key: str) -> int:
        val = item.get(key, {})
        if isinstance(val, dict):
            n = val.get("N", "0")
            return int(n) if n else 0
        return 0
    
    created_at = None
    created_ms = get_n("created_at")
    if created_ms > 0:
        created_at = datetime.fromtimestamp(created_ms / 1000.0, tz=timezone.utc)
    
    model = get_s("model")
    if not model:
        model = "gemini"  # default for backward compatibility
    
    return JobStatus(
        id=get_s("id"),
        user_id=get_s("uid"),
        type=get_s("type"),
        status=get_s("status"),
        prompt=get_s("prompt"),
        source_key=get_s("source_key"),
        model=model,
        style_code=get_s("style_code") or None,
        style_name=get_s("style_name") or None,
        nickname=get_s("nickname") or None,
        output_key=get_s("output_key") or None,
        output_url=get_s("output_url") or None,
        post_id=get_s("post_id") or None,
        error_code=get_s("error_code") or None,
        error_msg=get_s("error_msg") or None,
        attempts=get_n("attempts"),
        priority=get_n("priority"),
        created_at=created_at,
    )


def get_job(job_id: str) -> Optional[JobStatus]:
    """
    Get a job by ID from DynamoDB.
    
    Args:
        job_id: The job UUID string
        
    Returns:
        JobStatus if found, None otherwise
    """
    client = get_dynamodb_client()
    response = client.get_item(
        TableName=DYNAMODB_JOBS_TABLE,
        Key={
            "pk": {"S": f"J#{job_id}"},
            "sk": {"S": "JOB"},
        },
    )
    return _parse_dynamodb_item(response.get("Item"))


def update_job(
    job_id: str,
    status: Optional[str] = None,
    output_key: Optional[str] = None,
    output_url: Optional[str] = None,
    error_code: Optional[str] = None,
    error_msg: Optional[str] = None,
    attempts: Optional[int] = None,
    extend_ttl_minutes: int = 60,
) -> bool:
    """
    Update a job in DynamoDB.
    
    This performs a read-modify-write to match the Go backend's behavior.
    
    Args:
        job_id: The job UUID string
        status: New status value
        output_key: S3 key of the output image
        output_url: Public URL of the output image
        error_code: Error code if failed
        error_msg: Error message if failed
        attempts: Number of processing attempts
        extend_ttl_minutes: Minutes to extend the TTL
        
    Returns:
        True if update succeeded, False otherwise
    """
    # Get existing job
    job = get_job(job_id)
    if not job:
        return False
    
    # Apply updates
    if status is not None:
        job.status = status
    if output_key is not None:
        job.output_key = output_key
    if output_url is not None:
        job.output_url = output_url
    if error_code is not None:
        job.error_code = error_code
    if error_msg is not None:
        job.error_msg = error_msg
    if attempts is not None:
        job.attempts = attempts
    
    # Calculate new TTL
    ttl_sec = int(time.time()) + (extend_ttl_minutes * 60)
    
    # Build the item
    item = {
        "pk": {"S": f"J#{job.id}"},
        "sk": {"S": "JOB"},
        "id": {"S": job.id},
        "uid": {"S": job.user_id},
        "type": {"S": job.type},
        "status": {"S": job.status},
        "prompt": {"S": job.prompt},
        "style_code": {"S": job.style_code or ""},
        "style_name": {"S": job.style_name or ""},
        "nickname": {"S": job.nickname or ""},
        "source_key": {"S": job.source_key},
        "output_key": {"S": job.output_key or ""},
        "output_url": {"S": job.output_url or ""},
        "post_id": {"S": job.post_id or ""},
        "error_code": {"S": job.error_code or ""},
        "error_msg": {"S": job.error_msg or ""},
        "model": {"S": job.model or "gemini"},
        "attempts": {"N": str(job.attempts)},
        "priority": {"N": str(job.priority)},
        "expires_at": {"N": str(ttl_sec)},
    }
    
    # Preserve created_at if it exists
    if job.created_at:
        item["created_at"] = {"N": str(int(job.created_at.timestamp() * 1000))}
    
    client = get_dynamodb_client()
    client.put_item(TableName=DYNAMODB_JOBS_TABLE, Item=item)
    return True


def update_job_running(job_id: str) -> bool:
    """Mark a job as running."""
    return update_job(job_id, status="running", extend_ttl_minutes=30)


def update_job_succeeded(job_id: str, output_key: str, output_url: str) -> bool:
    """Mark a job as succeeded with output URL."""
    return update_job(
        job_id,
        status="succeeded",
        output_key=output_key,
        output_url=output_url,
        extend_ttl_minutes=60,
    )


def update_job_failed(job_id: str, error_code: str, error_msg: str) -> bool:
    """Mark a job as failed with error details."""
    return update_job(
        job_id,
        status="failed",
        error_code=error_code,
        error_msg=error_msg,
        extend_ttl_minutes=60,
    )


def update_job_canceled(job_id: str, error_code: str, error_msg: str) -> bool:
    """Mark a job as canceled."""
    return update_job(
        job_id,
        status="canceled",
        error_code=error_code,
        error_msg=error_msg,
        extend_ttl_minutes=10,
    )

