"""S3 storage operations for downloading source images and uploading results."""
import io
from typing import Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig

from config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_PUBLIC_URL,
)

_s3_client = None


def get_s3_client():
    """Get or create the S3 client singleton."""
    global _s3_client
    if _s3_client is None:
        config = BotoConfig(
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=30,
        )
        kwargs = {
            "service_name": "s3",
            "region_name": AWS_REGION,
            "config": config,
        }
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = S3_ENDPOINT_URL
        _s3_client = boto3.client(**kwargs)
    return _s3_client


def download_image(key: str) -> Tuple[bytes, str]:
    """
    Download an image from S3.
    
    Args:
        key: The S3 object key
        
    Returns:
        Tuple of (image_bytes, content_type)
    """
    client = get_s3_client()
    response = client.get_object(Bucket=S3_BUCKET_NAME, Key=key)
    content_type = response.get("ContentType", "image/jpeg")
    image_bytes = response["Body"].read()
    return image_bytes, content_type


def upload_image(key: str, image_bytes: bytes, content_type: str = "image/jpeg") -> str:
    """
    Upload an image to S3.
    
    Args:
        key: The S3 object key
        image_bytes: The image data
        content_type: MIME type of the image
        
    Returns:
        The public URL of the uploaded image
    """
    client = get_s3_client()
    client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=io.BytesIO(image_bytes),
        ContentType=content_type,
    )
    return build_public_url(key)


def build_public_url(key: str) -> str:
    """Build the public URL for an S3 object."""
    if S3_PUBLIC_URL:
        base = S3_PUBLIC_URL.rstrip("/")
        return f"{base}/{key}"
    if S3_ENDPOINT_URL:
        base = S3_ENDPOINT_URL.rstrip("/")
        return f"{base}/{S3_BUCKET_NAME}/{key}"
    return f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"


def object_exists(key: str) -> bool:
    """Check if an object exists in S3."""
    client = get_s3_client()
    try:
        client.head_object(Bucket=S3_BUCKET_NAME, Key=key)
        return True
    except client.exceptions.ClientError:
        return False

