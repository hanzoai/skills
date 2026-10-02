---
name: hanzo-storage
description: "S3-compatible object storage with Hanzo S3. Upload, download, and manage files, model artifacts, and datasets with erasure coding, encryption, and lifecycle rules."
metadata:
  {
    "bot":
      {
        "requires": { "bins": ["python3"] },
        "install":
          [
            {
              "id": "pip",
              "kind": "pip",
              "package": "boto3",
              "label": "Install AWS SDK for Python (pip)",
            },
          ],
      },
  }
---

# Hanzo Storage — Object Storage

`pip install boto3`

S3-compatible object storage for files, model artifacts, datasets, and media.
Served by [hanzoai/s3](https://github.com/hanzoai/s3) — a SeaweedFS fork
(Apache-2.0) — with erasure coding, server-side encryption, and lifecycle rules.

The wire protocol is standard S3, so any S3 SDK works: `boto3` (Python),
`aws-sdk-go-v2` (Go), `@aws-sdk/client-s3` (JS/TS). Use those. For the small
Hanzo-branded Python surface see the `hanzo-s3` skill.

## Quick Start

```python
import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT"],
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)

# Create bucket if absent — create_bucket errors when it already exists
try:
    s3.head_bucket(Bucket="models")
except ClientError:
    s3.create_bucket(Bucket="models")

# Upload file
s3.upload_file("/path/to/model.bin", "models", "llama3-8b.bin")

# Download file
s3.download_file("models", "llama3-8b.bin", "/tmp/model.bin")
```

Path-style addressing is required — a self-hosted endpoint has no per-bucket
DNS. Signature v4 is the only version to use.

## Upload Operations

```python
from io import BytesIO

# Upload from bytes
s3.put_object(Bucket="bucket", Key="path/file.txt", Body=b"file content here")

# Upload from a stream
s3.put_object(Bucket="bucket", Key="path/file.txt", Body=BytesIO(b"file content here"))

# Upload with content type and user metadata.
# boto3 prefixes Metadata keys with `x-amz-meta-` on the wire.
s3.upload_file(
    "/local/image.png", "bucket", "image.png",
    ExtraArgs={"ContentType": "image/png", "Metadata": {"project": "hanzo-bot"}},
)

# Multipart is automatic in upload_file for large objects
s3.upload_file("/path/to/50gb-model.bin", "models", "large-model.bin")
```

## Download Operations

```python
# Download to file (multipart-aware)
s3.download_file("bucket", "file.txt", "/local/file.txt")

# Stream download
response = s3.get_object(Bucket="bucket", Key="file.txt")
data = response["Body"].read()
response["Body"].close()
```

## List & Search

```python
# List objects (paginated — buckets can hold more than one page)
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="bucket", Prefix="models/"):
    for obj in page.get("Contents", []):
        print(f"{obj['Key']} ({obj['Size']} bytes)")

# List buckets
for bucket in s3.list_buckets()["Buckets"]:
    print(f"{bucket['Name']} (created: {bucket['CreationDate']})")
```

Pass `Delimiter="/"` to list one "directory" level instead of recursing;
prefixes then come back under `page["CommonPrefixes"]`.

## Delete Operations

```python
# Delete single object
s3.delete_object(Bucket="bucket", Key="path/file.txt")

# Delete up to 1000 objects per call
s3.delete_objects(
    Bucket="bucket",
    Delete={"Objects": [{"Key": "file1.txt"}, {"Key": "file2.txt"}]},
)
```

## Presigned URLs

```python
# Download URL (1 hour)
url = s3.generate_presigned_url(
    "get_object", Params={"Bucket": "bucket", "Key": "file.pdf"}, ExpiresIn=3600
)

# Upload URL (1 hour)
url = s3.generate_presigned_url(
    "put_object", Params={"Bucket": "bucket", "Key": "upload.txt"}, ExpiresIn=3600
)
```

## CLI

```bash
aws --endpoint-url http://localhost:9000 s3 ls
aws --endpoint-url http://localhost:9000 s3 cp ./model.bin s3://models/model.bin
```

## Run a local server

```bash
docker run -p 9000:9000 \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  -v s3_data:/data \
  ghcr.io/hanzoai/s3:v1.0.14
```

The image entrypoint is `s3`; the default command is
`server -s3 -s3.port=9000 -dir=/data`. The server reads `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` from the environment and creates a matching admin
identity. Published for `linux/amd64` only.

## Ports

| Port | Service |
| ---- | ------- |
| 9000 | S3 API  |
| 8888 | Filer   |
| 9333 | Master  |
| 8080 | Volume  |

There is no web console. Health: `GET :9000/healthz` (also `/readyz`, `/status`).

## Environment Variables

```bash
S3_ENDPOINT=http://localhost:9000        # in-cluster: http://s3.hanzo.svc:9000
AWS_ACCESS_KEY_ID=${S3_ACCESS_KEY}       # inject from KMS, never hardcode
AWS_SECRET_ACCESS_KEY=${S3_SECRET_KEY}
S3_BUCKET=default
```
