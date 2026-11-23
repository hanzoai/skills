---
name: hanzo-s3
description: "Manage object storage with Hanzo S3. List, stat, delete, and presign objects in S3-compatible buckets via the hanzo-s3 Python SDK."
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
              "package": "hanzo-s3",
              "label": "Install Hanzo S3 SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo S3 — Object Storage

`pip install hanzo-s3`

Thin native adapter over `boto3` for [hanzoai/s3](https://github.com/hanzoai/s3)
— a SeaweedFS fork (Apache-2.0). It exposes a small, stable surface: buckets,
object listing, stat, delete, and presigned reads. Anything beyond that surface
(uploads, streaming reads, bulk delete) is plain `boto3` — see the
`hanzo-storage` skill.

## Quick Start

```python
from hanzo_s3 import S3Client

client = S3Client(
    "s3.hanzo.ai",              # in-cluster: "s3.hanzo.svc:9000" with secure=False
    access_key="YOUR-ACCESS-KEY",
    secret_key="YOUR-SECRET-KEY",
)
```

Constructor: `S3Client(endpoint, access_key, secret_key, secure=True, region="us-east-1")`.
A bare `host:port` endpoint gets an `http://`/`https://` scheme from `secure`.
Credentials come from KMS — never store them in plaintext.

## Bucket Operations

```python
# List all buckets -> list[Bucket(name, creation_date)]
for b in client.list_buckets():
    print(b.name, b.creation_date)

# Create bucket
client.make_bucket("my-bucket")

# Remove bucket (must be empty)
client.remove_bucket("my-bucket")
```

## List Objects

```python
# Returns list[Object(object_name, size, last_modified, is_dir)], fully paginated.
for obj in client.list_objects("my-bucket", prefix="data/", recursive=True):
    print(obj.object_name, obj.size, obj.last_modified)

# Non-recursive: one level, with pseudo-directories flagged is_dir=True
for obj in client.list_objects("my-bucket", prefix="data/"):
    print(obj.object_name, "dir" if obj.is_dir else obj.size)
```

## Stat & Delete

```python
# Stat(size, etag, content_type, last_modified)
stat = client.stat_object("my-bucket", "remote/path.txt")
print(stat.size, stat.content_type, stat.etag)

# Delete a single object
client.remove_object("my-bucket", "remote/path.txt")
```

## Presigned URLs

```python
from datetime import timedelta

url = client.presigned_get_object(
    "my-bucket", "file.pdf", expires=timedelta(hours=1)
)
```

## Errors

```python
from hanzo_s3 import S3Error   # botocore ClientError; aliased Error / S3Exception

try:
    client.stat_object("my-bucket", "missing.txt")
except S3Error as err:
    print(err.response["Error"]["Code"])
```

## Exports

| Name                                | Purpose                                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `S3Client` / `Client`               | `list_buckets`, `make_bucket`, `remove_bucket`, `list_objects`, `stat_object`, `remove_object`, `presigned_get_object` |
| `Bucket` / `Object` / `Stat`        | Frozen dataclass result types                                                                              |
| `S3Error` / `Error` / `S3Exception` | Errors (`botocore.exceptions.ClientError`)                                                                 |

## Uploads and downloads

The SDK deliberately has no upload/download methods. Use `boto3` against the
same endpoint — `upload_file`, `download_file`, `put_object`, `get_object`,
`delete_objects`. See the `hanzo-storage` skill.
