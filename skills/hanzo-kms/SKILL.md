---
name: hanzo-kms
description: "Manage secrets, keys, and encryption with Hanzo KMS. Create, read, update, delete secrets, inject env vars, and encrypt/decrypt data via the hanzo-kms Python SDK."
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
              "package": "hanzo-kms",
              "label": "Install Hanzo KMS SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo KMS — Key Management Service

`pip install hanzo-kms`

## Quick Start

```python
from hanzo_kms import KMSClient, ClientSettings, AuthenticationOptions, UniversalAuthMethod

client = KMSClient(ClientSettings(
    auth=AuthenticationOptions(
        universal_auth=UniversalAuthMethod(
            client_id="your-client-id",
            client_secret="your-client-secret"
        )
    )
))
```

Or auto-load from environment:

```python
client = KMSClient()  # Uses HANZO_KMS_* env vars
```

## Environment Variables

```bash
HANZO_KMS_URL=https://kms.hanzo.ai       # KMS server URL
HANZO_KMS_CLIENT_ID=...                    # Universal Auth client ID
HANZO_KMS_CLIENT_SECRET=...                # Universal Auth client secret
```

## Authentication Methods

### Universal Auth (recommended)

```python
client = KMSClient(ClientSettings(
    auth=AuthenticationOptions(
        universal_auth=UniversalAuthMethod(
            client_id="...",
            client_secret="..."
        )
    )
))
```

### Kubernetes Auth (for K8s workloads)

```python
from hanzo_kms import KubernetesAuthMethod

client = KMSClient(ClientSettings(
    auth=AuthenticationOptions(
        kubernetes=KubernetesAuthMethod(identity_id="...")
    )
))
```

### AWS IAM Auth

```python
from hanzo_kms import AWSIamAuthMethod

client = KMSClient(ClientSettings(
    auth=AuthenticationOptions(
        aws_iam=AWSIamAuthMethod(identity_id="...")
    )
))
```

## Secret Operations

### List Secrets

```python
secrets = client.list_secrets(
    project_id="my-project",
    environment="production",
    path="/"
)
for s in secrets:
    print(f"{s.secret_name} = {s.secret_value}")
```

### Get Secret

```python
secret = client.get_secret(
    project_id="my-project",
    environment="production",
    secret_name="DATABASE_URL"
)
print(secret.secret_value)
```

### Get Value (convenience)

```python
db_url = client.get_value("my-project", "production", "DATABASE_URL")
api_key = client.get_value("my-project", "production", "API_KEY", default="fallback")
```

### Create Secret

```python
client.create_secret(
    project_id="my-project",
    environment="production",
    secret_name="NEW_SECRET",
    secret_value="super-secret-value"
)
```

### Update Secret

```python
client.update_secret(
    project_id="my-project",
    environment="production",
    secret_name="NEW_SECRET",
    secret_value="updated-value"
)
```

### Delete Secret

```python
client.delete_secret(
    project_id="my-project",
    environment="production",
    secret_name="NEW_SECRET"
)
```

## Inject Secrets into Environment

```python
# Load all project secrets as environment variables
client.inject_env(
    project_id="my-project",
    environment="production",
    overwrite=False  # Don't overwrite existing env vars
)

import os
print(os.environ["DATABASE_URL"])  # Now available
```

## Key Management (via Core SDK)

```python
from hanzoai import Hanzo

hanzo = Hanzo(api_key="...")

# Create encryption key
key = hanzo.kms.create_key()

# Encrypt data
encrypted = hanzo.kms.encrypt(key_id=key.id, data="sensitive-data")

# Decrypt data
decrypted = hanzo.kms.decrypt(key_id=key.id, data=encrypted.ciphertext)

# Sign data
signature = hanzo.kms.sign(key_id=key.id, data="message")

# Verify signature
valid = hanzo.kms.verify(key_id=key.id, data="message", signature=signature.sig)
```

## Compatibility

Works with:

- Hanzo KMS (`kms.hanzo.ai`)
- Lux KMS (`kms.lux.network`)
- Infisical-compatible servers

Also aliased as `InfisicalClient` for compatibility:

```python
from hanzo_kms import InfisicalClient  # Same as KMSClient
```
