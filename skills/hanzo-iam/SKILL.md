---
name: hanzo-iam
description: "Authenticate users and manage identity with Hanzo IAM. OAuth2 flows, user CRUD, org management, JWT validation, and FastAPI integration via the hanzo-iam Python SDK."
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
              "package": "hanzoai-iam",
              "label": "Install Hanzo IAM SDK (pip)",
            },
          ],
      },
  }
---

# Hanzo IAM — Identity & Access Management

`pip install hanzoai-iam`

## Quick Start

```python
from hanzo_iam import IAMClient, IAMConfig

config = IAMConfig(
    endpoint="https://iam.hanzo.ai",
    client_id="your-client-id",
    client_secret="your-client-secret",
    org_name="HANZO"
)
client = IAMClient(config)
```

## Environment Variables

```bash
HANZO_IAM_ENDPOINT=https://iam.hanzo.ai   # IAM server URL
HANZO_IAM_CLIENT_ID=...                     # OAuth client ID
HANZO_IAM_CLIENT_SECRET=...                 # OAuth client secret
HANZO_IAM_ORG_NAME=HANZO                    # Organization name
HANZO_IAM_APP_NAME=your-app                 # Application name (optional)
HANZO_IAM_CERTIFICATE=path/to/cert.pem      # TLS cert (optional)
```

## Supported Organizations

| Org   | Endpoint                | Domain   |
| ----- | ----------------------- | -------- |
| HANZO | https://iam.hanzo.ai    | hanzo.id |
| ZOO   | https://iam.zoo.dev     | zoo.id   |
| LUX   | https://iam.lux.network | lux.id   |
| PARS  | https://iam.pars.dev    | pars.id  |

## OAuth2 Authorization Code Flow

```python
# Step 1: Get authorization URL (redirect user here)
auth_url = client.get_auth_url(
    redirect_uri="https://yourapp.com/callback",
    state="random-state-string",
    scope="openid profile email"
)

# Step 2: Exchange code for tokens (after callback)
tokens = client.get_token(code="auth-code-from-callback")
# tokens.access_token, tokens.refresh_token, tokens.expires_in

# Step 3: Get user info
user = client.get_user_info(access_token=tokens.access_token)
# user.id, user.name, user.email, user.avatar, user.owner (org)
```

## Client Credentials Flow (Service-to-Service)

```python
tokens = client.get_client_credentials_token()
# Use tokens.access_token for API calls between services
```

## Refresh Token

```python
new_tokens = client.refresh_token(refresh_token=tokens.refresh_token)
```

## User Management

```python
# List users
users = client.get_users()

# Get user by ID
user = client.get_user(user_id="user-123")

# Create user
user = client.create_user(
    name="Jane Doe",
    email="jane@example.com",
    password="secure-password"
)

# Update user
client.update_user(user_id="user-123", name="Jane Smith")

# Delete user
client.delete_user(user_id="user-123")
```

## Organization Management

```python
orgs = client.get_organizations()
org = client.get_organization(name="HANZO")
```

## JWT Validation

```python
claims = client.parse_jwt(token=access_token)
# claims["sub"], claims["iss"], claims["exp"], etc.
```

## FastAPI Integration

```python
from fastapi import FastAPI, Depends
from hanzo_iam.fastapi import IAMAuth
from hanzo_iam import User

app = FastAPI()
auth = IAMAuth(config)

@app.get("/protected")
async def protected(user: User = Depends(auth.require_user)):
    return {"user": user.name, "org": user.owner}

@app.get("/optional")
async def optional(user: User | None = Depends(auth.optional_user)):
    return {"user": user.name if user else "anonymous"}
```

## Async Client

```python
from hanzo_iam import AsyncIAMClient

async_client = AsyncIAMClient(config)
user = await async_client.get_user_info(access_token=token)
```
