---
name: hanzo-cli
description: "Hanzo unified CLI for IAM login, KMS secrets, PaaS deployments, and agent management. Single command-line interface for all Hanzo services."
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
              "package": "hanzo-cli",
              "label": "Install Hanzo CLI (pip)",
            },
          ],
      },
  }
---

# Hanzo CLI — Unified Command Line

`pip install hanzo-cli`

Single CLI for authenticating, managing secrets, deploying apps, and running agents.

## Quick Start

```bash
# Login to Hanzo (opens browser for OAuth)
hanzo login

# Check who you are
hanzo whoami

# Logout
hanzo logout
```

## Authentication

```bash
# Browser-based OAuth login (default)
hanzo login

# Password-based login
hanzo login --password

# Login to specific org
hanzo login --org=my-org --app=my-app
```

Token stored at `~/.hanzo/auth/token.json`.

## IAM Commands

```bash
# Show current user info
hanzo whoami

# List users in org
hanzo iam users

# List organizations
hanzo iam orgs

# List roles
hanzo iam roles
```

## KMS Commands

```bash
# List secrets
hanzo kms list

# Get a secret
hanzo kms get DATABASE_URL

# Set a secret
hanzo kms set DATABASE_URL "postgresql://..."

# Delete a secret
hanzo kms delete OLD_SECRET
```

## PaaS Commands

```bash
# List projects
hanzo paas projects --org=hanzo

# List environments
hanzo paas env --org=hanzo --project=my-app

# List deployments
hanzo paas deployments --org=hanzo --project=my-app --env=prod

# View logs
hanzo paas logs --org=hanzo --project=my-app --env=prod --container=web

# Redeploy
hanzo paas redeploy --org=hanzo --project=my-app --env=prod --container=web

# Shortcut
hanzo deploy
```

## Bot Commands

```bash
# Bot management
hanzo bot status
hanzo bot start
hanzo bot stop
```

## Environment Variables

```bash
HANZO_IAM_URL=https://hanzo.id            # IAM endpoint
HANZO_IAM_ORG=hanzo                        # Default organization
HANZO_IAM_APP=app-hanzo                    # Default application
HANZO_IAM_CLIENT_ID=...                    # OAuth client ID
HANZO_IAM_CLIENT_SECRET=...                # OAuth client secret
HANZO_KMS_URL=https://kms.hanzo.ai         # KMS endpoint
```
